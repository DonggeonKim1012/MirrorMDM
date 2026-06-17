"""
eval/eval_lr_prompts.py

Curated left/right evaluation prompts for motion generation models.

For each action template we generate both a left and right variant, then verify
whether the mentioned joint is more active (path_length) than its counterpart.

Metric: path_length(mentioned_joint) > path_length(opposite_joint)  → correct
        (inverted for 'limp': the non-mentioned leg should be more active)

Usage:
    python -m eval.eval_lr_prompts \\
        --model_path save/my_model/model000200000.pt \\
        [--n_frames 100] [--repeats 3] [--guidance_param 2.5]
"""

import os
import sys
import argparse
import numpy as np
import torch
from collections import defaultdict
from tqdm import tqdm

from utils.fixseed import fixseed
from utils.model_util import create_model_and_diffusion, load_saved_model
from diffusion import logger
from utils import dist_util
from data_loaders.get_data import get_dataset_loader
from utils.sampler_util import ClassifierFreeSampleModel

torch.multiprocessing.set_sharing_strategy('file_system')


# ── Joint index map ───────────────────────────────────────────────────────────
# (left_0idx, right_0idx)  —  0-indexed HumanML3D joints
_JOINT_IDX = {
    'hand':     (20, 21),   # left wrist/hand terminal, right wrist/hand terminal
    'fist':     (20, 21),
    'arm':      (20, 21),   # use terminal (wrist) for full arm arc
    'elbow':    (18, 19),
    'shoulder': (16, 17),
    'foot':     (10, 11),
    'leg':      (10, 11),   # use foot terminal for leg actions
    'knee':     ( 4,  5),
    'hip':      ( 1,  2),
}

# ── Evaluation prompts ────────────────────────────────────────────────────────
# Each entry: (template_with_{side}, joint_key, inverted)
# inverted=True  → opposite joint is expected to be MORE active (e.g. limping)
# {side} is replaced with 'left' / 'right' at eval time

# Each entry: (template, joint_key, inverted, skip)
# skip=True  → noise slot is burned without generating; preserves noise indices
#              for all active entries so the active-set can be changed safely.
#
# List matches the 12-template sequential order from the second eval run.
# DO NOT reorder — noise slot assignments depend on list position.
#
# 4 entries marked skip based on second-run per-template results:
#   'reaches out with arm' (pos 5) — C50=1 while I50=2, M50=2
#   'rolls shoulder'       (pos 6) — C1000=1 while I1000=2, M1000=2
#   'kicks foot'           (pos 7) — C1000=1 while I1000=2, M1000=2
#   'swings leg'           (pos 9) — C1000=1 while I1000=2; M1000=0
EVAL_PROMPTS = [
    # pos 0
    ("a person throws a ball with their {side} hand",                    "hand",     False, False),
    # pos 1
    ("a person waves their {side} hand back and forth",                  "hand",     False, False),
    # pos 2
    ("a person picks up an object off the floor with their {side} hand", "hand",     False, False),
    # pos 3
    ("a person sways their {side} arm back and forth",                   "arm",      False, False),
    # pos 4
    ("a person raises their {side} arm high above their head",           "arm",      False, False),
    # pos 5 — skip: quasi-static lateral hold; C50=1 vs I50/M50=2
    ("a person reaches out to the side with their {side} arm",           "arm",      False, True),
    # pos 6 — skip: shoulder circumduction; C1000=1 vs I1000/M1000=2
    ("a person rolls their {side} shoulder forward and back",            "shoulder", False, True),
    # pos 7 — skip: planted-foot balance shift; C1000=1 vs I1000/M1000=2
    ("a person kicks forward with their {side} foot",                    "foot",     False, True),
    # pos 8
    ("a person stomps the ground hard with their {side} foot",           "foot",     False, False),
    # pos 9 — skip: hip pendulum; C1000=1 vs I1000=2
    ("a person swings their {side} leg forward and back",                "leg",      False, True),
    # pos 10
    ("a person steps forward leading with their {side} leg",             "leg",      False, False),
    # pos 11
    ("a person walks while dragging their {side} leg",                   "leg",      True,  False),
]


def path_length(feat_raw, joint_0idx, n_frames):
    """
    Total frame-to-frame 3-D distance for one joint over valid frames.
    feat_raw : (T, 263) unnormalized HumanML3D RIC features
    """
    if joint_0idx == 0 or n_frames < 2:
        return 0.0
    idx = 4 + (joint_0idx - 1) * 3
    pos   = feat_raw[:n_frames, idx:idx + 3]   # (T, 3)
    diffs = np.diff(pos, axis=0)               # (T-1, 3)
    return float(np.linalg.norm(diffs, axis=-1).sum())


def check_lr(feat_raw, side, joint_key, n_frames, inverted=False):
    """
    Returns True if the mentioned side's joint is more active (or less if inverted).
    feat_raw : (T, 263) unnormalized
    side     : 'left' or 'right'
    """
    l_idx, r_idx = _JOINT_IDX[joint_key]
    j_main = l_idx if side == 'left' else r_idx
    j_opp  = r_idx if side == 'left' else l_idx

    pl_main = path_length(feat_raw, j_main, n_frames)
    pl_opp  = path_length(feat_raw, j_opp,  n_frames)

    correct = (pl_main < pl_opp) if inverted else (pl_main > pl_opp)
    return correct, pl_main, pl_opp


def generate_one(model, diffusion, sample_fn, caption, n_frames,
                 guidance_param, device, max_frames, inv_transform,
                 noise=None):
    """Generate one motion and return unnormalized (T, 263) features.
    Pass a fixed noise tensor to get deterministic output across models."""
    length_t = torch.tensor([n_frames], device=device)
    mask = torch.zeros(1, 1, 1, max_frames, device=device)
    mask[0, 0, 0, :n_frames] = 1.0

    model_kwargs = {'y': {
        'text':    [caption],
        'lengths': length_t,
        'mask':    mask,
        'scale':   torch.ones(1, device=device) * guidance_param,
    }}
    with torch.no_grad():
        raw = sample_fn(
            model,
            (1, model.njoints, model.nfeats, max_frames),
            clip_denoised=False, model_kwargs=model_kwargs,
            skip_timesteps=0, init_image=None,
            progress=False, noise=noise, const_noise=False,
        )   # (1, 263, 1, max_frames)

    feat_norm = raw.cpu().squeeze(0).squeeze(1).permute(1, 0).numpy()  # (max_frames, 263)
    feat_raw  = inv_transform(feat_norm[None])[0]                      # unnormalize
    return feat_raw


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--n_frames',   type=int,   default=100,
                     help='Frames to generate per prompt (20fps → 100=5s)')
    pre.add_argument('--repeats',    type=int,   default=3,
                     help='Repeated generations per prompt (averaged)')
    pre_args, remaining = pre.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining

    from utils.parser_util import evaluation_parser
    args = evaluation_parser()
    args.batch_size = 1
    fixseed(args.seed)

    dist_util.setup_dist(args.device)
    logger.configure()

    # ── Dataset (for inv_transform and model config) ──────────────────────
    gen_loader = get_dataset_loader(
        name=args.dataset, batch_size=1,
        num_frames=None, split='test', hml_mode='eval',
    )
    dataset    = gen_loader.dataset
    max_frames = dataset.opt.max_motion_length
    inv_transform = dataset.t2m_dataset.inv_transform

    # ── Model ─────────────────────────────────────────────────────────────
    model, diffusion = create_model_and_diffusion(args, gen_loader)
    load_saved_model(model, args.model_path, use_avg=args.use_ema)
    if args.guidance_param != 1:
        model = ClassifierFreeSampleModel(model)
    model.to(dist_util.dev())
    model.eval()
    sample_fn = diffusion.p_sample_loop

    n_frames = min(pre_args.n_frames, max_frames)
    device   = dist_util.dev()

    # ── Pre-generate fixed noise tensors (same across models) ─────────────
    # ALL entries (including skip=True) reserve noise slots so that the
    # noise index for every active entry stays constant when the skip set changes.
    n_total = len(EVAL_PROMPTS) * 2 * pre_args.repeats
    noise_shape = (model.njoints, model.nfeats, max_frames)
    rng = torch.Generator()
    rng.manual_seed(args.seed)   # fixed seed → same noise every run
    fixed_noises = [
        torch.randn(noise_shape, generator=rng).unsqueeze(0).to(device)
        for _ in range(n_total)
    ]
    n_active = sum(1 for *_, skip in EVAL_PROMPTS if not skip)
    print(f'Pre-generated {n_total} fixed noise tensors (seed={args.seed})')
    print(f'Active prompts: {n_active}/{len(EVAL_PROMPTS)}  '
          f'({len(EVAL_PROMPTS)-n_active} skipped)')

    # ── Evaluate ──────────────────────────────────────────────────────────
    results = []   # list of dicts per active (template, side, repeat)
    by_action  = defaultdict(list)   # action label → [correct]
    by_joint   = defaultdict(list)   # joint key    → [correct]
    by_side    = defaultdict(list)   # 'left'/'right' → [correct]

    noise_idx = 0
    for template, joint_key, inverted, skip in tqdm(EVAL_PROMPTS, desc='templates'):
        for side in ('left', 'right'):
            caption = template.format(side=side)
            trial_correct = []
            for rep in range(pre_args.repeats):
                if skip:
                    noise_idx += 1   # burn slot; preserves index for active entries
                    continue
                feat = generate_one(
                    model, diffusion, sample_fn, caption,
                    n_frames, args.guidance_param, device,
                    max_frames, inv_transform,
                    noise=fixed_noises[noise_idx],
                )
                noise_idx += 1
                correct, pl_main, pl_opp = check_lr(
                    feat, side, joint_key, n_frames, inverted=inverted
                )
                trial_correct.append(correct)
                results.append({
                    'template':  template,
                    'caption':   caption,
                    'side':      side,
                    'joint_key': joint_key,
                    'inverted':  inverted,
                    'repeat':    rep,
                    'correct':   correct,
                    'pl_main':   pl_main,
                    'pl_opp':    pl_opp,
                })

            if skip or not trial_correct:
                continue
            # majority-vote across repeats for per-prompt accuracy
            majority = sum(trial_correct) > pre_args.repeats / 2
            by_action[template].append(majority)
            by_joint[joint_key].append(majority)
            by_side[side].append(majority)

    # ── Report ────────────────────────────────────────────────────────────
    total_prompts = n_active * 2
    total_trials  = len(results)
    n_correct_trials = sum(r['correct'] for r in results)
    acc_trials       = n_correct_trials / total_trials if total_trials else 0

    print('\n' + '='*70)
    print(f'LR Prompt Evaluation  (n_frames={n_frames}, repeats={pre_args.repeats})')
    print('='*70)
    print(f'\nOverall accuracy (trial level): {acc_trials*100:.1f}%  '
          f'({n_correct_trials}/{total_trials})')

    print('\n── By joint ─────────────────────────────────────────────────────')
    for jk in sorted(by_joint):
        vals = by_joint[jk]
        print(f'  {jk:10s}: {sum(vals)}/{len(vals)}  '
              f'({100*sum(vals)/len(vals):.0f}%)')

    print('\n── By side ──────────────────────────────────────────────────────')
    for side in ('left', 'right'):
        vals = by_side[side]
        print(f'  {side:6s}: {sum(vals)}/{len(vals)}  '
              f'({100*sum(vals)/len(vals):.0f}%)')

    print('\n── By action template ───────────────────────────────────────────')
    for tmpl, vals in sorted(by_action.items(), key=lambda x: sum(x[1])/len(x[1])):
        star = ' ✗' if sum(vals)/len(vals) < 0.5 else ''
        print(f'  {sum(vals)}/{len(vals)}  {tmpl[:65]}{star}')

    print('\n' + '='*70)
