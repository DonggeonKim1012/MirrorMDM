"""
train/train_mdm_infonce.py

Standard MDM + InfoNCE alignment loss (baseline).

MDM CLIP text embedding과 x_0 prediction trick으로 예측된 모션을
in-batch contrastive loss로 정렬한다.

구조:
  proj: MLP(263 → 512 → 512) — 학습 가능, inference 시 불필요
  L_total = L_diffusion + lambda_infonce * L_infonce

  L_infonce = symmetric InfoNCE(
                  normalize(proj(mean_pool(x_0_pred))),
                  normalize(clip_text_emb)
              )

Usage:
    python -m train.train_mdm_infonce \\
        --save_dir save/humanml_infonce_baseline \\
        --dataset humanml \\
        --diffusion_steps 1000 --noise_schedule cosine \\
        --lambda_infonce 0.1 \\
        --infonce_temperature 0.07 \\
        --num_steps 200000 \\
        --train_platform_type WandBPlatform \\
        --wandb_project mdm_infonce
"""

import os
import glob
import re
import json
import traceback
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

import clip as clip_lib

from utils.fixseed import fixseed
from utils import parser_util as _parser_util
from utils import dist_util
from utils.model_util import create_model_and_diffusion
from train.training_loop import TrainLoop, log_loss_dict
from train.train_platforms import (WandBPlatform, TensorboardPlatform,
                                   ClearmlPlatform, NoPlatform)
from data_loaders.get_data import get_dataset_loader
from diffusion import logger
from diffusion.resample import LossAwareSampler

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True


# ─────────────────────────────────────────────────────────────────────────────
# WandB platform
# ─────────────────────────────────────────────────────────────────────────────

class InfoNCEWandBPlatform(WandBPlatform):
    def __init__(self, save_dir, wandb_project='mdm_infonce',
                 wandb_entity=None, config=None):
        self.path, name = os.path.split(save_dir)
        self.name = name
        import wandb
        self.wandb = wandb
        self.wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=self.name,
            id=self.name,
            resume='allow',
            save_code=True,
            config=config,
        )


# ─────────────────────────────────────────────────────────────────────────────
# CLI args
# ─────────────────────────────────────────────────────────────────────────────

def add_infonce_args(parser):
    g = parser.add_argument_group('infonce')
    g.add_argument('--lambda_infonce',      type=float, default=0.1,
                   help='InfoNCE loss weight')
    g.add_argument('--infonce_temperature', type=float, default=0.07,
                   help='Softmax temperature for InfoNCE (CLIP default: 0.07)')
    g.add_argument('--infonce_t_min',       type=int,   default=500,
                   help='Min timestep for x_0 prediction trick. '
                        'Clamped to [T//2, T-1] automatically.')
    g.add_argument('--proj_hidden_dim',     type=int,   default=512,
                   help='Hidden dim of projection MLP (motion → CLIP space)')
    g.add_argument('--wandb_project', type=str, default='mdm_infonce')
    g.add_argument('--wandb_entity',  type=str, default=None)
    return parser


# ─────────────────────────────────────────────────────────────────────────────
# InfoNCE TrainLoop
# ─────────────────────────────────────────────────────────────────────────────

class InfoNCETrainLoop(TrainLoop):
    """
    Standard TrainLoop + InfoNCE alignment loss.

    추가 컴포넌트:
      self.proj       — Motion feature → CLIP space projection MLP
      self.proj_opt   — proj 전용 AdamW optimizer
    """

    def __init__(self, args, train_platform, model, diffusion, data):
        super().__init__(args, train_platform, model, diffusion, data)

        self.lambda_infonce      = args.lambda_infonce
        self.infonce_temperature = args.infonce_temperature
        self.infonce_t_min       = args.infonce_t_min

        # CLIP output dim (text embedding dim)
        clip_dim   = model.clip_model.ln_final.weight.shape[0]  # 512
        motion_dim = model.njoints * model.nfeats               # 263

        # Projection head: motion features → CLIP space
        self.proj = nn.Sequential(
            nn.Linear(motion_dim, args.proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(args.proj_hidden_dim, clip_dim),
        ).to(self.device)

        # Separate optimizer for proj (not managed by mp_trainer)
        self.proj_opt = AdamW(self.proj.parameters(), lr=args.lr,
                              weight_decay=args.weight_decay)

        # Resume proj if checkpoint exists
        self._load_proj_checkpoint()

        print(f'[InfoNCE] lambda={self.lambda_infonce}, '
              f'temperature={self.infonce_temperature}, '
              f't_min={self.infonce_t_min}, '
              f'proj: {motion_dim}→{args.proj_hidden_dim}→{clip_dim}')

    # ── projection head checkpoint ────────────────────────────────────────

    def _proj_ckpt_path(self, step):
        return os.path.join(self.save_dir, f'proj{step:09d}.pt')

    def _load_proj_checkpoint(self):
        """Resume step에 맞는 proj 체크포인트 로드."""
        if self.resume_step == 0:
            return
        path = self._proj_ckpt_path(self.resume_step)
        if os.path.exists(path):
            state = torch.load(path, map_location=self.device)
            self.proj.load_state_dict(state['proj'])
            self.proj_opt.load_state_dict(state['proj_opt'])
            print(f'[InfoNCE] proj resumed from step {self.resume_step}')
        else:
            print(f'[InfoNCE] proj checkpoint not found at {path}, starting fresh')

    # ── CLIP text embedding ───────────────────────────────────────────────

    @torch.no_grad()
    def _get_clip_text_emb(self, captions):
        """
        Returns L2-normalized CLIP text embeddings: (B, 512).
        Uses model's already-loaded clip_model — no extra memory.
        """
        tokens = clip_lib.tokenize(captions, truncate=True).to(self.device)
        emb    = self.model.clip_model.encode_text(tokens).float()  # (B, 512)
        return F.normalize(emb, dim=-1)

    # ── x_0 prediction + motion pool ─────────────────────────────────────

    def _predict_and_pool(self, captions, m_lens):
        """
        x_0 prediction trick → mask-weighted mean pool → (B, motion_dim).
        Gradient flows through model output.
        """
        B          = len(captions)
        max_frames = self.data.dataset.opt.max_motion_length
        T          = self.diffusion.num_timesteps

        t_min = min(self.infonce_t_min, T - 1)
        t_min = max(t_min, T // 2)
        t     = torch.randint(t_min, T, (B,), device=self.device)

        x_t  = torch.randn(B, self.model.njoints, self.model.nfeats,
                           max_frames, device=self.device)

        mask = torch.zeros(B, 1, 1, max_frames, device=self.device)
        for i, l in enumerate(m_lens.tolist()):
            mask[i, 0, 0, :min(int(l), max_frames)] = 1.0

        cond_y = {
            'text':    captions,
            'lengths': m_lens,
            'mask':    mask,
            'scale':   torch.ones(B, device=self.device),
        }
        t_scaled = self.diffusion._scale_timesteps(t)
        x_0_pred = self.model(x_t, t_scaled, **{'y': cond_y})  # (B, J, 1, T)

        # (B, J, 1, T) → (B, T, J)
        x_0_motion = x_0_pred.permute(0, 2, 3, 1).squeeze(1)   # (B, T, 263)

        # mask-weighted mean pool → (B, 263)
        mask_2d  = mask.squeeze(1).squeeze(1)                   # (B, T)
        n_valid  = mask_2d.sum(dim=1, keepdim=True).clamp(min=1)
        pooled   = (x_0_motion * mask_2d.unsqueeze(-1)).sum(1) / n_valid  # (B, 263)

        return pooled

    # ── InfoNCE loss ──────────────────────────────────────────────────────

    def _infonce_loss(self, motion_emb, text_emb):
        """
        Symmetric InfoNCE (bidirectional cross-entropy).
        motion_emb, text_emb: (B, D) L2-normalized.
        """
        B   = motion_emb.shape[0]
        sim = motion_emb @ text_emb.T / self.infonce_temperature  # (B, B)
        labels = torch.arange(B, device=self.device)
        L = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2
        return L

    # ── forward_backward ──────────────────────────────────────────────────

    def forward_backward(self, batch, cond):
        self.mp_trainer.zero_grad()
        self.proj_opt.zero_grad()

        micro      = batch
        micro_cond = cond
        t, weights = self.schedule_sampler.sample(micro.shape[0], self.device)

        # ── 1. Standard diffusion loss ────────────────────────────────
        losses = self.diffusion.training_losses(
            self.ddp_model, micro, t,
            model_kwargs=micro_cond,
            dataset=self.data.dataset,
        )
        if isinstance(self.schedule_sampler, LossAwareSampler):
            self.schedule_sampler.update_with_local_losses(
                t, losses['loss'].detach())

        L_diffusion = (losses['loss'] * weights).mean()
        log_loss_dict(self.diffusion, t,
                      {k: v * weights for k, v in losses.items()})

        # ── 2. InfoNCE loss ───────────────────────────────────────────
        captions = cond['y']['text']
        m_lens   = cond['y']['lengths'].long()

        motion_pool = self._predict_and_pool(captions, m_lens)       # (B, 263) w/ grad
        motion_emb  = F.normalize(self.proj(motion_pool), dim=-1)    # (B, 512) w/ grad
        text_emb    = self._get_clip_text_emb(captions)              # (B, 512) no grad

        L_infonce = self._infonce_loss(motion_emb, text_emb)

        # ── 3. Total loss + backward ──────────────────────────────────
        total_loss = L_diffusion + self.lambda_infonce * L_infonce
        self.mp_trainer.backward(total_loss)

        # ── 4. Logging ────────────────────────────────────────────────
        logger.logkv_mean('L_diffusion', L_diffusion.item())
        logger.logkv_mean('L_infonce',   L_infonce.item())
        logger.logkv_mean('L_total',     total_loss.item())

        # Mean diagonal similarity (higher = better alignment)
        with torch.no_grad():
            diag_sim = (motion_emb * text_emb).sum(dim=-1).mean()
        logger.logkv_mean('align_sim', diag_sim.item())

    # ── run_step: step both optimizers ───────────────────────────────────

    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)
        self.mp_trainer.optimize(self.opt)
        self.proj_opt.step()          # proj는 mp_trainer 밖에서 직접 step
        self.update_average_model()
        self._anneal_lr()
        self.log_step()

    # ── save: model + proj ────────────────────────────────────────────────

    def save(self):
        super().save()   # model + opt 저장
        proj_path = self._proj_ckpt_path(self.total_step())
        torch.save({
            'proj':     self.proj.state_dict(),
            'proj_opt': self.proj_opt.state_dict(),
        }, proj_path)
        logger.log(f'proj saved → {proj_path}')


# ─────────────────────────────────────────────────────────────────────────────
# Resume helper
# ─────────────────────────────────────────────────────────────────────────────

def find_latest_checkpoint(save_dir):
    candidates = {}
    for path in glob.glob(os.path.join(save_dir, 'model*.pt')):
        m = re.match(r'model(\d+)\.pt$', os.path.basename(path))
        if m:
            candidates[int(m.group(1))] = path
    return (max(candidates), candidates[max(candidates)]) if candidates else (None, None)


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    _parser_util.add_base_options(parser)
    _parser_util.add_data_options(parser)
    _parser_util.add_model_options(parser)
    _parser_util.add_diffusion_options(parser)
    _parser_util.add_training_options(parser)
    add_infonce_args(parser)
    parser.set_defaults(num_steps=200_000)
    args = _parser_util.apply_rules(parser.parse_args())
    args.overwrite = True

    fixseed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    with open(os.path.join(args.save_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4, sort_keys=True)

    # ── Resume 탐지 ───────────────────────────────────────────────────
    latest_step, latest_ckpt = find_latest_checkpoint(args.save_dir)
    if latest_ckpt and not args.resume_checkpoint:
        args.resume_checkpoint = latest_ckpt
        print(f'[Resume] step {latest_step:,} 부터 재개: {latest_ckpt}')
    else:
        print('[Start] Scratch부터 학습 (200,000 steps)')

    # ── 학습 플랫폼 ───────────────────────────────────────────────────
    if args.train_platform_type == 'WandBPlatform':
        train_platform = InfoNCEWandBPlatform(
            save_dir=args.save_dir,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            config=vars(args),
        )
    else:
        platform_cls = {
            'NoPlatform':        NoPlatform,
            'TensorboardPlatform': TensorboardPlatform,
            'ClearmlPlatform':   ClearmlPlatform,
        }[args.train_platform_type]
        train_platform = platform_cls(args.save_dir)
    train_platform.report_args(args, name='Args')

    dist_util.setup_dist(args.device)
    logger.configure()

    # ── 데이터 ────────────────────────────────────────────────────────
    print('Loading dataset...')
    data = get_dataset_loader(
        name=args.dataset,
        batch_size=args.batch_size,
        num_frames=args.num_frames,
        split='train',
        hml_mode='train',
    )

    # ── 모델 ──────────────────────────────────────────────────────────
    print('Creating model...')
    model, diffusion = create_model_and_diffusion(args, data)
    model.to(dist_util.dev())
    model.rot2xyz.smpl_model.eval()

    total_params = sum(p.numel() for p in model.parameters_wo_clip()) / 1e6
    print(f'Total params (wo clip): {total_params:.2f}M')

    # ── 학습 ──────────────────────────────────────────────────────────
    print('Training (InfoNCE baseline)...')
    log_path = os.path.join(args.save_dir, 'train.log')
    try:
        InfoNCETrainLoop(args, train_platform, model, diffusion, data).run_loop()
    except Exception:
        tb = traceback.format_exc()
        print(tb)
        with open(log_path, 'a') as f:
            f.write(tb)
        raise
    finally:
        train_platform.close()


if __name__ == '__main__':
    main()
