"""
train_mdm_contrastive_v2.py

train_mdm_contrastive.py (v1) 대비 변경:
  - Model    : MDM 또는 MDMWithLRAdapter (--use_lr_adapter 플래그로 선택)
  - Dataset  : ContrastiveHumanML3DV2 (flip 엔트리를 정식 훈련 데이터로 포함)
  - Diffusion: GaussianDiffusionContrastiveV2
               - L_text bidirectional triplet
               - Curriculum lambda (--lambda_warmup_steps)
  - Weight decay: --weight_decay 0.01 권장

사용법 (LR Adapter 없이, curriculum lambda, weight decay):
    python -m train.train_mdm_contrastive_v2 \\
        --save_dir save/humanml_contrastive_v2 \\
        --dataset humanml \\
        --flipped_motion_dir dataset/HumanML3D_flipped/new_joint_vecs \\
        --lambda_contrastive 0.1 \\
        --contrastive_margin 0.05 \\
        --lambda_warmup_steps 10000 \\
        --weight_decay 0.01

사용법 (LR Adapter 포함):
    python -m train.train_mdm_contrastive_v2 \\
        ... (위와 동일) \\
        --use_lr_adapter \\
        --lr_adapter_hidden 128
"""

import os
import sys
import json
import traceback
import glob
import re

from utils.fixseed import fixseed
from utils import parser_util as _parser_util
from utils import dist_util
from train.training_loop import TrainLoop, log_loss_dict
from train.train_platforms import (WandBPlatform, ClearmlPlatform,
                                    TensorboardPlatform, NoPlatform)
from diffusion.gaussian_diffusion_contrastive_v2 import GaussianDiffusionContrastiveV2
from data_loaders.get_data_contrastive_v2 import get_contrastive_dataset_loader_v2
from diffusion.respace import SpacedDiffusion, space_timesteps
from diffusion import gaussian_diffusion as gd

import torch
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True


# ─────────────────────────────────────────────────────────────────────────────
# WandB 플랫폼
# ─────────────────────────────────────────────────────────────────────────────

class ContrastiveWandBPlatformV2(WandBPlatform):
    def __init__(self, save_dir, wandb_project='mdm_contrastive_v2',
                 wandb_entity=None, config=None):
        import os as _os
        self.path, name = _os.path.split(save_dir)
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
# 인자
# ─────────────────────────────────────────────────────────────────────────────

def add_contrastive_args_v2(parser):
    group = parser.add_argument_group('contrastive_v2')
    group.add_argument('--flipped_motion_dir',
                       default='dataset/HumanML3D_flipped/new_joint_vecs', type=str)
    group.add_argument('--lambda_contrastive',  default=0.1,   type=float)
    group.add_argument('--contrastive_margin',  default=0.05,  type=float)
    group.add_argument('--lambda_warmup_steps', default=10000, type=int,
                       help='이 step까지 lambda를 0→target으로 linear warmup. 0이면 비활성화.')
    group.add_argument('--use_lr_adapter',      action='store_true',
                       help='CLIP 뒤에 LRAdapter를 붙임 (기본값: False)')
    group.add_argument('--lr_adapter_hidden',   default=128,   type=int,
                       help='LRAdapter MLP 중간 차원 (--use_lr_adapter 일 때만 사용)')
    group.add_argument('--wandb_project', default='mdm_contrastive_v2', type=str)
    group.add_argument('--wandb_entity',  default=None, type=str)
    return parser


# ─────────────────────────────────────────────────────────────────────────────
# Diffusion
# ─────────────────────────────────────────────────────────────────────────────

def create_contrastive_diffusion_v2(args):
    steps             = args.diffusion_steps
    betas             = gd.get_named_beta_schedule(args.noise_schedule, steps, scale_betas=1.)
    lambda_target_loc = getattr(args, 'lambda_target_loc', 0.)

    class ContrastiveSpacedDiffusionV2(GaussianDiffusionContrastiveV2, SpacedDiffusion):
        pass

    return ContrastiveSpacedDiffusionV2(
        use_timesteps=space_timesteps(steps, [steps]),
        betas=betas,
        model_mean_type=gd.ModelMeanType.START_X,
        model_var_type=(gd.ModelVarType.FIXED_SMALL if args.sigma_small
                        else gd.ModelVarType.FIXED_LARGE),
        loss_type=gd.LossType.MSE,
        rescale_timesteps=False,
        lambda_vel=args.lambda_vel,
        lambda_rcxyz=args.lambda_rcxyz,
        lambda_fc=args.lambda_fc,
        lambda_target_loc=lambda_target_loc,
        lambda_contrastive=args.lambda_contrastive,
        contrastive_margin=args.contrastive_margin,
        lambda_warmup_steps=args.lambda_warmup_steps,
    )


# ─────────────────────────────────────────────────────────────────────────────
# TrainLoop 서브클래스: diffusion의 current_step을 매 step 동기화
# ─────────────────────────────────────────────────────────────────────────────

class ContrastiveTrainLoopV2(TrainLoop):
    """
    TrainLoop를 상속해 매 step마다 diffusion.current_step을 업데이트한다.
    GaussianDiffusionContrastiveV2의 curriculum lambda가 이 값을 참조한다.
    """
    def run_step(self, batch, cond):
        self.diffusion.current_step = self.total_step()
        super().run_step(batch, cond)


# ─────────────────────────────────────────────────────────────────────────────
# Resume 탐지
# ─────────────────────────────────────────────────────────────────────────────

def find_latest_checkpoint(save_dir):
    candidates = {}
    for path in glob.glob(os.path.join(save_dir, 'model*.pt')):
        m = re.match(r'model(\d+)\.pt$', os.path.basename(path))
        if m:
            candidates[int(m.group(1))] = path
    if not candidates:
        return None
    latest = max(candidates)
    return candidates[latest], latest


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
    add_contrastive_args_v2(parser)
    parser.set_defaults(num_steps=750_000)
    args = _parser_util.apply_rules(parser.parse_args())
    args.overwrite = True

    fixseed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    # ── Resume 탐지 ───────────────────────────────────────────────────
    ckpt_result = find_latest_checkpoint(args.save_dir)
    if ckpt_result and not args.resume_checkpoint:
        latest_ckpt, latest_step = ckpt_result
        print(f'[Resume] step {latest_step:,} 부터 재개: {latest_ckpt}')
    else:
        print('[Resume] 처음부터 학습')

    # ── 학습 플랫폼 ───────────────────────────────────────────────────
    if args.train_platform_type == 'WandBPlatform':
        train_platform = ContrastiveWandBPlatformV2(
            save_dir=args.save_dir,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            config=vars(args),
        )
    else:
        platform_cls = {'NoPlatform': NoPlatform,
                        'TensorboardPlatform': TensorboardPlatform,
                        'ClearmlPlatform': ClearmlPlatform}[args.train_platform_type]
        train_platform = platform_cls(args.save_dir)
    train_platform.report_args(args, name='Args')

    log_path = os.path.join(args.save_dir, 'train.log')
    with open(os.path.join(args.save_dir, 'args.json'), 'w') as fw:
        json.dump(vars(args), fw, indent=4, sort_keys=True)

    dist_util.setup_dist(args.device)

    # ── 데이터 로더 (v2) ─────────────────────────────────────────────
    print('Creating contrastive data loader v2...')
    flipped_dir = args.flipped_motion_dir
    if not os.path.isabs(flipped_dir):
        flipped_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            flipped_dir)

    data = get_contrastive_dataset_loader_v2(
        name=args.dataset,
        batch_size=args.batch_size,
        num_frames=args.num_frames,
        flipped_motion_dir=flipped_dir,
        split='train',
        hml_mode='train',
        fixed_len=args.pred_len + args.context_len,
        pred_len=args.pred_len,
        device=dist_util.dev(),
    )

    # ── 모델 생성 ─────────────────────────────────────────────────────
    from utils.model_util import get_model_args
    model_args = get_model_args(args, data)

    if args.use_lr_adapter:
        from model.mdm_v2 import MDMWithLRAdapter
        model = MDMWithLRAdapter(lr_adapter_hidden=args.lr_adapter_hidden, **model_args)
        print(f'Creating MDMWithLRAdapter (hidden={args.lr_adapter_hidden}) '
              f'+ contrastive diffusion v2...')
    else:
        from model.mdm import MDM
        model = MDM(**model_args)
        print('Creating MDM (no LR adapter) + contrastive diffusion v2...')

    diffusion = create_contrastive_diffusion_v2(args)

    model.to(dist_util.dev())
    model.rot2xyz.smpl_model.eval()

    total_params = sum(p.numel() for p in model.parameters_wo_clip()) / 1e6
    print(f'Total params (wo clip): {total_params:.2f}M')
    if args.use_lr_adapter:
        adapter_params = sum(p.numel() for p in model.lr_adapter.parameters()) / 1e6
        print(f'  LRAdapter: {adapter_params:.4f}M')
    print(f'lambda_contrastive={args.lambda_contrastive}, '
          f'margin={args.contrastive_margin}, '
          f'warmup_steps={args.lambda_warmup_steps}, '
          f'weight_decay={args.weight_decay}')

    # ── 학습 ──────────────────────────────────────────────────────────
    print('Training...')
    try:
        ContrastiveTrainLoopV2(args, train_platform, model, diffusion, data).run_loop()
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
