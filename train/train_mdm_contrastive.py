"""
train_mdm_contrastive.py

Contrastive Learning을 적용한 MDM 학습 entry point.
기존 train/train_mdm.py를 수정하지 않고 별도 스크립트로 작성.

사용법:
    # 처음 학습
    python -m train.train_mdm_contrastive \
        --save_dir save/humanml_contrastive \
        --dataset humanml \
        --flipped_motion_dir dataset/HumanML3D_flipped/new_joint_vecs \
        --lambda_contrastive 0.1 \
        --contrastive_margin 0.05

    # WandB 로깅
    python -m train.train_mdm_contrastive \
        --save_dir save/humanml_contrastive \
        --train_platform_type WandBPlatform \
        --wandb_project my_project \
        --wandb_entity my_entity \
        ...

    # 체크포인트 재개 (save_dir에 model*.pt가 있으면 자동 탐지)
    python -m train.train_mdm_contrastive \
        --save_dir save/humanml_contrastive \   # 기존 save_dir 그대로
        ...

변경 사항 (기존 train_mdm.py 대비):
  - DataLoader : get_data_contrastive.get_contrastive_dataset_loader 사용
  - Diffusion  : GaussianDiffusionContrastive (lambda_contrastive, contrastive_margin)
  - TrainLoop  : 기존 TrainLoop 그대로 재사용
  - WandB      : --wandb_project / --wandb_entity 로 설정 가능
  - Resume     : save_dir 내 최신 model*.pt 자동 탐지 후 이어서 학습
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
from train.training_loop import TrainLoop
from train.train_platforms import (WandBPlatform, ClearmlPlatform,
                                    TensorboardPlatform, NoPlatform)
from utils.model_util import create_gaussian_diffusion
from diffusion.gaussian_diffusion_contrastive import GaussianDiffusionContrastive
from data_loaders.get_data_contrastive import get_contrastive_dataset_loader
from diffusion.respace import SpacedDiffusion, space_timesteps
from diffusion import gaussian_diffusion as gd

import torch
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True


# ─────────────────────────────────────────────────────────────────────────────
# WandB 플랫폼 (project / entity 설정 가능 버전)
# ─────────────────────────────────────────────────────────────────────────────

class ContrastiveWandBPlatform(WandBPlatform):
    """
    WandBPlatform을 상속해 project / entity를 외부에서 주입할 수 있게 한다.
    같은 save_dir로 재실행하면 run id(= save_dir 폴더명)가 동일하므로
    WandB가 자동으로 이어서 기록한다 (resume='allow').
    """
    def __init__(self, save_dir, wandb_project='motion_diffusion_contrastive',
                 wandb_entity=None, config=None):
        import os as _os
        import glob as _glob
        # TrainPlatform.__init__ 호출 (self.name 설정)
        self.path, name = _os.path.split(save_dir)
        self.name = name

        import wandb
        self.wandb = wandb
        self.wandb.init(
            project=wandb_project,
            entity=wandb_entity,           # None이면 WandB 기본 entity 사용
            name=self.name,
            id=self.name,                  # 동일 run ID → resume 연결
            resume='allow',
            save_code=True,
            config=config,
        )


# ─────────────────────────────────────────────────────────────────────────────
# 인자 추가
# ─────────────────────────────────────────────────────────────────────────────

def add_contrastive_args(parser):
    """기존 train_args parser에 contrastive / wandb 전용 인자 추가."""
    group = parser.add_argument_group('contrastive')
    group.add_argument(
        '--flipped_motion_dir',
        default='dataset/HumanML3D_flipped/new_joint_vecs',
        type=str,
        help='flip된 모션이 저장된 디렉토리',
    )
    group.add_argument(
        '--lambda_contrastive',
        default=0.1,
        type=float,
        help='Contrastive loss 가중치 (0이면 비활성화)',
    )
    group.add_argument(
        '--contrastive_margin',
        default=0.05,
        type=float,
        help='Triplet margin: max(0, d_pos - d_neg + margin)',
    )
    group.add_argument(
        '--wandb_project',
        default='motion_diffusion_contrastive',
        type=str,
        help='WandB project 이름 (--train_platform_type WandBPlatform 일 때 사용)',
    )
    group.add_argument(
        '--wandb_entity',
        default=None,
        type=str,
        help='WandB entity (팀/유저). None이면 WandB 기본값 사용.',
    )
    return parser


# ─────────────────────────────────────────────────────────────────────────────
# Diffusion 생성
# ─────────────────────────────────────────────────────────────────────────────

def create_contrastive_diffusion(args):
    """GaussianDiffusionContrastive + SpacedDiffusion 인스턴스를 생성한다."""
    steps              = args.diffusion_steps
    betas              = gd.get_named_beta_schedule(args.noise_schedule, steps, scale_betas=1.)
    timestep_respacing = [steps]
    lambda_target_loc  = getattr(args, 'lambda_target_loc', 0.)

    class ContrastiveSpacedDiffusion(GaussianDiffusionContrastive, SpacedDiffusion):
        pass

    return ContrastiveSpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=gd.ModelMeanType.START_X,
        model_var_type=(
            gd.ModelVarType.FIXED_SMALL if args.sigma_small
            else gd.ModelVarType.FIXED_LARGE
        ),
        loss_type=gd.LossType.MSE,
        rescale_timesteps=False,
        lambda_vel=args.lambda_vel,
        lambda_rcxyz=args.lambda_rcxyz,
        lambda_fc=args.lambda_fc,
        lambda_target_loc=lambda_target_loc,
        lambda_contrastive=args.lambda_contrastive,
        contrastive_margin=args.contrastive_margin,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 체크포인트 탐지
# ─────────────────────────────────────────────────────────────────────────────

def find_latest_checkpoint(save_dir):
    """
    save_dir 내 model{N}.pt 파일 중 가장 높은 step의 경로를 반환.
    없으면 None.
    """
    pattern = os.path.join(save_dir, 'model*.pt')
    candidates = {}
    for path in glob.glob(pattern):
        fname = os.path.basename(path)
        m = re.match(r'model(\d+)\.pt$', fname)
        if m:
            candidates[int(m.group(1))] = path
    if not candidates:
        return None
    latest_step = max(candidates)
    return candidates[latest_step], latest_step


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # ── 인자 파싱 ────────────────────────────────────────────────────────────
    from argparse import ArgumentParser
    parser = ArgumentParser()
    _parser_util.add_base_options(parser)
    _parser_util.add_data_options(parser)
    _parser_util.add_model_options(parser)
    _parser_util.add_diffusion_options(parser)
    _parser_util.add_training_options(parser)
    add_contrastive_args(parser)
    args = _parser_util.apply_rules(parser.parse_args())
    args.overwrite = True  # save_dir가 존재해도 항상 허용 (resume 지원)

    fixseed(args.seed)

    # ── save_dir 준비 ────────────────────────────────────────────────────────
    if args.save_dir is None:
        raise FileNotFoundError('--save_dir 를 지정해 주세요.')
    os.makedirs(args.save_dir, exist_ok=True)

    # ── 체크포인트 자동 탐지 (resume) ────────────────────────────────────────
    # TrainLoop._load_and_sync_parameters() 가 find_resume_checkpoint() 를 먼저
    # 호출하므로, resume_checkpoint 를 명시하지 않아도 save_dir 내 최신 체크포인트를
    # 자동으로 찾아 이어서 학습한다.
    ckpt_result = find_latest_checkpoint(args.save_dir)
    if ckpt_result is not None and not args.resume_checkpoint:
        latest_ckpt, latest_step = ckpt_result
        print(f'[Resume] save_dir에서 체크포인트 발견 → step {latest_step:,} 부터 재개')
        print(f'         {latest_ckpt}')
        # TrainLoop.find_resume_checkpoint()가 동일하게 탐지하므로 별도 세팅 불필요
    else:
        print('[Resume] 이전 체크포인트 없음 → 처음부터 학습')

    # ── 학습 플랫폼 ──────────────────────────────────────────────────────────
    if args.train_platform_type == 'WandBPlatform':
        train_platform = ContrastiveWandBPlatform(
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

    # ── args 저장 ─────────────────────────────────────────────────────────────
    log_path = os.path.join(args.save_dir, 'train.log')
    with open(os.path.join(args.save_dir, 'args.json'), 'w') as fw:
        json.dump(vars(args), fw, indent=4, sort_keys=True)

    dist_util.setup_dist(args.device)

    # ── 데이터 로더 ──────────────────────────────────────────────────────────
    print('Creating contrastive data loader...')
    flipped_dir = args.flipped_motion_dir
    if not os.path.isabs(flipped_dir):
        flipped_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            flipped_dir,
        )

    data = get_contrastive_dataset_loader(
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

    # ── 모델 & Diffusion ──────────────────────────────────────────────────────
    print('Creating model and contrastive diffusion...')
    from utils.model_util import get_model_args
    from model.mdm import MDM
    model = MDM(**get_model_args(args, data))
    diffusion = create_contrastive_diffusion(args)

    model.to(dist_util.dev())
    model.rot2xyz.smpl_model.eval()

    print(f'Total params: {sum(p.numel() for p in model.parameters_wo_clip()) / 1e6:.2f}M')
    print(f'lambda_contrastive={args.lambda_contrastive}, '
          f'contrastive_margin={args.contrastive_margin}')

    # ── 학습 루프 ─────────────────────────────────────────────────────────────
    print('Training...')
    try:
        TrainLoop(args, train_platform, model, diffusion, data).run_loop()
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
