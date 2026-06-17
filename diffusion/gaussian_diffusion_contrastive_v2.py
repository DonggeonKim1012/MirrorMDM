"""
gaussian_diffusion_contrastive_v2.py

v1 대비 변경:
  1. L_text — joint-specific triplet (self-limiting):
       캡션에서 언급된 신체 부위 양쪽 + root까지의 kinematic chain에 해당하는
       feature만 사용해 거리를 계산. 관계없는 관절의 noise를 제거.
       triplet 공식 (bounded, 이상 동작 방지):
           d_pos_joint      = per_sample_l2(x_pred_orig, x_orig, seq_mask, joint_feat_mask)
           d_pos_swap_joint = per_sample_l2(x_pred_swap, x_orig, seq_mask, joint_feat_mask)
           L_text = clamp(d_pos_joint - d_pos_swap_joint + margin, 0) * has_swap
       d_pos_swap_joint > d_pos_joint 이면 loss=0 → self-limiting.

  2. Curriculum lambda:
       effective_lambda = lambda_contrastive * min(1, current_step / warmup_steps)
       lambda_warmup_steps=0 이면 즉시 full lambda.
       current_step은 ContrastiveTrainLoopV2가 매 스텝 업데이트한다.
"""

import torch
from diffusion.gaussian_diffusion import ModelMeanType
from diffusion.gaussian_diffusion_contrastive import GaussianDiffusionContrastive
from diffusion.joint_mask_util import build_joint_feat_mask


class GaussianDiffusionContrastiveV2(GaussianDiffusionContrastive):
    """
    GaussianDiffusionContrastive (v1) 상속.
    L_text: joint-specific feature masking + v1 push-away 공식 복원.
    Curriculum lambda 추가.

    추가 init 인자:
        lambda_warmup_steps (int): 0이면 즉시 full lambda,
                                   >0이면 이 step까지 linear warmup.
    """

    def __init__(self, *, lambda_warmup_steps: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.lambda_warmup_steps = lambda_warmup_steps
        self.current_step = 0          # ContrastiveTrainLoopV2가 매 step 업데이트

    # ------------------------------------------------------------------
    @property
    def effective_lambda(self) -> float:
        """Curriculum에 따른 현재 effective lambda_contrastive."""
        if self.lambda_warmup_steps <= 0:
            return self.lambda_contrastive
        return self.lambda_contrastive * min(1.0, self.current_step / self.lambda_warmup_steps)

    # ------------------------------------------------------------------
    def training_losses(self, model, x_start, t,
                        model_kwargs=None, noise=None, dataset=None):
        import torch as th

        if noise is None:
            noise = th.randn_like(x_start)

        # ── 표준 diffusion loss ──────────────────────────────────────
        terms = super(GaussianDiffusionContrastive, self).training_losses(
            model, x_start, t,
            model_kwargs=model_kwargs,
            noise=noise,
            dataset=dataset,
        )

        eff_lambda = self.effective_lambda
        if eff_lambda <= 0.:
            return terms

        y    = (model_kwargs or {}).get('y', {})
        mask = y.get('mask')           # [bs, 1, 1, T]
        if mask is None:
            return terms

        has_flip = y.get('has_flip')
        has_swap = y.get('has_swap')
        if (has_flip is None or not has_flip.any()) and \
           (has_swap is None or not has_swap.any()):
            return terms

        if self.model_mean_type != ModelMeanType.START_X:
            return terms

        x_t      = self.q_sample(x_start, t, noise=noise)
        t_scaled = self._scale_timesteps(t)

        contra_loss = th.zeros(x_start.shape[0], device=x_start.device)

        # ── L_flip: 원본 텍스트 forward (gradient 유지, v1과 동일) ───
        if has_flip is not None and has_flip.any():
            flipped_motion = y['flipped_motion']

            model_output_pos = model(x_t, t_scaled, **model_kwargs)
            d_pos = self._per_sample_masked_l2(model_output_pos, x_start, mask)

            d_neg_flip = self._per_sample_masked_l2(
                model_output_pos, flipped_motion, mask)

            L_flip = th.clamp(d_pos - d_neg_flip + self.contrastive_margin, min=0.)
            L_flip = L_flip * has_flip.float()
            contra_loss = contra_loss + L_flip
            terms['L_flip'] = L_flip

        # ── L_text: joint-specific triplet (self-limiting) ───────────
        if has_swap is not None and has_swap.any():
            captions     = y.get('text', [''] * x_start.shape[0])
            swapped_text = y['swapped_text']

            swap_kwargs = dict(model_kwargs)
            swap_y      = dict(y)
            swap_y['text'] = swapped_text
            swap_y.pop('text_embed', None)   # swap 텍스트를 새로 인코딩
            swap_kwargs['y'] = swap_y

            # 캡션별 관련 관절 feature mask (양쪽 포함): [bs, 263, 1, 1]
            joint_feat_mask = build_joint_feat_mask(captions, device=x_start.device)

            # d_pos_joint: 원본 텍스트 예측과 원본 모션의 관련 관절 거리 (gradient 유지)
            # L_flip이 없을 때도 필요하므로 별도 forward pass
            if has_flip is not None and has_flip.any():
                # L_flip forward pass 결과 재사용
                d_pos_joint = self._per_sample_masked_l2_joint(
                    model_output_pos, x_start, mask, joint_feat_mask)
            else:
                model_output_pos = model(x_t, t_scaled, **model_kwargs)
                d_pos_joint = self._per_sample_masked_l2_joint(
                    model_output_pos, x_start, mask, joint_feat_mask)

            # d_pos_swap_joint: swap 텍스트 예측과 원본 모션의 관련 관절 거리
            with th.no_grad():
                model_output_swap = model(x_t, t_scaled, **swap_kwargs)
                d_pos_swap_joint  = self._per_sample_masked_l2_joint(
                    model_output_swap, x_start, mask, joint_feat_mask)

            # triplet: 올바른 텍스트 예측이 swap 예측보다 관련 관절에서 원본에 더 가까워야 함
            # d_pos_swap_joint > d_pos_joint 이면 loss=0 → self-limiting, 이상 동작 방지
            L_text = th.clamp(d_pos_joint - d_pos_swap_joint + self.contrastive_margin, min=0.)
            L_text = L_text * has_swap.float()
            contra_loss = contra_loss + L_text
            terms['L_text'] = L_text

        terms['contrastive']      = contra_loss
        terms['effective_lambda'] = torch.tensor(eff_lambda)
        terms['loss']             = terms['loss'] + eff_lambda * contra_loss

        return terms

    # ------------------------------------------------------------------
    @staticmethod
    def _per_sample_masked_l2_joint(a, b, seq_mask, feat_mask):
        """
        관절별 feature mask를 추가로 적용한 per-sample masked L2.

        Args:
            a, b      : [bs, 263, 1, T]
            seq_mask  : [bs,   1, 1, T]  (True = valid frame)
            feat_mask : [bs, 263, 1, 1]  (True = relevant feature dim)

        Returns:
            [bs] — 샘플별 평균 제곱 오차 (관련 관절 + 유효 프레임만)
        """
        diff_sq = (a - b) ** 2                                        # [bs, 263, 1, T]
        masked  = diff_sq * seq_mask.float() * feat_mask.float()      # [bs, 263, 1, T]
        summed  = masked.sum(dim=(1, 2, 3))                           # [bs]

        n_feats = feat_mask.squeeze(-1).squeeze(-1).sum(dim=1).float()  # [bs]
        n_valid = seq_mask.squeeze(1).squeeze(1).sum(dim=-1).float()    # [bs]
        return summed / (n_valid * n_feats + 1e-8)                    # [bs]
