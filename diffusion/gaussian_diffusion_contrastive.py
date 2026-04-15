"""
gaussian_diffusion_contrastive.py

GaussianDiffusion을 상속해 Bidirectional Contrastive Loss를 추가한다.
기존 gaussian_diffusion.py는 수정하지 않는다.

Loss 구성 (bidirectional triplet):

  1. Motion flip loss (L_flip):
       d_pos  = per_sample_mse(x_pred, x_orig,  mask)   ← gradient 있음 (부모 재사용)
       d_neg  = per_sample_mse(x_pred, x_flip,  mask)   ← no_grad (x_pred detach)
       L_flip = clamp(d_pos - d_neg + margin, 0) * has_flip

  2. Text swap loss (L_text):
       x_pred_swap = model(x_t, t, text=swapped_text)   ← no_grad
       d_pos_swap  = per_sample_mse(x_pred_swap, x_orig, mask)
       d_neg_swap  = per_sample_mse(x_pred_swap, x_flip, mask)  (flip available)
               or  per_sample_mse(x_pred_swap, x_orig, mask)    (no flip: penalize if too close)
       단순화: swap된 텍스트 예측이 원본과 멀어야 한다
       L_text = clamp(margin - d_pos_swap, 0) * has_swap        (push-away term)

  L_total = L_diffusion + lambda_c * (L_flip + L_text)

Gradient 전략:
  - d_pos는 부모 terms["rot_mse"]를 재사용해 extra forward pass 없이 gradient를 유지.
  - d_neg, d_neg_swap 계산은 torch.no_grad() 로 감싸 추가 메모리 없이 처리.
  - L_flip의 gradient: d_pos 항이 x_pred에 대해 미분 가능 → 정상적으로 전파.
  - L_text는 push-away term만 계산 (gradient 불필요, 상수 penalty 역할).
"""

import copy
import torch
import torch.nn.functional as F
from diffusion.gaussian_diffusion import GaussianDiffusion, LossType, ModelMeanType


class GaussianDiffusionContrastive(GaussianDiffusion):
    """
    GaussianDiffusion에 Bidirectional Contrastive Loss를 추가한 서브클래스.

    추가 init 인자:
        lambda_contrastive (float): contrastive loss 가중치 (default 0.1)
        contrastive_margin  (float): triplet margin (default 0.05)
    """

    def __init__(self, *, lambda_contrastive=0.1, contrastive_margin=0.05, **kwargs):
        super().__init__(**kwargs)
        self.lambda_contrastive = lambda_contrastive
        self.contrastive_margin = contrastive_margin

    # ------------------------------------------------------------------
    def training_losses(self, model, x_start, t,
                        model_kwargs=None, noise=None, dataset=None):
        """
        부모의 training_losses를 실행한 뒤 bidirectional contrastive term을 추가한다.

        Noise를 미리 샘플링해 부모와 동일한 x_t를 contrastive pass에서도 사용.

        cond['y'] 키:
          - 'flipped_motion' : [bs, 263, 1, T]  — motion flip negative
          - 'has_flip'       : [bs] bool
          - 'swapped_text'   : List[str]         — text swap negative
          - 'has_swap'       : [bs] bool
        """
        import torch as th

        # noise를 미리 고정 → 부모와 동일한 x_t 재현 가능
        if noise is None:
            noise = th.randn_like(x_start)

        # ── 표준 diffusion loss 계산 ─────────────────────────────────
        terms = super().training_losses(
            model, x_start, t,
            model_kwargs=model_kwargs,
            noise=noise,
            dataset=dataset,
        )

        if self.lambda_contrastive <= 0.:
            return terms

        y = (model_kwargs or {}).get('y', {})
        mask = y.get('mask')           # [bs, 1, 1, T]

        if mask is None:
            return terms

        has_flip = y.get('has_flip')   # [bs] bool or None
        has_swap = y.get('has_swap')   # [bs] bool or None

        # 둘 다 없으면 skip
        if (has_flip is None or not has_flip.any()) and \
           (has_swap is None or not has_swap.any()):
            return terms

        # model_mean_type 체크
        if self.model_mean_type != ModelMeanType.START_X:
            return terms

        # ── x_t 재현 (noise가 고정돼 있으므로 부모와 동일) ──────────
        x_t = self.q_sample(x_start, t, noise=noise)
        t_scaled = self._scale_timesteps(t)

        # ── d_pos : 부모 forward pass의 rot_mse를 per-sample로 재계산 ──
        # terms["rot_mse"]는 배치 평균 스칼라이므로 per-sample 계산이 필요.
        # 단, gradient를 살리기 위해 x_pred를 새로 얻어야 한다.
        # forward pass 1회 (gradient 유지).
        model_output_pos = model(x_t, t_scaled, **model_kwargs)  # [bs, 263, 1, T]
        d_pos = self._per_sample_masked_l2(model_output_pos, x_start, mask)  # [bs]

        contra_loss = th.zeros_like(d_pos)  # [bs]

        # ── L_flip : motion flip negative ───────────────────────────
        if has_flip is not None and has_flip.any():
            flipped_motion = y['flipped_motion']  # [bs, 263, 1, T]

            with th.no_grad():
                d_neg_flip = self._per_sample_masked_l2(
                    model_output_pos.detach(), flipped_motion, mask)  # [bs]

            L_flip = th.clamp(d_pos - d_neg_flip + self.contrastive_margin, min=0.0)
            L_flip = L_flip * has_flip.float()
            contra_loss = contra_loss + L_flip
            terms['L_flip'] = L_flip

        # ── L_text : text swap negative (push-away term) ────────────
        if has_swap is not None and has_swap.any():
            swapped_text = y['swapped_text']   # List[str], len=bs

            # swapped_text로 conditioning한 model_kwargs 생성
            swap_kwargs = copy.copy(model_kwargs)
            swap_y = dict(y)
            swap_y['text'] = swapped_text
            swap_kwargs = dict(model_kwargs)
            swap_kwargs['y'] = swap_y

            with th.no_grad():
                model_output_swap = model(x_t, t_scaled, **swap_kwargs)
                # swap된 텍스트 조건에서 예측된 x_0가 원본과 멀어야 한다
                d_pos_swap = self._per_sample_masked_l2(
                    model_output_swap, x_start, mask)  # [bs]

            # push-away: swap 예측이 원본에 너무 가까우면 penalty
            L_text = th.clamp(self.contrastive_margin - d_pos_swap, min=0.0)
            L_text = L_text * has_swap.float()
            contra_loss = contra_loss + L_text
            terms['L_text'] = L_text

        terms['contrastive'] = contra_loss
        terms['loss'] = terms['loss'] + self.lambda_contrastive * contra_loss

        return terms

    # ------------------------------------------------------------------
    @staticmethod
    def _per_sample_masked_l2(a, b, mask):
        """
        Per-sample masked MSE. 반환 shape: [bs]

        Args:
            a, b : [bs, J, Jdim, T]
            mask : [bs, 1,  1,   T]  (True = valid frame)
        """
        diff_sq = (a - b) ** 2                              # [bs, J, Jdim, T]
        masked  = diff_sq * mask.float()                    # broadcast over J, Jdim
        summed  = masked.sum(dim=(1, 2, 3))                 # [bs]
        n_entries = a.shape[1] * a.shape[2]
        n_valid   = mask.squeeze(1).squeeze(1).sum(dim=-1)  # [bs]
        return summed / (n_valid * n_entries + 1e-8)        # [bs]
