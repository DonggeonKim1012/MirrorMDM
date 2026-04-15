"""
get_data_contrastive.py

Contrastive training용 DataLoader 생성 모듈.
기존 get_data.py를 변경하지 않고 별도로 작성.

주요 변경:
  - ContrastiveHumanML3D 사용
  - t2m_collate_contrastive: flipped_motion을 cond['y']['flipped_motion'] 으로 전달
  - swapped_text (index 8): cond['y']['swapped_text'] 와 cond['y']['has_swap'] 추가
"""

from torch.utils.data import DataLoader
import torch
import numpy as np

from data_loaders.tensors import collate as all_collate, collate_tensors
from data_loaders.tensors import lengths_to_mask


# ─────────────────────────────────────────────────────────────────────────────
# Collate
# ─────────────────────────────────────────────────────────────────────────────

def t2m_collate_contrastive(batch, target_batch_size):
    """
    ContrastiveText2MotionDataset.__getitem__ 반환값을 배치로 변환.

    __getitem__ 반환 인덱스:
      0 word_embeddings  1 pos_one_hots  2 caption   3 sent_len
      4 motion           5 length        6 tokens_str
      7 flipped_motion (numpy array or None)
      8 swapped_caption (str or None)

    flipped_motion이 None인 샘플은 has_flip=False, 나머지는 has_flip=True.
    flipped_motion 텐서는 cond['y']['flipped_motion'] 으로 전달 ([bs, 263, 1, T]).
    has_flip 마스크는 cond['y']['has_flip'] 으로 전달 (dtype=bool, shape [bs]).

    swapped_caption이 None인 샘플은 has_swap=False, 나머지는 has_swap=True.
    swapped_text 리스트는 cond['y']['swapped_text'] 로 전달 (List[str], len=bs).
    has_swap 마스크는 cond['y']['has_swap'] 으로 전달 (dtype=bool, shape [bs]).
    """
    # 배치 크기 맞추기 (기존 t2m_collate와 동일한 반복 방식)
    repeat_factor = -(-target_batch_size // len(batch))
    repeated = (batch * repeat_factor)[:target_batch_size]

    adapted = []
    flipped_list = []
    has_flip_list = []
    swapped_text_list = []
    has_swap_list = []

    for b in repeated:
        adapted.append({
            'inp':     torch.tensor(b[4].T).float().unsqueeze(1),  # [J, 1, seqlen]
            'text':    b[2],
            'tokens':  b[6],
            'lengths': b[5],
        })
        flipped_list.append(b[7])           # numpy array or None
        has_flip_list.append(b[7] is not None)
        swapped_text_list.append(b[8])      # str or None
        has_swap_list.append(b[8] is not None)

    # 기본 collate
    motion, cond = _base_collate(adapted)

    # flipped_motion 처리
    has_flip = torch.tensor(has_flip_list, dtype=torch.bool)  # [bs]
    cond['y']['has_flip'] = has_flip

    # None인 샘플은 zeros로 채워서 하나의 텐서로 만들기
    flipped_tensor = torch.zeros_like(motion)  # [bs, J, 1, T]
    for i, fm in enumerate(flipped_list):
        if fm is not None:
            # fm: (max_motion_length, 263) → [263, 1, max_motion_length]
            flipped_tensor[i] = torch.tensor(fm.T).float().unsqueeze(1)
    cond['y']['flipped_motion'] = flipped_tensor  # [bs, 263, 1, T]

    # swapped_text 처리
    has_swap = torch.tensor(has_swap_list, dtype=torch.bool)  # [bs]
    cond['y']['has_swap'] = has_swap
    # None인 경우 원본 텍스트로 채워 길이를 맞춤 (has_swap으로 마스킹하므로 무시됨)
    filled_swapped = [
        s if s is not None else adapted[i]['text']
        for i, s in enumerate(swapped_text_list)
    ]
    cond['y']['swapped_text'] = filled_swapped  # List[str], len=bs

    return motion, cond


def _base_collate(notnone_batches):
    """tensors.py의 collate 함수와 동일 로직 (import 순환 방지용 인라인)."""
    databatch = [b['inp'] for b in notnone_batches]
    lenbatch = [b['lengths'] for b in notnone_batches]

    databatchTensor = collate_tensors(databatch)
    lenbatchTensor = torch.as_tensor(lenbatch)
    maskbatchTensor = (
        lengths_to_mask(lenbatchTensor, databatchTensor.shape[-1])
        .unsqueeze(1).unsqueeze(1)
    )

    motion = databatchTensor
    cond = {'y': {'mask': maskbatchTensor, 'lengths': lenbatchTensor}}

    if 'text' in notnone_batches[0]:
        cond['y']['text'] = [b['text'] for b in notnone_batches]
    if 'tokens' in notnone_batches[0]:
        cond['y']['tokens'] = [b['tokens'] for b in notnone_batches]

    return motion, cond


# ─────────────────────────────────────────────────────────────────────────────
# Dataset & DataLoader
# ─────────────────────────────────────────────────────────────────────────────

def get_contrastive_dataset_loader(
    name,
    batch_size,
    num_frames,
    flipped_motion_dir,
    split='train',
    hml_mode='train',
    fixed_len=0,
    pred_len=0,
    device=None,
    autoregressive=False,
    abs_path='.',
    cache_path=None,
):
    """
    Contrastive learning용 DataLoader를 반환한다.

    Args:
        name:               데이터셋 이름 ('humanml' 만 지원)
        batch_size:         배치 크기
        num_frames:         최대 프레임 수
        flipped_motion_dir: HumanML3D_flipped/new_joint_vecs/ 경로
        기타 인자는 get_data.get_dataset_loader와 동일

    Returns:
        DataLoader (motion, cond) 형식이며 cond['y']에
        'flipped_motion' ([bs, 263, 1, T]), 'has_flip' ([bs]),
        'swapped_text' (List[str]), 'has_swap' ([bs]) 추가
    """
    if name not in ('humanml', 'kit'):
        raise ValueError(
            f'get_contrastive_dataset_loader는 humanml/kit만 지원합니다. (got {name})')

    from data_loaders.humanml.data.dataset_contrastive import ContrastiveHumanML3D

    dataset = ContrastiveHumanML3D(
        flipped_motion_dir=flipped_motion_dir,
        mode=hml_mode,
        split=split,
        num_frames=num_frames,
        abs_path=abs_path,
        fixed_len=fixed_len,
        device=device,
        autoregressive=autoregressive,
        cache_path=cache_path or abs_path,
    )

    collate_fn = lambda x: t2m_collate_contrastive(x, batch_size)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=True,
        collate_fn=collate_fn,
    )
    return loader
