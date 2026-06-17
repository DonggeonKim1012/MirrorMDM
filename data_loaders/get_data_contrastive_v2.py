"""
get_data_contrastive_v2.py

get_data_contrastive.py (v1) 대비 변경:
  - index 9 (has_lr bool) 처리 추가
  - cond['y']['has_lr'] ([bs] bool tensor) 추가 → LRAdapter gate로 사용
  - ContrastiveHumanML3DV2 사용
"""

from torch.utils.data import DataLoader
import torch
import numpy as np

from data_loaders.tensors import collate_tensors
from data_loaders.tensors import lengths_to_mask


def t2m_collate_contrastive_v2(batch, target_batch_size):
    """
    ContrastiveText2MotionDatasetV2.__getitem__ 반환값을 배치로 변환.

    인덱스:
      0 word_embeddings  1 pos_one_hots  2 caption   3 sent_len
      4 motion           5 length        6 tokens_str
      7 flipped_motion (numpy or None)
      8 swapped_caption (str or None)
      9 has_lr (bool)

    cond['y'] 추가 키:
      'flipped_motion' : [bs, 263, 1, T]
      'has_flip'       : [bs] bool
      'swapped_text'   : List[str]
      'has_swap'       : [bs] bool
      'has_lr'         : [bs] bool   ← NEW (LRAdapter gate)
    """
    repeat_factor = -(-target_batch_size // len(batch))
    repeated = (batch * repeat_factor)[:target_batch_size]

    adapted         = []
    flipped_list    = []
    has_flip_list   = []
    swapped_text_list = []
    has_swap_list   = []
    has_lr_list     = []

    for b in repeated:
        adapted.append({
            'inp':     torch.tensor(b[4].T).float().unsqueeze(1),
            'text':    b[2],
            'tokens':  b[6],
            'lengths': b[5],
        })
        flipped_list.append(b[7])
        has_flip_list.append(b[7] is not None)
        swapped_text_list.append(b[8])
        has_swap_list.append(b[8] is not None)
        has_lr_list.append(bool(b[9]))

    motion, cond = _base_collate(adapted)

    # flipped_motion
    has_flip = torch.tensor(has_flip_list, dtype=torch.bool)
    cond['y']['has_flip'] = has_flip
    flipped_tensor = torch.zeros_like(motion)
    for i, fm in enumerate(flipped_list):
        if fm is not None:
            flipped_tensor[i] = torch.tensor(fm.T).float().unsqueeze(1)
    cond['y']['flipped_motion'] = flipped_tensor

    # swapped_text
    has_swap = torch.tensor(has_swap_list, dtype=torch.bool)
    cond['y']['has_swap'] = has_swap
    cond['y']['swapped_text'] = [
        s if s is not None else adapted[i]['text']
        for i, s in enumerate(swapped_text_list)
    ]

    # has_lr (LRAdapter gate)
    cond['y']['has_lr'] = torch.tensor(has_lr_list, dtype=torch.bool)

    return motion, cond


def _base_collate(notnone_batches):
    databatch      = [b['inp']     for b in notnone_batches]
    lenbatch       = [b['lengths'] for b in notnone_batches]
    databatchTensor = collate_tensors(databatch)
    lenbatchTensor  = torch.as_tensor(lenbatch)
    maskbatchTensor = (
        lengths_to_mask(lenbatchTensor, databatchTensor.shape[-1])
        .unsqueeze(1).unsqueeze(1)
    )
    motion = databatchTensor
    cond   = {'y': {'mask': maskbatchTensor, 'lengths': lenbatchTensor}}
    if 'text'   in notnone_batches[0]: cond['y']['text']   = [b['text']   for b in notnone_batches]
    if 'tokens' in notnone_batches[0]: cond['y']['tokens'] = [b['tokens'] for b in notnone_batches]
    return motion, cond


def get_contrastive_dataset_loader_v2(
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
    if name not in ('humanml', 'kit'):
        raise ValueError(f'v2 loader는 humanml/kit만 지원합니다. (got {name})')

    from data_loaders.humanml.data.dataset_contrastive_v2 import ContrastiveHumanML3DV2

    dataset = ContrastiveHumanML3DV2(
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

    collate_fn = lambda x: t2m_collate_contrastive_v2(x, batch_size)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        collate_fn=collate_fn,
    )
