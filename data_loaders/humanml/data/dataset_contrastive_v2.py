"""
dataset_contrastive_v2.py

dataset_contrastive.py (v1) 대비 변경 사항:
  1. Flipped 샘플을 정식 훈련 데이터로 추가
     - key='{id}_flip', motion=flipped, text=[swapped_caption]
     - 학습 데이터: 원본 23K + flip 10K = ~33K
  2. 양방향 negative pair
     - 원본 엔트리의 negative: flip 모션 (v1과 동일)
     - flip 엔트리의 negative: 원본 모션
  3. has_lr 플래그 추가 (index 9)
     - LRAdapter의 gate로 사용

__getitem__ 반환 인덱스:
  0  word_embeddings
  1  pos_one_hots
  2  caption
  3  sent_len
  4  motion          (Z-norm, padded)
  5  length
  6  tokens_str
  7  flipped_motion  (Z-norm, same crop) or None
  8  swapped_caption or None
  9  has_lr          (bool)
"""

import os
import re
import random
import numpy as np

from data_loaders.humanml.data.dataset_contrastive import (
    ContrastiveText2MotionDataset,
    ContrastiveHumanML3D,
    _swap_lr_text,
    _FULL_ID_RE,
    _LEFT_BODY,
    _RIGHT_BODY,
)


def _swap_tokens(tokens):
    """토큰 리스트에서 left↔right 교환. 'left/ADJ' → 'right/ADJ'"""
    result = []
    for tok in tokens:
        if '/' in tok:
            word, pos = tok.split('/', 1)
            if word.lower() == 'left':
                result.append('right/' + pos)
            elif word.lower() == 'right':
                result.append('left/' + pos)
            else:
                result.append(tok)
        else:
            result.append(tok)
    return result


class ContrastiveText2MotionDatasetV2(ContrastiveText2MotionDataset):
    """
    v1 대비:
      - __init__에서 flip 엔트리를 data_dict / name_list에 추가
      - __getitem__에서 flip 엔트리의 negative는 원본 모션을 반환
      - 반환 튜플 index 9에 has_lr (bool) 추가
    """

    def __init__(self, opt, mean, std, split_file, w_vectorizer,
                 flipped_motion_dir=None):
        super().__init__(opt, mean, std, split_file, w_vectorizer,
                         flipped_motion_dir=flipped_motion_dir)

        if not flipped_motion_dir or not os.path.exists(flipped_motion_dir):
            return

        # ── flip 엔트리를 data_dict / name_list에 추가 ──────────────
        min_motion_len = 40 if opt.dataset_name == 't2m' else 24

        # name_list는 tuple, 수정을 위해 list로 변환
        name_list   = list(self.name_list)
        length_list = list(self.length_arr)
        data_dict   = self.data_dict          # dict (mutable)

        added = 0
        for key in list(name_list):           # 원본 리스트만 순회
            if not _FULL_ID_RE.match(key):
                continue
            if key not in self.flipped_ids:
                continue

            orig_data = data_dict[key]

            # 최소 하나의 캡션이 swappable해야 flip 엔트리 추가
            swappable = [td for td in orig_data['text']
                         if _swap_lr_text(td['caption']) is not None]
            if not swappable:
                continue

            flip_path = os.path.join(flipped_motion_dir, key + '.npy')
            if not os.path.exists(flip_path):
                continue

            # flip은 공간 변환이라 프레임 수 = 원본과 동일
            flip_len = orig_data['length']
            if flip_len < min_motion_len or flip_len >= 200:
                continue

            # swapped text list 생성
            swapped_text_data = []
            for td in swappable:
                swapped_cap = _swap_lr_text(td['caption'])
                if swapped_cap is None:
                    continue
                swapped_text_data.append({
                    'caption': swapped_cap,
                    'tokens':  _swap_tokens(td['tokens']),
                })
            if not swapped_text_data:
                continue

            flip_key = key + '_flip'
            data_dict[flip_key] = {
                'motion':    None,        # lazy: loaded in __getitem__
                'flip_path': flip_path,
                'length':    flip_len,
                'text':      swapped_text_data,
                '_orig_key': key,
            }
            name_list.append(flip_key)
            length_list.append(flip_len)
            added += 1

        # 길이순 재정렬 (pointer 계산이 length_arr 기반)
        paired = sorted(zip(length_list, name_list), key=lambda x: x[0])
        length_list, name_list = zip(*paired)

        self.name_list  = tuple(name_list)
        self.length_arr = np.array(length_list)
        self.data_dict  = data_dict
        # pointer 재설정
        self.reset_max_len(self.max_length)

        print(f'[ContrastiveDatasetV2] {added} flip entries added '
              f'(total {len(self.name_list)} samples)')

    # ------------------------------------------------------------------
    def __getitem__(self, item):
        idx  = self.pointer + item
        key  = self.name_list[idx]
        data = self.data_dict[key]
        # lazy load for flip entries
        raw_motion = data['motion']
        if raw_motion is None:
            raw_motion = np.load(data['flip_path'])
        motion, m_length, text_list = raw_motion, data['length'], data['text']

        text_data = random.choice(text_list)
        caption, tokens = text_data['caption'], text_data['tokens']

        if len(tokens) < self.opt.max_text_len:
            tokens   = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
            tokens   = tokens + ['unk/OTHER'] * (self.opt.max_text_len + 2 - sent_len)
        else:
            tokens   = tokens[:self.opt.max_text_len]
            tokens   = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)

        pos_one_hots   = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots    = np.concatenate(pos_one_hots,    axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        if self.opt.unit_length < 10:
            coin2 = np.random.choice(['single', 'single', 'double'])
        else:
            coin2 = 'single'

        if coin2 == 'double':
            m_length = (m_length // self.opt.unit_length - 1) * self.opt.unit_length
        elif coin2 == 'single':
            m_length = (m_length // self.opt.unit_length) * self.opt.unit_length

        original_length = None
        if self.opt.fixed_len > 0:
            original_length = m_length
            m_length = self.opt.fixed_len

        crop_start = random.randint(0, len(motion) - m_length)
        if self.opt.disable_offset_aug:
            crop_start = random.randint(0, self.opt.unit_length)

        motion = motion[crop_start:crop_start + m_length]
        motion = (motion - self.mean) / self.std
        if m_length < self.max_motion_length:
            motion = np.concatenate(
                [motion,
                 np.zeros((self.max_motion_length - m_length, motion.shape[1]),
                           dtype=np.float32)],
                axis=0)

        length = (original_length, m_length) if self.opt.fixed_len > 0 else m_length

        # ── has_lr 판별 ────────────────────────────────────────────
        has_lr = (bool(_LEFT_BODY.search(caption)) or
                  bool(_RIGHT_BODY.search(caption)))

        # ── negative 로드 (LR 캡션일 때만) ────────────────────────
        flipped_motion = self._load_flipped_v2(key, data, crop_start, m_length) if has_lr else None

        # ── text swap ─────────────────────────────────────────────
        swapped_caption = _swap_lr_text(caption)

        return (word_embeddings, pos_one_hots, caption, sent_len,
                motion, length, '_'.join(tokens),
                flipped_motion, swapped_caption, has_lr)

    def _load_flipped_v2(self, key, data, crop_start, m_length):
        """
        - flip 엔트리 (key ends with '_flip'): negative = 원본 모션 (data_dict에 있음)
        - 원본 LR 엔트리: negative = flip 파일 로드 (v1과 동일)
        - non-LR 엔트리: None
        """
        if key.endswith('_flip'):
            orig_key   = data.get('_orig_key', key[:-5])
            orig_data  = self.data_dict.get(orig_key)
            if orig_data is None:
                return None
            orig_motion = orig_data['motion']
            if crop_start + m_length > len(orig_motion):
                crop_start = max(0, len(orig_motion) - m_length)
            flipped = orig_motion[crop_start:crop_start + m_length]
            flipped = (flipped - self.mean) / self.std
            if m_length < self.max_motion_length:
                flipped = np.concatenate(
                    [flipped,
                     np.zeros((self.max_motion_length - m_length, flipped.shape[1]),
                               dtype=np.float32)],
                    axis=0)
            return flipped.astype(np.float32)
        else:
            return self._load_flipped(key, crop_start, m_length)


class ContrastiveHumanML3DV2(ContrastiveHumanML3D):
    """
    ContrastiveHumanML3D의 v2 버전.
    t2m_dataset을 ContrastiveText2MotionDatasetV2로 교체.
    """

    def __init__(self, flipped_motion_dir=None, **kwargs):
        super().__init__(flipped_motion_dir=flipped_motion_dir, **kwargs)

        mode = kwargs.get('mode', 'train')
        if mode in ('text_only', 'gt') or flipped_motion_dir is None:
            return

        self.t2m_dataset = ContrastiveText2MotionDatasetV2(
            self.opt,
            self.mean,
            self.std,
            self.split_file,
            self.w_vectorizer,
            flipped_motion_dir=flipped_motion_dir,
        )
