"""
dataset_contrastive.py

Text2MotionDatasetV2 / HumanML3D의 Contrastive Learning용 확장 버전.

각 샘플에 대해 두 가지 hard negative를 생성한다:
  1. Motion flip:  원본 텍스트 + 좌우 flip된 모션   → hard negative pair
  2. Text swap:    좌우 교환 텍스트 + 원본 모션      → bidirectional hard negative

Text swap 조건:
  - 캡션에 left 또는 right body-part 표현이 있을 것
  - 단, 동일 캡션에 left/right 둘 다 있으면 swap 후 의미가 애매해지므로 제외

__getitem__ 반환 인덱스:
  0  word_embeddings
  1  pos_one_hots
  2  caption
  3  sent_len
  4  motion          (Z-normalized, padded to max_motion_length)
  5  length
  6  tokens_str
  7  flipped_motion  (Z-normalized, same crop/pad as motion) or None
  8  swapped_caption  (left↔right 교환된 텍스트) or None
"""

import os
import re
import random
import numpy as np
import codecs as cs
import torch
from os.path import join as pjoin
from tqdm import tqdm

from data_loaders.humanml.data.dataset import Text2MotionDatasetV2, HumanML3D
from data_loaders.humanml.utils.word_vectorizer import WordVectorizer
from data_loaders.humanml.utils.get_opt import get_opt

# 6자리 순수 숫자 ID (full-entry), sub-clip ('A_000001' 형태) 제외
_FULL_ID_RE = re.compile(r'^\d+$')

# body-part left/right 패턴
_BODY_PARTS = (
    r'hand|foot|feet|arm|leg|knee|elbow|shoulder|wrist|hip|ankle|'
    r'forearm|thigh|calf|calves|finger|thumb|toe|toes|palm'
)
_LEFT_BODY  = re.compile(r'\bleft\s+(?:' + _BODY_PARTS + r')s?\b',  re.IGNORECASE)
_RIGHT_BODY = re.compile(r'\bright\s+(?:' + _BODY_PARTS + r')s?\b', re.IGNORECASE)


def _swap_lr_text(caption: str):
    """
    캡션 내 left/right body-part 표현을 left↔right 교환한다.
    동일 캡션에 left와 right 둘 다 있으면 None 반환 (swap 후 의미 모호).

    두 단계 치환으로 double-swap 방지:
        left → __LEFT_TMP__ → right
        right → __RIGHT_TMP__ → left
    """
    has_left  = bool(_LEFT_BODY.search(caption))
    has_right = bool(_RIGHT_BODY.search(caption))

    if not (has_left or has_right):
        return None
    if has_left and has_right:
        return None

    tmp = re.sub(
        r'\b(left|right)\b',
        lambda m: '__LEFT_TMP__' if m.group(0).lower() == 'left' else '__RIGHT_TMP__',
        caption,
        flags=re.IGNORECASE,
    )
    tmp = tmp.replace('__LEFT_TMP__', 'right').replace('__RIGHT_TMP__', 'left')
    return tmp


class ContrastiveText2MotionDataset(Text2MotionDatasetV2):
    """
    Text2MotionDatasetV2를 상속.
    flipped_motion_dir이 주어지면 해당 디렉토리에서 flip된 모션을 로드해
    원본 모션과 동일한 crop/padding을 적용한 뒤 튜플의 7번째 원소로 반환.
    flip 파일이 없거나 sub-clip 엔트리이면 None 반환.
    텍스트 swap 캡션은 8번째 원소로 반환 (교환 불가능하면 None).
    """

    def __init__(self, opt, mean, std, split_file, w_vectorizer,
                 flipped_motion_dir=None):
        super().__init__(opt, mean, std, split_file, w_vectorizer)

        self.flipped_motion_dir = flipped_motion_dir
        self.flipped_ids: set = set()

        if flipped_motion_dir and os.path.exists(flipped_motion_dir):
            for fname in os.listdir(flipped_motion_dir):
                if fname.endswith('.npy'):
                    self.flipped_ids.add(fname[:-4])
            print(f'[ContrastiveDataset] {len(self.flipped_ids)} flip IDs '
                  f'loaded from {flipped_motion_dir}')
        else:
            print('[ContrastiveDataset] No flipped_motion_dir — '
                  'contrastive pairs will all be None.')

    # ------------------------------------------------------------------
    # __getitem__ : 부모 코드를 그대로 재현하되 crop_start를 기록해 flip에 적용
    # ------------------------------------------------------------------
    def __getitem__(self, item):
        idx = self.pointer + item
        key = self.name_list[idx]
        data = self.data_dict[key]
        motion, m_length, text_list = data['motion'], data['length'], data['text']

        # 캡션 랜덤 선택
        text_data = random.choice(text_list)
        caption, tokens = text_data['caption'], text_data['tokens']

        # 토큰 처리 (부모와 동일)
        if len(tokens) < self.opt.max_text_len:
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
            tokens = tokens + ['unk/OTHER'] * (self.opt.max_text_len + 2 - sent_len)
        else:
            tokens = tokens[:self.opt.max_text_len]
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)

        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        # crop 길이 결정 (부모와 동일 로직)
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

        # crop 시작점 (이 값을 flip 모션에도 동일 적용)
        crop_start = random.randint(0, len(motion) - m_length)
        if self.opt.disable_offset_aug:
            crop_start = random.randint(0, self.opt.unit_length)

        motion = motion[crop_start:crop_start + m_length]

        # Z 정규화
        motion = (motion - self.mean) / self.std

        # 패딩
        if m_length < self.max_motion_length:
            motion = np.concatenate(
                [motion,
                 np.zeros((self.max_motion_length - m_length, motion.shape[1]),
                           dtype=np.float32)],
                axis=0
            )

        length = (original_length, m_length) if self.opt.fixed_len > 0 else m_length

        # ----------------------------------------------------------
        # Flip 모션 로드 (full-entry만, sub-clip 제외)
        # ----------------------------------------------------------
        flipped_motion = self._load_flipped(key, crop_start, m_length)

        # ----------------------------------------------------------
        # Text swap (left↔right body-part 교환)
        # ----------------------------------------------------------
        swapped_caption = _swap_lr_text(caption)

        return (word_embeddings, pos_one_hots, caption, sent_len,
                motion, length, '_'.join(tokens), flipped_motion, swapped_caption)

    def _load_flipped(self, key, crop_start, m_length):
        """
        key가 순수 숫자 ID이고 flipped_motion_dir에 파일이 있으면
        동일한 crop/padding을 적용한 flip 모션을 반환, 그렇지 않으면 None.
        """
        if (not _FULL_ID_RE.match(key) or
                self.flipped_motion_dir is None or
                key not in self.flipped_ids):
            return None

        flip_path = os.path.join(self.flipped_motion_dir, key + '.npy')
        try:
            flipped_raw = np.load(flip_path)

            # 원본과 동일한 crop
            flipped = flipped_raw[crop_start:crop_start + m_length]

            # Z 정규화 (원본과 동일 mean/std 사용)
            flipped = (flipped - self.mean) / self.std

            # 패딩
            if m_length < self.max_motion_length:
                flipped = np.concatenate(
                    [flipped,
                     np.zeros((self.max_motion_length - m_length, flipped.shape[1]),
                               dtype=np.float32)],
                    axis=0
                )
            return flipped.astype(np.float32)
        except Exception:
            return None


class ContrastiveHumanML3D(HumanML3D):
    """
    HumanML3D의 contrastive 버전.
    내부 t2m_dataset을 ContrastiveText2MotionDataset으로 교체한다.

    추가 인자:
        flipped_motion_dir (str): HumanML3D_flipped/new_joint_vecs/ 경로
    """

    def __init__(self, flipped_motion_dir=None, **kwargs):
        super().__init__(**kwargs)

        mode = kwargs.get('mode', 'train')
        # text_only / gt 모드는 contrastive 불필요
        if mode in ('text_only', 'gt') or flipped_motion_dir is None:
            return

        # 기존 t2m_dataset을 ContrastiveText2MotionDataset으로 교체.
        # super().__init__()에서 캐시가 만들어지므로 두 번째 로드는 캐시 활용.
        self.t2m_dataset = ContrastiveText2MotionDataset(
            self.opt,
            self.mean,
            self.std,
            self.split_file,
            self.w_vectorizer,
            flipped_motion_dir=flipped_motion_dir,
        )
