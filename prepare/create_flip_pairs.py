"""
create_flip_pairs.py

HumanML3D 데이터셋에서 텍스트 프롬프트에 좌우 신체 부위 표현이 있는 샘플만 골라
모션을 좌우 flip한 false pair를 생성한다.

False pair: (원본 텍스트, 좌우 flip된 모션)
           → 텍스트가 "swing left arm"인데 모션은 오른팔을 swing → hard negative

출력 디렉토리 (원본은 절대 수정하지 않음):
    dataset/HumanML3D_flipped/
        new_joint_vecs/   ← flip된 모션 (263-dim)
        texts/            ← 원본 텍스트 그대로 복사
        flipped_ids.txt   ← 처리된 샘플 ID 목록
"""

import os
import re
import shutil
import numpy as np
from tqdm import tqdm

# ─────────────────────────────────────────────
# 경로 설정
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_TEXTS = os.path.join(BASE_DIR, 'dataset', 'HumanML3D', 'texts')
SRC_VECS  = os.path.join(BASE_DIR, 'dataset', 'downloaded-HumanML3D', 'new_joint_vecs')
OUT_DIR   = os.path.join(BASE_DIR, 'dataset', 'HumanML3D_flipped')
OUT_VECS  = os.path.join(OUT_DIR, 'new_joint_vecs')
OUT_TEXTS = os.path.join(OUT_DIR, 'texts')

os.makedirs(OUT_VECS, exist_ok=True)
os.makedirs(OUT_TEXTS, exist_ok=True)

# ─────────────────────────────────────────────
# 텍스트 필터: left/right + 신체 부위 조합만
# ─────────────────────────────────────────────
# 방향성 표현 (walk to the left, turn right 등)이 아닌
# 신체 부위 지칭 (left hand, right arm 등)만 포함
BODY_PARTS = (
    r'hand|foot|feet|arm|leg|knee|elbow|shoulder|wrist|hip|ankle|'
    r'forearm|thigh|calf|calves|finger|thumb|toe|toes|palm'
)
# "left/right" 바로 뒤에 (s) 옵션 붙은 신체 부위 단어가 오는 패턴
BODY_PART_LR = re.compile(
    r'\b(left|right)\s+(' + BODY_PARTS + r')s?\b',
    re.IGNORECASE
)

def has_body_part_lr(caption: str) -> bool:
    """캡션에 좌/우 신체 부위 표현이 있으면 True."""
    return bool(BODY_PART_LR.search(caption))


def file_has_body_part_lr(text_path: str) -> bool:
    """텍스트 파일의 캡션 중 하나라도 좌/우 신체 부위 표현이 있으면 True."""
    with open(text_path, 'r') as f:
        for line in f:
            caption = line.strip().split('#')[0]
            if has_body_part_lr(caption):
                return True
    return False

# ─────────────────────────────────────────────
# HumanML3D 263-dim 좌우 flip
# ─────────────────────────────────────────────
# 22개 관절 순서 (HML_JOINT_NAMES):
#  0:pelvis, 1:left_hip, 2:right_hip, 3:spine1, 4:left_knee, 5:right_knee,
#  6:spine2, 7:left_ankle, 8:right_ankle, 9:spine3,
#  10:left_foot, 11:right_foot, 12:neck,
#  13:left_collar, 14:right_collar, 15:head,
#  16:left_shoulder, 17:right_shoulder, 18:left_elbow, 19:right_elbow,
#  20:left_wrist, 21:right_wrist
#
# 263-dim 구성:
#  [0]      root_rot_velocity   (Y축 각속도, arcsin 적용)
#  [1:3]    root_linear_velocity (XZ 선속도)
#  [3]      root_y              (높이)
#  [4:67]   ric_data            (joints 1-21, 각 3dim = 21*3=63)
#  [67:193] rot_data            (joints 1-21, 각 6dim = 21*6=126, 6D rotation)
#  [193:259]local_velocity      (joints 0-21, 각 3dim = 22*3=66)
#  [259:263]foot_contact        (left_ankle, left_foot, right_ankle, right_foot)

# 좌우 대칭 관절 쌍 (22-joint 인덱스 기준)
SWAP_PAIRS_22 = [(1, 2), (4, 5), (7, 8), (10, 11),
                 (13, 14), (16, 17), (18, 19), (20, 21)]
# ric_data / rot_data 용: pelvis(0) 제외하므로 인덱스 -1
SWAP_PAIRS_21 = [(l - 1, r - 1) for l, r in SWAP_PAIRS_22]


def flip_hml_vec(motion: np.ndarray) -> np.ndarray:
    """
    HumanML3D 263-dim feature vector를 좌우 flip한다.

    Args:
        motion: (T, 263) float32 array

    Returns:
        flipped: (T, 263) float32 array
    """
    assert motion.shape[1] == 263, f"Expected 263-dim, got {motion.shape[1]}"
    T = motion.shape[0]
    flipped = motion.copy()

    # 1. root_rot_velocity (dim 0)
    #    Y축 회전 방향이 반전되므로 부호 반전
    flipped[:, 0] = -motion[:, 0]

    # 2. root_linear_velocity (dims 1-2 = [X, Z])
    #    X 방향(좌우) 속도 부호 반전, Z(전후)는 유지
    flipped[:, 1] = -motion[:, 1]

    # 3. ric_data (dims 4~66): joints 1-21의 root-relative local 위치, 각 3dim
    #    좌우 관절 쌍 swap 후 X 성분 부호 반전
    ric = motion[:, 4:67].reshape(T, 21, 3).copy()
    for l, r in SWAP_PAIRS_21:
        ric[:, [l, r]] = ric[:, [r, l]]
    ric[:, :, 0] *= -1          # X(좌우) 부호 반전
    flipped[:, 4:67] = ric.reshape(T, 63)

    # 4. rot_data (dims 67~192): joints 1-21의 6D rotation, 각 6dim
    #    6D = [r1_x, r1_y, r1_z, r2_x, r2_y, r2_z] (회전행렬 첫 두 열)
    #    Mirror flip M=diag(-1,1,1) 적용: M R M 변환
    #      r1' = [r1_x, -r1_y, -r1_z]
    #      r2' = [-r2_x,  r2_y,  r2_z]
    #    → 부호 패턴: [+, -, -, -, +, +]
    rot = motion[:, 67:193].reshape(T, 21, 6).copy()
    for l, r in SWAP_PAIRS_21:
        rot[:, [l, r]] = rot[:, [r, l]]
    rot[:, :, 1] *= -1          # -r1_y
    rot[:, :, 2] *= -1          # -r1_z
    rot[:, :, 3] *= -1          # -r2_x
    flipped[:, 67:193] = rot.reshape(T, 126)

    # 5. local_velocity (dims 193~258): joints 0-21의 local frame 속도, 각 3dim
    #    좌우 관절 쌍 swap 후 X 성분 부호 반전
    vel = motion[:, 193:259].reshape(T, 22, 3).copy()
    for l, r in SWAP_PAIRS_22:
        vel[:, [l, r]] = vel[:, [r, l]]
    vel[:, :, 0] *= -1          # X(좌우) 부호 반전
    flipped[:, 193:259] = vel.reshape(T, 66)

    # 6. foot_contact (dims 259~262): [left_ankle, left_foot, right_ankle, right_foot]
    #    좌우 쌍 swap
    flipped[:, 259] = motion[:, 261]   # left_ankle  ← right_ankle
    flipped[:, 260] = motion[:, 262]   # left_foot   ← right_foot
    flipped[:, 261] = motion[:, 259]   # right_ankle ← left_ankle
    flipped[:, 262] = motion[:, 260]   # right_foot  ← left_foot

    return flipped


# ─────────────────────────────────────────────
# 메인 처리
# ─────────────────────────────────────────────
def main():
    text_files = sorted(f for f in os.listdir(SRC_TEXTS) if f.endswith('.txt'))
    print(f"전체 샘플 수: {len(text_files)}")

    included_ids = []
    skipped_no_vec = 0
    skipped_no_body_part = 0

    for txt_fname in tqdm(text_files, desc='Processing'):
        sample_id = txt_fname.replace('.txt', '')
        text_path = os.path.join(SRC_TEXTS, txt_fname)
        vec_path  = os.path.join(SRC_VECS, sample_id + '.npy')

        # 신체 부위 left/right 표현이 없으면 제외
        if not file_has_body_part_lr(text_path):
            skipped_no_body_part += 1
            continue

        # 모션 파일이 없으면 건너뜀
        if not os.path.exists(vec_path):
            skipped_no_vec += 1
            continue

        # 모션 flip
        motion = np.load(vec_path)
        flipped = flip_hml_vec(motion)

        # 저장
        np.save(os.path.join(OUT_VECS, sample_id + '.npy'), flipped)
        shutil.copy2(text_path, os.path.join(OUT_TEXTS, txt_fname))

        included_ids.append(sample_id)

    # ID 목록 저장
    ids_path = os.path.join(OUT_DIR, 'flipped_ids.txt')
    with open(ids_path, 'w') as f:
        f.write('\n'.join(included_ids))

    print(f"\n완료!")
    print(f"  생성된 false pair 수:    {len(included_ids)}")
    print(f"  제외 (신체부위 표현 없음): {skipped_no_body_part}")
    print(f"  제외 (모션 파일 없음):    {skipped_no_vec}")
    print(f"  저장 경로: {OUT_DIR}")
    print(f"  ID 목록:   {ids_path}")


if __name__ == '__main__':
    main()
