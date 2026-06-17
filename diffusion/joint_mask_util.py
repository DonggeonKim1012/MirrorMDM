"""
joint_mask_util.py

HumanML3D 263-dim 벡터에서 관절별 feature index를 계산하고,
텍스트 캡션에서 관련 관절 kinematic chain을 추출하는 유틸.

HumanML3D 263-dim layout:
  [0:4]     root features  (angular_vel=1, linear_vel_xz=2, root_y=1)
  [4:67]    ric positions, joints 1-21 × 3   (non-root)
  [67:193]  6D rotations,  joints 1-21 × 6   (non-root)
  [193:259] velocities,    joints 0-21 × 3   (includes root)
  [259:263] foot contacts  (l_heel, l_toe, r_heel, r_toe)

SMPL 22-joint ordering (0-indexed):
  0  pelvis       1  l_hip        2  r_hip
  3  spine1       4  l_knee       5  r_knee
  6  spine2       7  l_ankle      8  r_ankle
  9  spine3       10 l_foot       11 r_foot
  12 neck         13 l_collar     14 r_collar
  15 head         16 l_shoulder   17 r_shoulder
  18 l_elbow      19 r_elbow      20 l_wrist    21 r_wrist
"""

import re
import torch

# ── Skeleton ───────────────────────────────────────────────────────────────
SMPL_PARENTS = [
    -1, 0, 0, 0,   # pelvis, l_hip, r_hip, spine1
     1, 2, 3,       # l_knee, r_knee, spine2
     4, 5, 6,       # l_ankle, r_ankle, spine3
     7, 8, 9,       # l_foot, r_foot, neck
     9, 9, 12,      # l_collar, r_collar, head  — wait, head parent=12(neck)
    13, 14,         # l_shoulder, r_shoulder
    16, 17,         # l_elbow, r_elbow
    18, 19,         # l_wrist, r_wrist
]
# corrected flat list (len=22):
SMPL_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]

# foot-contact feature indices for specific joints
_FOOT_CONTACT_IDX = {
    10: [259, 260],  # l_foot  → l_heel, l_toe
    11: [261, 262],  # r_foot  → r_heel, r_toe
     7: [259],       # l_ankle → l_heel proxy
     8: [261],       # r_ankle → r_heel proxy
}


# ── Feature index mapping ──────────────────────────────────────────────────

def joint_to_feat_indices(j: int):
    """
    SMPL joint index j → list of 263-dim feature indices for that joint.
    Root (j=0) → root features [0:4] + root velocity.
    Others     → ric position + 6D rotation + velocity.
    """
    if j == 0:
        return list(range(0, 4)) + list(range(193, 196))
    ric = 4  + (j - 1) * 3
    rot = 67 + (j - 1) * 6
    vel = 193 + j * 3
    indices = list(range(ric, ric + 3)) + list(range(rot, rot + 6)) + list(range(vel, vel + 3))
    if j in _FOOT_CONTACT_IDX:
        indices += _FOOT_CONTACT_IDX[j]
    return indices


def get_kinematic_chain(terminal: int):
    """terminal joint → [terminal, ..., pelvis] (root-inclusive chain)."""
    chain, j = [], terminal
    while j != -1:
        chain.append(j)
        j = SMPL_PARENTS[j]
    return chain


def joints_to_feat_mask_1d(joint_indices, device='cpu'):
    """List of joint indices → [263] bool tensor."""
    mask = torch.zeros(263, dtype=torch.bool, device=device)
    for j in joint_indices:
        for idx in joint_to_feat_indices(j):
            if idx < 263:
                mask[idx] = True
    return mask


# ── Text parser ────────────────────────────────────────────────────────────
# (left terminal joint, right terminal joint)
# wrist/hand/arm all use wrist as terminal so the chain covers the whole arm.
_PART_TERMINAL = {
    'finger':    (20, 21),
    'wrist':     (20, 21),
    'hand':      (20, 21),
    'arm':       (20, 21),
    'elbow':     (18, 19),
    'shoulder':  (16, 17),
    'foot':      (10, 11),
    'feet':      (10, 11),
    'toe':       (10, 11),
    'ankle':     (7,  8 ),
    'knee':      (4,  5 ),
    'leg':       (10, 11),
    'thigh':     (4,  5 ),
    'hip':       (1,  2 ),
    'head':      (15, 15),
    'face':      (15, 15),
    'neck':      (12, 12),
    'chest':     (9,  9 ),
    'torso':     (9,  9 ),
    'back':      (9,  9 ),
}

def caption_to_terminal_joints(caption: str):
    """
    캡션에서 신체 부위를 파싱해 terminal joint 인덱스 리스트를 반환.
    방향(left/right)에 관계없이 항상 양쪽 모두 포함.

    이유: "lift left arm" → swap → "lift right arm" 비교 시,
    원본의 왼팔(들린 상태)과 오른팔(내린 상태) 모두 swap 예측과 달라야 하므로
    양쪽 관절을 모두 mask에 포함한다.

    부위를 찾지 못하면 [] 반환 → caller에서 full-body fallback.
    """
    terminals = []
    for part, (l_j, r_j) in _PART_TERMINAL.items():
        if re.search(r'\b' + part + r'\b', caption, re.IGNORECASE):
            terminals.append(l_j)
            if r_j != l_j:
                terminals.append(r_j)
    return list(set(terminals))


def build_joint_feat_mask(captions, device='cpu'):
    """
    List[str] 캡션 배치 → [B, 263, 1, 1] bool 텐서.

    파싱된 신체 부위가 없는 샘플은 전체 feature를 사용 (full-body fallback).
    """
    B = len(captions)
    out = torch.zeros(B, 263, dtype=torch.bool, device=device)
    for i, cap in enumerate(captions):
        terminals = caption_to_terminal_joints(cap)
        if not terminals:
            out[i] = True                           # full-body fallback
        else:
            chain = []
            for t in terminals:
                chain.extend(get_kinematic_chain(t))
            out[i] = joints_to_feat_mask_1d(list(set(chain)), device=device)
    return out.unsqueeze(2).unsqueeze(3)            # [B, 263, 1, 1]
