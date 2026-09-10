"""Finger IK for pose_transfer_photo.py's --fingers flag.

Reuses pinocchio_ik.py's build_figure_model/solve_ik unchanged for every
digit bone except the last: that bone's own rotation affects no observable
position (it has no child bone in the Genesis 9 rig), so IK cannot target
it, and its single free axis is computed directly from landmark geometry
instead. See docs/superpowers/specs/2026-09-10-finger-posing-design.md.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import pinocchio_ik as pik

# MediaPipe HandLandmarker's 21 landmarks, standard indices.
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20


@dataclass(frozen=True)
class DigitChain:
    """One digit's IK chain. `chain_bones` is root -> tip (a 4-bone finger:
    metacarpal + 3 phalanges; a 3-bone thumb: thumb1-3). The IK solve targets
    `chain_bones[-2]` and `chain_bones[-1]`'s own positions (via
    `target1_landmark`/`target2_landmark`) while treating every bone except
    the last as a free rotation -- exactly `pinocchio_ik.py`'s existing
    "trailing bone as pure position marker" convention, already used for
    `l_hand`/`r_hand` themselves in `_CHAINS`. `chain_bones[-1]`'s own
    rotation is never IK-targeted (nothing observable depends on it); its
    one free axis (`tip_hinge_axis`) is set from `tip_landmark` geometry
    instead (see `solve_digit_chain`).
    """

    chain_bones: list[str]
    target1_landmark: int
    target2_landmark: int
    tip_landmark: int
    tip_hinge_axis: str

    @property
    def effector_bones(self) -> list[str]:
        return self.chain_bones[-2:]


def _finger_chain(side: str, name: str, mcp: int) -> DigitChain:
    return DigitChain(
        chain_bones=[f"{side}_{name}metacarpal", f"{side}_{name}1", f"{side}_{name}2", f"{side}_{name}3"],
        target1_landmark=mcp + 1,  # PIP-equivalent
        target2_landmark=mcp + 2,  # DIP-equivalent
        tip_landmark=mcp + 3,      # TIP
        tip_hinge_axis="z",
    )


def _thumb_chain(side: str) -> DigitChain:
    return DigitChain(
        chain_bones=[f"{side}_thumb1", f"{side}_thumb2", f"{side}_thumb3"],
        target1_landmark=THUMB_MCP,
        target2_landmark=THUMB_IP,
        tip_landmark=THUMB_TIP,
        tip_hinge_axis="y",
    )


DIGIT_CHAINS: dict[str, DigitChain] = {}
for _side in ("l", "r"):
    DIGIT_CHAINS[f"{_side}_index"] = _finger_chain(_side, "index", INDEX_MCP)
    DIGIT_CHAINS[f"{_side}_mid"] = _finger_chain(_side, "mid", MIDDLE_MCP)
    DIGIT_CHAINS[f"{_side}_ring"] = _finger_chain(_side, "ring", RING_MCP)
    DIGIT_CHAINS[f"{_side}_pinky"] = _finger_chain(_side, "pinky", PINKY_MCP)
    DIGIT_CHAINS[f"{_side}_thumb"] = _thumb_chain(_side)
del _side


_AXIS_PLANE_INDICES = {"x": (1, 2), "y": (2, 0), "z": (0, 1)}


def hinge_angle_from_vectors(v_in: np.ndarray, v_out: np.ndarray, axis: str) -> float:
    """Signed angle (degrees) about `axis` that rotates `v_in` toward `v_out`,
    ignoring any component along `axis` itself. Matches the right-hand-rule
    sign convention of `pinocchio_ik._axis_rotation` -- verified directly
    against it in `test_hinge_angle_from_vectors_recovers_known_rotation`.
    """
    i, j = _AXIS_PLANE_INDICES[axis]
    cross = v_in[i] * v_out[j] - v_in[j] * v_out[i]
    dot = v_in[i] * v_out[i] + v_in[j] * v_out[j]
    return float(np.degrees(np.arctan2(cross, dot)))
