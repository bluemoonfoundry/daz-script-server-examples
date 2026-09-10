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
import pinocchio as pin

import pinocchio_ik as pik
from pose_transfer_photo import to_daz_world

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


def solve_digit_chain(
    fm: pik.FigureModel,
    digit: DigitChain,
    target1_point: np.ndarray,
    target2_point: np.ndarray,
    tip_point: np.ndarray,
    initial_angles: dict[str, dict[str, float]],
    **solve_ik_kwargs,
) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    """IK-solve every bone in `digit.chain_bones` except the last (against
    `target1_point`/`target2_point`), then set the last bone's one free axis
    directly from `tip_point` geometry -- nothing observable depends on that
    bone's own rotation, so IK cannot target it. See this function's
    docstring in the implementation plan for the derivation.
    """
    solved_angles, final_error = pik.solve_ik(
        fm, digit.effector_bones, np.array([target1_point, target2_point]),
        initial_angles=initial_angles, **solve_ik_kwargs,
    )

    q_solved = pik.configuration_from_angles(fm, solved_angles)
    pin.forwardKinematics(fm.model, fm.data, q_solved)
    R_ref = fm.data.oMi[fm.joint_of[digit.chain_bones[-2]]].rotation

    v_in_world = np.asarray(target2_point) - np.asarray(target1_point)
    v_out_world = np.asarray(tip_point) - np.asarray(target2_point)

    tip_meta = fm.by_name[digit.chain_bones[-1]]
    O_tip = pik._rest_orientation_matrix(tip_meta)
    w_in = O_tip @ (R_ref.T @ v_in_world)
    w_out = O_tip @ (R_ref.T @ v_out_world)
    theta = hinge_angle_from_vectors(w_in, w_out, digit.tip_hinge_axis)

    solved_angles[digit.chain_bones[-1]][digit.tip_hinge_axis] = theta
    solved_angles = pik.clamp_angles(fm, solved_angles)
    return solved_angles, final_error


def anchor_hand_landmarks(
    hand_landmarks: list[tuple[float, float, float]],
    mp_hip_mid: tuple[float, float, float],
    daz_hip_world: tuple[float, float, float],
    unit_scale: float,
    wrist_world: tuple[float, float, float],
) -> list[np.ndarray]:
    """Apply the body pipeline's existing scale+rotation transform to a
    hand's 21 landmarks, then translate the whole set so its own wrist
    landmark lands exactly on `wrist_world` (the already-solved l_hand/
    r_hand world position). Per the design spec: this assumes MediaPipe's
    Pose and Hand world-landmark tasks share a rotational axis convention
    (verified live in Task 6, not re-derived here) -- a pure translation,
    no separate rotation estimate.
    """
    transformed = [
        np.array(to_daz_world(lm, mp_hip_mid, daz_hip_world, unit_scale))
        for lm in hand_landmarks
    ]
    offset = np.array(wrist_world) - transformed[0]
    return [p + offset for p in transformed]
