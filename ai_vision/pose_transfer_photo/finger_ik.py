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


def wrist_world_rotation(
    rest_metadata: list[dict],
    arm_chain_bones: list[str],
    wrist_bone: str,
    solved_angles: dict[str, tuple[float, float, float]],
) -> np.ndarray:
    """World rotation matrix of `wrist_bone` (e.g. "l_hand"/"r_hand") after
    applying the body IK's already-solved angles to the arm chain, built
    from REST `bone_metadata()` (i.e. `pose_transfer_photo.py`'s `metadata`/
    `by_name` snapshot captured *before* the body solve -- do not pass
    post-solve metadata here, or the resulting rotation would double-apply
    whatever the body solve already baked into the positions).

    `arm_chain_bones` must be the same chain `pose_transfer_photo._CHAINS`
    uses for this hand (e.g. `["l_shoulder", "l_upperarm", "l_forearm"]`);
    `wrist_bone` is appended as the trailing marker bone, mirroring how the
    body solve's own `union_chain` includes it.

    This exists because `bone_metadata()` has no world-orientation field --
    only `world_position`. Fixes the "finger model omits the wrist's
    accumulated world rotation" bug (final review of
    docs/superpowers/plans/2026-09-10-finger-posing.md): `build_figure_model`
    roots a digit chain (whose real parent, the wrist, is outside the given
    bone set) directly to the universe frame with IDENTITY rotation, which is
    only exact when that parent's own accumulated world rotation is actually
    identity -- true for the zeroed-figure body solve, false for fingers
    once the wrist has been rotated by the body solve. Callers use this
    matrix (via `rotate_target_into_rest_wrist_frame`) to pre-rotate finger
    targets into the wrist's REST frame before solving each digit against a
    REST-metadata model, rather than trying to teach `build_figure_model`
    about a rotated root (which `pinocchio_ik.py` must not be modified to
    support -- see the plan's hard constraint).
    """
    full_chain = list(arm_chain_bones)
    if wrist_bone not in full_chain:
        full_chain.append(wrist_bone)
    fm = pik.build_figure_model(rest_metadata, full_chain)
    angles = {
        name: {"x": solved_angles[name][0], "y": solved_angles[name][1], "z": solved_angles[name][2]}
        for name in full_chain
    }
    q = pik.configuration_from_angles(fm, angles)
    pin.forwardKinematics(fm.model, fm.data, q)
    return fm.data.oMi[fm.joint_of[wrist_bone]].rotation.copy()


def rotate_target_into_rest_wrist_frame(
    target_point: np.ndarray,
    wrist_rest_world: tuple[float, float, float],
    wrist_posed_world: tuple[float, float, float],
    wrist_world_rotation_matrix: np.ndarray,
) -> np.ndarray:
    """p' = wrist_rest_world + W_hand^T @ (p - wrist_posed_world).

    Maps a world-space target (e.g. an anchored landmark, already translated
    onto the *posed* wrist position by `anchor_hand_landmarks`) into the
    wrist's REST frame, undoing the wrist's accumulated world rotation
    `wrist_world_rotation_matrix` (from `wrist_world_rotation`). A digit
    chain solved against REST metadata (see `build_figure_model`'s
    universe-rooting behavior) implicitly assumes its root's parent frame is
    identity -- this function is what makes that assumption true by
    construction, by moving the target into the frame the model actually
    solves in, rather than moving the model.
    """
    return np.asarray(wrist_rest_world) + wrist_world_rotation_matrix.T @ (
        np.asarray(target_point) - np.asarray(wrist_posed_world)
    )


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
