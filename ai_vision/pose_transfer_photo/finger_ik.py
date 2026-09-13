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


def _right_handed_frame(v1: np.ndarray, v2: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Right-handed orthonormal basis (e1, e2, e3) with e1 along `v1` and e3
    normal to the `v1`/`v2` plane -- built from cross/dot/norm only (no
    matrix-matrix multiply), per `kabsch_rotation`'s environment constraint.
    """
    e1 = v1 / np.linalg.norm(v1)
    e3 = np.cross(v1, v2)
    e3 = e3 / np.linalg.norm(e3)
    e2 = np.cross(e3, e1)
    return e1, e2, e3


def kabsch_rotation(source_vectors: np.ndarray, target_vectors: np.ndarray) -> np.ndarray:
    """Proper rotation `R` mapping the orthonormal frame spanned by
    `source_vectors[0]`/`source_vectors[1]` exactly onto the frame spanned
    by `target_vectors[0]`/`target_vectors[1]` -- a rigid rotation fit from
    a two-vector correspondence (this project's Kabsch/Procrustes use case:
    aligning a WRIST/INDEX_MCP/PINKY_MCP triangle, see
    `anchor_hand_landmarks`), built as `R = sum_k outer(target_k, source_k)`
    over each frame's three basis vectors -- exact for an orthonormal
    source basis, since `R @ source_k = target_k` for each `k`.

    Deliberately avoids `numpy.linalg.svd`/`det`/`inv`/`eigh` and any 2-D
    `@`/`np.dot`/`np.matmul` (matrix-matrix multiply): all of those reliably
    crash with an illegal-instruction fault on this project's numpy/
    OpenBLAS build on this machine (a dgemm/LAPACK BLAS3 bug -- confirmed
    even for `np.eye(3) @ np.eye(3)` in isolation, see bd
    daz-script-server-ap3l), while matrix-vector products, `np.dot` on 1-D
    vectors, `np.cross`, and `np.outer` are all BLAS1/2 operations and are
    unaffected. This is an exact frame-to-frame rotation rather than a
    least-squares fit over noisy many-point data, which is fine here since
    there are only ever two correspondence vectors.
    """
    s1, s2, s3 = _right_handed_frame(source_vectors[0], source_vectors[1])
    t1, t2, t3 = _right_handed_frame(target_vectors[0], target_vectors[1])
    return np.outer(t1, s1) + np.outer(t2, s2) + np.outer(t3, s3)


def anchor_hand_landmarks(
    hand_landmarks: list[tuple[float, float, float]],
    mp_hip_mid: tuple[float, float, float],
    daz_hip_world: tuple[float, float, float],
    unit_scale: float,
    wrist_world: tuple[float, float, float],
    rest_wrist_world: tuple[float, float, float],
    rest_index_mcp_world: tuple[float, float, float],
    rest_pinky_mcp_world: tuple[float, float, float],
) -> list[np.ndarray]:
    """Apply the body pipeline's existing scale+rotation transform to a
    hand's 21 landmarks, then a per-hand Kabsch/Procrustes rotation about
    the wrist, then translate the whole set so its own wrist landmark lands
    exactly on `wrist_world` (the already-solved l_hand/r_hand world
    position).

    The design spec originally assumed MediaPipe's Pose and Hand
    world-landmark tasks share a rotational axis convention -- a pure
    translation, no rotation estimate. Live validation (bd
    daz-script-server-ap3l) falsified that: fingers were driven to the
    wrong (hyperextension) end of their joint-limit range. Per the spec's
    own fallback plan, the extra rotation is fit from the stable
    WRIST/INDEX_MCP/PINKY_MCP triangle -- aligning the transformed
    landmarks' own wrist-relative geometry onto the DAZ rig's REST-pose
    geometry for the corresponding bones (`{side}_hand`/`{side}_index1`/
    `{side}_pinky1` -- a bone's `world_position` is its own proximal joint,
    which is exactly the MCP joint for `index1`/`pinky1`).
    """
    transformed = [
        np.array(to_daz_world(lm, mp_hip_mid, daz_hip_world, unit_scale))
        for lm in hand_landmarks
    ]
    wrist_t = transformed[WRIST]

    source_vectors = np.array([
        transformed[INDEX_MCP] - wrist_t,
        transformed[PINKY_MCP] - wrist_t,
    ])
    target_vectors = np.array([
        np.array(rest_index_mcp_world) - np.array(rest_wrist_world),
        np.array(rest_pinky_mcp_world) - np.array(rest_wrist_world),
    ])
    R = kabsch_rotation(source_vectors, target_vectors)

    wrist_world_arr = np.array(wrist_world)
    return [wrist_world_arr + R @ (p - wrist_t) for p in transformed]
