from __future__ import annotations

import numpy as np
import pytest

import finger_ik as fik
import pinocchio_ik as pik
from pose_transfer_photo import to_daz_world


def test_digit_chains_cover_both_hands_and_all_five_digits():
    expected_keys = {
        f"{side}_{digit}"
        for side in ("l", "r")
        for digit in ("index", "mid", "ring", "pinky", "thumb")
    }
    assert set(fik.DIGIT_CHAINS.keys()) == expected_keys


def test_finger_chains_have_four_bones_thumb_chains_have_three():
    for key, digit in fik.DIGIT_CHAINS.items():
        side, name = key.split("_", 1)
        if name == "thumb":
            assert len(digit.chain_bones) == 3
            assert digit.chain_bones == [f"{side}_thumb1", f"{side}_thumb2", f"{side}_thumb3"]
            assert digit.tip_hinge_axis == "y"
        else:
            assert len(digit.chain_bones) == 4
            assert digit.chain_bones == [
                f"{side}_{name}metacarpal", f"{side}_{name}1", f"{side}_{name}2", f"{side}_{name}3",
            ]
            assert digit.tip_hinge_axis == "z"


def test_effector_bones_are_last_two_chain_bones():
    for digit in fik.DIGIT_CHAINS.values():
        assert digit.effector_bones == digit.chain_bones[-2:]


def test_landmark_indices_are_within_valid_range_and_ordered():
    for digit in fik.DIGIT_CHAINS.values():
        assert 0 <= digit.target1_landmark <= 20
        assert 0 <= digit.target2_landmark <= 20
        assert 0 <= digit.tip_landmark <= 20
        # MCP/CMC-equiv < PIP/MCP-equiv < DIP/IP-equiv < TIP, always consecutive
        assert digit.target1_landmark + 1 == digit.target2_landmark
        assert digit.target2_landmark + 1 == digit.tip_landmark


def test_index_chain_landmarks_match_known_indices():
    d = fik.DIGIT_CHAINS["l_index"]
    assert (d.target1_landmark, d.target2_landmark, d.tip_landmark) == (6, 7, 8)


def test_thumb_chain_landmarks_match_known_indices():
    d = fik.DIGIT_CHAINS["r_thumb"]
    assert (d.target1_landmark, d.target2_landmark, d.tip_landmark) == (2, 3, 4)


@pytest.mark.parametrize("axis,in_plane_indices", [("x", (1, 2)), ("y", (2, 0)), ("z", (0, 1))])
@pytest.mark.parametrize("angle_deg", [0.0, 30.0, -45.0, 90.0, -170.0])
def test_hinge_angle_from_vectors_recovers_known_rotation(axis, in_plane_indices, angle_deg):
    # Build v_in as a unit vector in the plane perpendicular to `axis`, and
    # v_out as v_in rotated by angle_deg about `axis` using pinocchio_ik's
    # own axis-rotation matrix -- so this test is validating against the
    # exact same rotation convention the rest of the codebase uses.
    import pinocchio_ik as pik

    v_in = np.zeros(3)
    i, j = in_plane_indices
    v_in[i] = 1.0
    R = pik._axis_rotation(axis.upper(), angle_deg)
    v_out = R @ v_in

    recovered = fik.hinge_angle_from_vectors(v_in, v_out, axis)
    assert recovered == pytest.approx(angle_deg, abs=1e-6)


def test_hinge_angle_from_vectors_ignores_component_along_axis():
    # A component along the hinge axis itself shouldn't affect the recovered
    # angle -- only the projection onto the perpendicular plane matters.
    v_in = np.array([1.0, 0.0, 5.0])
    v_out = np.array([0.0, 1.0, -3.0])  # v_in's xy-part (1,0) rotated +90 deg about z
    assert fik.hinge_angle_from_vectors(v_in, v_out, "z") == pytest.approx(90.0, abs=1e-6)


def _bone_meta(name, parent_name, world_position, rotation_order="XYZ",
                rest_orientation=(1.0, 0.0, 0.0, 0.0), axis_limits=None):
    w, x, y, z = rest_orientation
    return {
        "name": name, "parent_name": parent_name,
        "world_position": {"x": world_position[0], "y": world_position[1], "z": world_position[2]},
        "rotation_order": rotation_order,
        "rest_orientation": {"w": w, "x": x, "y": y, "z": z},
        "axis_limits": axis_limits or {a: {"min": -180.0, "max": 180.0} for a in "xyz"},
    }


def _synthetic_finger_chain():
    """root -> mid -> hinge2 -> hinge3, mirroring l_index's real shape:
    root and mid have real x/y/z range, hinge2/hinge3 are pure z-hinges."""
    root = _bone_meta("root", None, (0.0, 0.0, 0.0))
    mid = _bone_meta("mid", "root", (1.0, 0.0, 0.0))
    hinge2 = _bone_meta(
        "hinge2", "mid", (2.0, 0.0, 0.0),
        axis_limits={"x": {"min": 0.0, "max": 0.0}, "y": {"min": 0.0, "max": 0.0}, "z": {"min": -105.0, "max": 12.0}},
    )
    hinge3 = _bone_meta(
        "hinge3", "hinge2", (2.7, 0.0, 0.0),
        axis_limits={"x": {"min": 0.0, "max": 0.0}, "y": {"min": 0.0, "max": 0.0}, "z": {"min": -90.0, "max": 20.0}},
    )
    fm = pik.build_figure_model([root, mid, hinge2, hinge3], ["root", "mid", "hinge2", "hinge3"])
    digit = fik.DigitChain(
        chain_bones=["root", "mid", "hinge2", "hinge3"],
        target1_landmark=6, target2_landmark=7, tip_landmark=8, tip_hinge_axis="z",
    )
    return fm, digit


def test_solve_digit_chain_recovers_known_tip_hinge_angle():
    fm, digit = _synthetic_finger_chain()
    true_angles = {
        "root": {"x": 5.0, "y": -8.0, "z": 12.0},
        "mid": {"x": 2.0, "y": 3.0, "z": -20.0},
        "hinge2": {"x": 0.0, "y": 0.0, "z": -35.0},
        "hinge3": {"x": 0.0, "y": 0.0, "z": -15.0},
    }
    q_true = pik.configuration_from_angles(fm, true_angles)
    positions = pik.forward_kinematics_positions(fm, q_true)
    # A synthetic "tip" landmark 0.6 units past hinge3, along its own local
    # +x direction at the true pose -- stands in for the real TIP landmark,
    # which has no corresponding bone.
    import pinocchio as pin
    pin.forwardKinematics(fm.model, fm.data, q_true)
    hinge3_world_rot = fm.data.oMi[fm.joint_of["hinge3"]].rotation
    tip_point = positions["hinge3"] + hinge3_world_rot @ np.array([0.6, 0.0, 0.0])

    initial_angles = {name: {"x": 0.0, "y": 0.0, "z": 0.0} for name in digit.chain_bones}
    solved, err = fik.solve_digit_chain(
        fm, digit,
        target1_point=positions["hinge2"], target2_point=positions["hinge3"], tip_point=tip_point,
        initial_angles=initial_angles, max_iterations=150, tolerance=0.01,
    )

    assert err["hinge3"] < 0.05
    assert solved["hinge3"]["z"] == pytest.approx(true_angles["hinge3"]["z"], abs=1.0)
    assert solved["hinge3"]["x"] == pytest.approx(0.0, abs=1e-6)
    assert solved["hinge3"]["y"] == pytest.approx(0.0, abs=1e-6)


def _synthetic_wrist_and_finger_chain():
    """wrist (parent None) -> root -> mid -> hinge2 -> hinge3, mirroring the
    real l_hand -> l_indexmetacarpal -> l_index1 -> l_index2 -> l_index3
    shape. Unlike `_synthetic_finger_chain` (whose `root` has no parent at
    all -- the one configuration where the frame bug is invisible, since a
    parentless root's "rooted to universe with identity rotation" is
    already correct), `root` here has a real parent ("wrist") with its own
    substantial, non-identity rotation -- the exact shape of `l_hand`'s
    accumulated world rotation after the body IK solve.
    """
    wrist = _bone_meta("wrist", None, (0.0, 0.0, 0.0))
    root = _bone_meta("root", "wrist", (1.0, 0.0, 0.0))
    mid = _bone_meta("mid", "root", (2.0, 0.0, 0.0))
    hinge2 = _bone_meta(
        "hinge2", "mid", (3.0, 0.0, 0.0),
        axis_limits={"x": {"min": 0.0, "max": 0.0}, "y": {"min": 0.0, "max": 0.0}, "z": {"min": -105.0, "max": 12.0}},
    )
    hinge3 = _bone_meta(
        "hinge3", "hinge2", (3.7, 0.0, 0.0),
        axis_limits={"x": {"min": 0.0, "max": 0.0}, "y": {"min": 0.0, "max": 0.0}, "z": {"min": -90.0, "max": 20.0}},
    )
    all_meta = [wrist, root, mid, hinge2, hinge3]
    fm_full = pik.build_figure_model(all_meta, ["wrist", "root", "mid", "hinge2", "hinge3"])
    fm_digit = pik.build_figure_model(all_meta, ["root", "mid", "hinge2", "hinge3"])
    digit = fik.DigitChain(
        chain_bones=["root", "mid", "hinge2", "hinge3"],
        target1_landmark=6, target2_landmark=7, tip_landmark=8, tip_hinge_axis="z",
    )
    return all_meta, fm_full, fm_digit, digit


def test_naive_digit_solve_is_wrong_under_rotated_wrist_but_frame_correction_fixes_it():
    """Regression test for the "finger model omits the wrist's accumulated
    world rotation" bug (final review of the finger-posing plan): a digit
    chain built from REST metadata roots its first bone directly to the
    universe frame with IDENTITY rotation, since that bone's real parent
    (the wrist) is outside the digit-only bone set. That's only exact if the
    wrist's own accumulated world rotation is actually identity -- true for
    a freshly-zeroed figure's body solve, false for fingers once the body
    solve has rotated the wrist. `_synthetic_finger_chain`'s existing test
    can't catch this: its `root` has no parent at all, so "rooted to
    universe with identity" is already correct there -- exactly why this bug
    passed six task reviews undetected (see the plan's final review).

    This test builds a chain with a genuinely rotated parent ("wrist") and
    shows: (1) naively solving the digit chain from REST metadata directly
    against the true world-space targets "converges" (small IK residual in
    its own, wrong frame) but produces WRONG bone angles -- applying them
    under the wrist's real rotation lands far from the true target
    positions; (2) pre-rotating the targets into the wrist's REST frame via
    `wrist_world_rotation`/`rotate_target_into_rest_wrist_frame` (the actual
    fix applied in pose_transfer_photo.py) recovers the true target
    positions closely.
    """
    all_meta, fm_full, fm_digit, digit = _synthetic_wrist_and_finger_chain()

    true_angles = {
        "wrist": {"x": 35.0, "y": -50.0, "z": 65.0},
        "root": {"x": 5.0, "y": -8.0, "z": 12.0},
        "mid": {"x": 2.0, "y": 3.0, "z": -20.0},
        "hinge2": {"x": 0.0, "y": 0.0, "z": -35.0},
        "hinge3": {"x": 0.0, "y": 0.0, "z": -15.0},
    }
    q_true = pik.configuration_from_angles(fm_full, true_angles)
    positions_true = pik.forward_kinematics_positions(fm_full, q_true)

    import pinocchio as pin
    pin.forwardKinematics(fm_full.model, fm_full.data, q_true)
    hinge3_world_rot_true = fm_full.data.oMi[fm_full.joint_of["hinge3"]].rotation
    tip_point_true = positions_true["hinge3"] + hinge3_world_rot_true @ np.array([0.6, 0.0, 0.0])

    target1 = positions_true["hinge2"]
    target2 = positions_true["hinge3"]
    tip = tip_point_true

    zero_angles = {name: {"x": 0.0, "y": 0.0, "z": 0.0} for name in digit.chain_bones}

    def _apply_digit_solution_under_true_wrist(solved_digit_angles):
        full_angles = {"wrist": true_angles["wrist"], **solved_digit_angles}
        q = pik.configuration_from_angles(fm_full, full_angles)
        return pik.forward_kinematics_positions(fm_full, q)

    # (1) Naive: solve the digit chain directly against the ground-truth
    # world targets, using a REST-metadata model that (wrongly) roots `root`
    # at the universe with identity rotation.
    naive_solved, naive_err = fik.solve_digit_chain(
        fm_digit, digit, target1_point=target1, target2_point=target2, tip_point=tip,
        initial_angles=zero_angles, max_iterations=400, tolerance=0.01,
    )
    # It need not converge to the same tight tolerance as the corrected solve
    # below (the wrong frame can leave some residual even with excess DOF),
    # but it's not wildly off in its OWN (wrong) frame either -- the point of
    # this test is that this apparent near-convergence is misleading once
    # the angles are actually applied under the wrist's true rotation, below.
    assert naive_err["hinge3"] < 0.75

    naive_positions_applied = _apply_digit_solution_under_true_wrist(naive_solved)
    naive_real_error = float(np.linalg.norm(naive_positions_applied["hinge3"] - target2))
    assert naive_real_error > 1.0, (
        "naive (uncorrected) digit solve should be substantially wrong once its angles "
        "are actually applied under the wrist's true rotation -- if this fails, the "
        "synthetic setup no longer exercises the frame bug"
    )

    # (2) Fixed: pre-rotate the targets into the wrist's REST frame before solving.
    solved_angles_tuple = {"wrist": (35.0, -50.0, 65.0)}
    wrist_rotation = fik.wrist_world_rotation(all_meta, [], "wrist", solved_angles_tuple)
    wrist_rest_world = (0.0, 0.0, 0.0)
    wrist_posed_world = positions_true["wrist"]

    corrected_target1 = fik.rotate_target_into_rest_wrist_frame(
        target1, wrist_rest_world, wrist_posed_world, wrist_rotation)
    corrected_target2 = fik.rotate_target_into_rest_wrist_frame(
        target2, wrist_rest_world, wrist_posed_world, wrist_rotation)
    corrected_tip = fik.rotate_target_into_rest_wrist_frame(
        tip, wrist_rest_world, wrist_posed_world, wrist_rotation)

    fixed_solved, fixed_err = fik.solve_digit_chain(
        fm_digit, digit,
        target1_point=corrected_target1, target2_point=corrected_target2, tip_point=corrected_tip,
        initial_angles=zero_angles, max_iterations=150, tolerance=0.01,
    )
    assert fixed_err["hinge3"] < 0.05

    fixed_positions_applied = _apply_digit_solution_under_true_wrist(fixed_solved)
    fixed_real_error = float(np.linalg.norm(fixed_positions_applied["hinge3"] - target2))
    assert fixed_real_error < 0.05, (
        "frame-corrected digit solve should reproduce the true target position closely "
        "once applied under the wrist's true rotation"
    )
    assert fixed_real_error < naive_real_error / 10.0


def _fake_hand_landmarks(index_mcp=(0.02, -0.01, 0.0), pinky_mcp=(0.03, 0.015, 0.005), extra=None):
    """21-entry fake landmark list (MediaPipe HandLandmarker shape) with only
    WRIST(0)/INDEX_MCP(5)/PINKY_MCP(17) set to meaningful values -- the rest
    zeroed, since `anchor_hand_landmarks` only uses those three plus whatever
    `extra` overrides for a given test.
    """
    landmarks = [(0.0, 0.0, 0.0)] * 21
    landmarks[fik.WRIST] = (0.0, 0.0, 0.0)
    landmarks[fik.INDEX_MCP] = index_mcp
    landmarks[fik.PINKY_MCP] = pinky_mcp
    if extra:
        for i, v in extra.items():
            landmarks[i] = v
    return landmarks


def _identity_rest_geometry(hand_landmarks, mp_hip_mid, daz_hip_world, unit_scale):
    """REST rig geometry that exactly matches this fake hand's own
    transformed WRIST/INDEX_MCP/PINKY_MCP triangle, so `kabsch_rotation`
    recovers the identity rotation -- isolates these tests to the
    translation behavior, unaffected by the new rotation-correction step.
    """
    wrist_t = np.array(to_daz_world(hand_landmarks[fik.WRIST], mp_hip_mid, daz_hip_world, unit_scale))
    index_t = np.array(to_daz_world(hand_landmarks[fik.INDEX_MCP], mp_hip_mid, daz_hip_world, unit_scale))
    pinky_t = np.array(to_daz_world(hand_landmarks[fik.PINKY_MCP], mp_hip_mid, daz_hip_world, unit_scale))
    return tuple(wrist_t), tuple(index_t), tuple(pinky_t)


def test_anchor_hand_landmarks_places_wrist_exactly_at_given_position():
    hand_landmarks = _fake_hand_landmarks()
    mp_hip_mid = (0.1, 0.2, 0.3)
    daz_hip_world = (5.0, 90.0, -2.0)
    unit_scale = 100.0
    wrist_world = (10.0, 95.0, 3.0)
    rest_wrist, rest_index, rest_pinky = _identity_rest_geometry(
        hand_landmarks, mp_hip_mid, daz_hip_world, unit_scale,
    )

    anchored = fik.anchor_hand_landmarks(
        hand_landmarks, mp_hip_mid, daz_hip_world, unit_scale, wrist_world,
        rest_wrist, rest_index, rest_pinky,
    )

    np.testing.assert_allclose(anchored[0], np.array(wrist_world), atol=1e-9)
    assert len(anchored) == 21


def test_anchor_hand_landmarks_preserves_relative_scaled_offsets():
    hand_landmarks = _fake_hand_landmarks(extra={1: (0.02, 0.0, 0.0)})
    mp_hip_mid = (0.0, 0.0, 0.0)
    daz_hip_world = (0.0, 0.0, 0.0)
    unit_scale = 100.0
    wrist_world = (0.0, 0.0, 0.0)
    rest_wrist, rest_index, rest_pinky = _identity_rest_geometry(
        hand_landmarks, mp_hip_mid, daz_hip_world, unit_scale,
    )

    anchored = fik.anchor_hand_landmarks(
        hand_landmarks, mp_hip_mid, daz_hip_world, unit_scale, wrist_world,
        rest_wrist, rest_index, rest_pinky,
    )

    # landmark[1] is +0.02 in x relative to landmark[0] (the wrist) in
    # MediaPipe space; to_daz_world's x sign is unflipped and unit_scale=100
    # -> expect +2.0 in DAZ x relative to the anchored wrist.
    np.testing.assert_allclose(anchored[1] - anchored[0], np.array([2.0, 0.0, 0.0]), atol=1e-9)


def test_kabsch_rotation_recovers_known_rotation():
    # NOTE: assertions here deliberately avoid np.linalg.det/svd/inv and any
    # 2-D `@` (matrix-matrix multiply) -- all of those crash with an
    # illegal-instruction fault on this project's numpy/OpenBLAS build on
    # this machine (see kabsch_rotation's docstring, bd daz-script-server-ap3l).
    # Matrix-vector products, np.dot on 1-D vectors, and np.cross are unaffected.
    source_vectors = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    R_true = pik._axis_rotation("Z", 37.0)
    target_vectors = np.array([R_true @ v for v in source_vectors])

    R = fik.kabsch_rotation(source_vectors, target_vectors)

    np.testing.assert_allclose(R, R_true, atol=1e-9)
    # Recovered R must be a proper rotation (orthonormal columns, det=+1),
    # not a reflection -- checked via cross/dot instead of det()/R@R.T.
    columns = [R[:, i] for i in range(3)]
    for c in columns:
        assert np.linalg.norm(c) == pytest.approx(1.0, abs=1e-9)
    assert np.dot(columns[0], columns[1]) == pytest.approx(0.0, abs=1e-9)
    assert np.dot(columns[0], columns[2]) == pytest.approx(0.0, abs=1e-9)
    triple_product = np.dot(columns[0], np.cross(columns[1], columns[2]))
    assert triple_product == pytest.approx(1.0, abs=1e-9)


def test_anchor_hand_landmarks_corrects_wrong_handed_orientation():
    # Rig's REST geometry: index/pinky MCPs spread out along +x/-x from the
    # wrist (a plausible flat-hand rest pose).
    rest_wrist = (0.0, 90.0, 0.0)
    rest_index = (2.0, 90.0, 0.0)
    rest_pinky = (-2.0, 90.0, -0.5)

    # Fake MediaPipe hand whose transformed WRIST/INDEX_MCP/PINKY_MCP
    # triangle is the REST triangle rotated 90 degrees about z -- the same
    # "wrong-handed" shape the live validation found (fingers landing on
    # the wrong side of the wrist relative to the rig's own geometry).
    R_wrong = pik._axis_rotation("Z", 90.0)
    mp_hip_mid = (0.0, 0.0, 0.0)
    daz_hip_world = (0.0, 90.0, 0.0)
    unit_scale = 1.0
    index_mp = R_wrong @ (np.array(rest_index) - np.array(rest_wrist))
    pinky_mp = R_wrong @ (np.array(rest_pinky) - np.array(rest_wrist))
    hand_landmarks = _fake_hand_landmarks(index_mcp=tuple(index_mp), pinky_mcp=tuple(pinky_mp))

    wrist_world = (10.0, 95.0, 3.0)
    anchored = fik.anchor_hand_landmarks(
        hand_landmarks, mp_hip_mid, daz_hip_world, unit_scale, wrist_world,
        rest_wrist, rest_index, rest_pinky,
    )

    # After correction, the anchored INDEX_MCP/PINKY_MCP should sit in the
    # same *direction* from the anchored wrist as the rig's own REST
    # geometry -- the 90-degree misalignment must be undone, not baked in.
    wrist_anchored = anchored[fik.WRIST]
    index_dir = (anchored[fik.INDEX_MCP] - wrist_anchored)
    index_dir /= np.linalg.norm(index_dir)
    expected_dir = (np.array(rest_index) - np.array(rest_wrist))
    expected_dir /= np.linalg.norm(expected_dir)
    np.testing.assert_allclose(index_dir, expected_dir, atol=1e-6)
