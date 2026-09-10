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


def test_anchor_hand_landmarks_places_wrist_exactly_at_given_position():
    hand_landmarks = [(0.0, 0.0, 0.0), (0.02, -0.01, 0.0), (0.04, -0.02, 0.0)]  # fake, only need [0]
    mp_hip_mid = (0.1, 0.2, 0.3)
    daz_hip_world = (5.0, 90.0, -2.0)
    unit_scale = 100.0
    wrist_world = (10.0, 95.0, 3.0)

    anchored = fik.anchor_hand_landmarks(hand_landmarks, mp_hip_mid, daz_hip_world, unit_scale, wrist_world)

    np.testing.assert_allclose(anchored[0], np.array(wrist_world), atol=1e-9)
    assert len(anchored) == 3


def test_anchor_hand_landmarks_preserves_relative_scaled_offsets():
    hand_landmarks = [(0.0, 0.0, 0.0), (0.02, 0.0, 0.0)]
    mp_hip_mid = (0.0, 0.0, 0.0)
    daz_hip_world = (0.0, 0.0, 0.0)
    unit_scale = 100.0
    wrist_world = (0.0, 0.0, 0.0)

    anchored = fik.anchor_hand_landmarks(hand_landmarks, mp_hip_mid, daz_hip_world, unit_scale, wrist_world)

    # landmark[1] is +0.02 in x relative to landmark[0] (the wrist) in
    # MediaPipe space; to_daz_world's x sign is unflipped and unit_scale=100
    # -> expect +2.0 in DAZ x relative to the anchored wrist.
    np.testing.assert_allclose(anchored[1] - anchored[0], np.array([2.0, 0.0, 0.0]), atol=1e-9)
