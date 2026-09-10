from __future__ import annotations

import numpy as np
import pytest

import finger_ik as fik


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
