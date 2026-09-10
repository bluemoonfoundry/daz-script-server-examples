"""Tests for the active-set joint-limit handling in pinocchio_ik.py (bd daz-script-server-hewu).

Run with the pinocchio-ik conda env (has real Pinocchio + pytest):
    conda run -n pinocchio-ik python -m pytest test_pinocchio_ik.py -v
"""
from __future__ import annotations

import numpy as np
import pytest

import pinocchio_ik as pik


# ── helpers to build synthetic bone_metadata without live DAZ Studio ───────

def _bone_meta(name, parent_name, world_position, rotation_order="XYZ",
                rest_orientation=(1.0, 0.0, 0.0, 0.0), axis_limits=None):
    w, x, y, z = rest_orientation
    return {
        "name": name,
        "parent_name": parent_name,
        "world_position": {"x": world_position[0], "y": world_position[1], "z": world_position[2]},
        "rotation_order": rotation_order,
        "rest_orientation": {"w": w, "x": x, "y": y, "z": z},
        "axis_limits": axis_limits or {
            "x": {"min": -180.0, "max": 180.0},
            "y": {"min": -180.0, "max": 180.0},
            "z": {"min": -180.0, "max": 180.0},
        },
    }


# ── euler_rate_jacobian: analytic vs. finite-difference ─────────────────────

@pytest.mark.parametrize("order", ["XYZ", "ZYX", "YZX"])
@pytest.mark.parametrize("angles", [
    {"x": 0.0, "y": 0.0, "z": 0.0},
    {"x": 20.0, "y": -35.0, "z": 50.0},
    {"x": 90.0, "y": 15.0, "z": -60.0},
])
def test_euler_rate_jacobian_matches_finite_difference(order, angles):
    meta = _bone_meta("b", None, (0, 0, 0), rotation_order=order)
    J = pik.euler_rate_jacobian(meta, angles)
    assert J.shape == (3, 3)

    R0 = pik.bone_local_rotation(meta, angles)
    h = 1e-6  # degrees
    numeric_cols = {}
    for axis in "xyz":
        perturbed = dict(angles)
        perturbed[axis] += h
        R1 = pik.bone_local_rotation(meta, perturbed)
        # Body-frame angular velocity: dR/dt = R @ skew(w)  =>  w = vee(R0^T @ dR)/dtheta
        dR = R0.T @ R1
        h_rad = np.deg2rad(h)
        omega = np.array([
            (dR[2, 1] - dR[1, 2]) / (2 * h_rad),
            (dR[0, 2] - dR[2, 0]) / (2 * h_rad),
            (dR[1, 0] - dR[0, 1]) / (2 * h_rad),
        ])
        numeric_cols[axis] = omega

    numeric_J = np.column_stack([numeric_cols["x"], numeric_cols["y"], numeric_cols["z"]])
    np.testing.assert_allclose(J, numeric_J, atol=1e-3)


# ── per-axis active-set decision ────────────────────────────────────────────

def test_axis_not_locked_when_well_inside_range():
    assert pik.axis_limit_active(current_angle=10.0, min_limit=-90.0, max_limit=90.0, proposed_rate=5.0) is False
    assert pik.axis_limit_active(current_angle=10.0, min_limit=-90.0, max_limit=90.0, proposed_rate=-5.0) is False


def test_axis_locked_when_at_max_and_pushing_further():
    assert pik.axis_limit_active(current_angle=90.0, min_limit=-90.0, max_limit=90.0, proposed_rate=0.5) is True


def test_axis_not_locked_when_at_max_but_pulling_back():
    assert pik.axis_limit_active(current_angle=90.0, min_limit=-90.0, max_limit=90.0, proposed_rate=-0.5) is False


def test_axis_locked_when_at_min_and_pushing_further():
    assert pik.axis_limit_active(current_angle=-90.0, min_limit=-90.0, max_limit=90.0, proposed_rate=-0.5) is True


def test_axis_always_locked_when_range_is_degenerate():
    # e.g. l_forearm's Z axis: min == max == 0, a hinge modeled as spherical.
    assert pik.axis_limit_active(current_angle=0.0, min_limit=0.0, max_limit=0.0, proposed_rate=3.0) is True
    assert pik.axis_limit_active(current_angle=0.0, min_limit=0.0, max_limit=0.0, proposed_rate=-3.0) is True


# ── solve_ik: a locked axis must not be used mid-solve to cheat convergence ─

def _three_bone_chain_with_locked_middle_axis():
    """root (free) -> child (X axis dead: min=max=0, Y/Z free) -> effector (leaf,
    a pure position marker -- its own rotation doesn't affect its own position,
    matching how `_CHAINS` in pose_transfer_photo.py appends the effector bone
    itself to the solved chain)."""
    root = _bone_meta("root", None, (0.0, 0.0, 0.0))
    child = _bone_meta(
        "child", "root", (1.0, 0.0, 0.0),
        axis_limits={
            "x": {"min": 0.0, "max": 0.0},
            "y": {"min": -90.0, "max": 90.0},
            "z": {"min": -90.0, "max": 90.0},
        },
    )
    effector = _bone_meta("effector", "child", (2.0, 0.0, 0.0))
    fm = pik.build_figure_model([root, child, effector], ["root", "child", "effector"])
    return fm


def test_solve_ik_keeps_locked_axis_at_zero_throughout_iteration():
    """The active-set mask must drop the dead axis from the very first
    iteration, not just at the final `clamp_angles` call -- otherwise a
    momentary mid-solve excursion off a limit (invisible in the final
    clamped output) can still have thrown off convergence for the other,
    free axes before being silently snapped back. See bd
    daz-script-server-hewu's "IK solve: works, but NOT limit-aware during
    iteration" note.
    """
    fm = _three_bone_chain_with_locked_middle_axis()
    initial_angles = {
        "root": {"x": 0.0, "y": 0.0, "z": 0.0},
        "child": {"x": 0.0, "y": 0.0, "z": 0.0},
        "effector": {"x": 0.0, "y": 0.0, "z": 0.0},
    }
    # Ground truth generated by forward kinematics with child.x == 0 (its only
    # legal value), so this target is exactly reachable within the limits.
    target = np.array([[1.45372487, -0.13834548, -1.17318265]])

    history: list = []
    solved, err = pik.solve_ik(
        fm, ["effector"], target,
        initial_angles=initial_angles,
        max_iterations=150,
        tolerance=0.05,
        history=history,
    )

    assert len(history) > 5, "solve converged in too few iterations to be a meaningful check"
    for iteration_angles in history:
        assert iteration_angles["child"]["x"] == pytest.approx(0.0, abs=1e-9)

    assert solved["child"]["x"] == pytest.approx(0.0, abs=1e-6)
    assert err["effector"] < 0.2
