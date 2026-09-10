# Finger Posing for pose_transfer_photo — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `--fingers` flag to `pose_transfer_photo.py` that poses each detected hand's fingers from the same photo, after the existing body IK solve.

**Architecture:** A new `finger_ik.py` module holds a per-digit chain table (10 entries: 5 digits × 2 hands) and reuses `pinocchio_ik.py`'s `build_figure_model`/`solve_ik` unchanged to IK-solve every bone in a digit except the last; the last bone's single free-axis hinge angle is computed directly from landmark geometry (it has no observable position, so IK cannot target it). A new `hand_landmarks.py` module extracts MediaPipe HandLandmarker world landmarks, mirroring the existing `extract_world_landmarks()` pattern for the body. `pose_transfer_photo.py` gains a small amount of orchestration wiring: anchor each hand's landmarks to its already-solved wrist position, then solve and push all 10 possible digit chains (skipping any hand MediaPipe didn't detect).

**Tech Stack:** Python, mediapipe (HandLandmarker Tasks API), numpy, Pinocchio (`pin`, via the `pinocchio-ik` conda env), dazpy, pytest.

**Spec:** `docs/superpowers/specs/2026-09-10-finger-posing-design.md`

## Global Constraints

- `--fingers` requires `--backend pinocchio`; combined with `--backend stacked` it must `sys.exit` with a clear error, not silently ignore the flag.
- `--fingers` requires wrist IK to be enabled; combined with `--no-hands` it must `sys.exit` with a clear error.
- A hand MediaPipe fails to detect (or reports as not present) is skipped entirely for finger posing — its fingers are left untouched, reported in the summary, never a hard failure.
- All finger-solving code lives in `finger_ik.py` and reuses `pinocchio_ik.build_figure_model`/`solve_ik` as-is — no changes to `pinocchio_ik.py` in this plan.
- New pure-function tests run with the `pinocchio-ik` conda env (same as `test_pinocchio_ik.py`): `export PATH="/x/apps/miniforge3/envs/pinocchio-ik/Library/bin:$PATH"` then `x:/apps/miniforge3/envs/pinocchio-ik/python.exe -m pytest <file> -v` (prepending `Library/bin` to PATH is required — invoking that `python.exe` directly without it segfaults numpy's BLAS on Windows, see `pinocchio_ik.py`'s module docstring).

---

## Confirmed Genesis 9 rig data (verified live against "Jason Cross" — do not re-derive)

Every finger's last two bones (`*2`, `*3`) are pure single-axis hinges about **z**; the thumb's last two bones (`thumb2`, `thumb3`) are pure single-axis hinges about **y**. This is uniform across `index`/`mid`/`ring`/`pinky`, both `l_*` and `r_*`:

```
l_index2/l_index3, l_mid2/l_mid3, l_ring2/l_ring3, l_pinky2/l_pinky3: only 'z' free (x,y both min==max==0)
l_thumb2/l_thumb3: only 'y' free (x,z both min==max==0)
```

(Same for `r_*`.) This confirms the plan's "IK all-but-last, geometry for last" split applies uniformly to all 5 digits.

MediaPipe HandLandmarker's 21 landmarks (standard indices, used throughout):

```
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20
```

---

### Task 1: Digit chain table

**Files:**
- Create: `ai_vision/pose_transfer_photo/finger_ik.py`
- Test: `ai_vision/pose_transfer_photo/test_finger_ik.py`

**Interfaces:**
- Produces: `DigitChain` dataclass (fields: `chain_bones: list[str]`, `target1_landmark: int`, `target2_landmark: int`, `tip_landmark: int`, `tip_hinge_axis: str`; property `effector_bones -> list[str]` returning `chain_bones[-2:]`); `DIGIT_CHAINS: dict[str, DigitChain]` keyed `"l_index"`, `"l_mid"`, `"l_ring"`, `"l_pinky"`, `"l_thumb"`, `"r_index"`, ... (10 keys total).

- [ ] **Step 1: Write the failing test**

```python
# test_finger_ik.py
from __future__ import annotations

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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `export PATH="/x/apps/miniforge3/envs/pinocchio-ik/Library/bin:$PATH" && "x:/apps/miniforge3/envs/pinocchio-ik/python.exe" -m pytest ai_vision/pose_transfer_photo/test_finger_ik.py -v` (from the `daz-script-server-examples` repo root)
Expected: FAIL/ERROR with `ModuleNotFoundError: No module named 'finger_ik'`

- [ ] **Step 3: Write minimal implementation**

```python
# finger_ik.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: same command as Step 2
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add ai_vision/pose_transfer_photo/finger_ik.py ai_vision/pose_transfer_photo/test_finger_ik.py
git commit -m "finger_ik: add per-digit chain table (10 digits, both hands)"
```

---

### Task 2: Hinge angle from landmark geometry

**Files:**
- Modify: `ai_vision/pose_transfer_photo/finger_ik.py`
- Test: `ai_vision/pose_transfer_photo/test_finger_ik.py`

**Interfaces:**
- Consumes: nothing new from Task 1 (pure numpy function).
- Produces: `hinge_angle_from_vectors(v_in: np.ndarray, v_out: np.ndarray, axis: str) -> float` — signed angle in degrees, about `axis` ("x"/"y"/"z"), that rotates `v_in` toward `v_out`, using the same right-hand-rule sign convention as `pinocchio_ik._axis_rotation`.

- [ ] **Step 1: Write the failing test**

```python
# append to test_finger_ik.py
import pytest


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: same pytest command as before, `-k hinge_angle`
Expected: FAIL with `AttributeError: module 'finger_ik' has no attribute 'hinge_angle_from_vectors'`

- [ ] **Step 3: Write minimal implementation**

Append to `finger_ik.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: same pytest command
Expected: PASS (11 new parametrized cases + 1)

- [ ] **Step 5: Commit**

```bash
git add ai_vision/pose_transfer_photo/finger_ik.py ai_vision/pose_transfer_photo/test_finger_ik.py
git commit -m "finger_ik: add hinge_angle_from_vectors for the tip bone's geometry-only DOF"
```

---

### Task 3: `solve_digit_chain` — IK + geometry, end to end per digit

**Files:**
- Modify: `ai_vision/pose_transfer_photo/finger_ik.py`
- Test: `ai_vision/pose_transfer_photo/test_finger_ik.py`

**Interfaces:**
- Consumes: `DigitChain` (Task 1), `hinge_angle_from_vectors` (Task 2), `pinocchio_ik.build_figure_model`, `pinocchio_ik.solve_ik`, `pinocchio_ik.clamp_angles`, `pinocchio_ik._rest_orientation_matrix` (all pre-existing, unchanged).
- Produces: `solve_digit_chain(fm: pik.FigureModel, digit: DigitChain, target1_point: np.ndarray, target2_point: np.ndarray, tip_point: np.ndarray, initial_angles: dict[str, dict[str, float]], **solve_ik_kwargs) -> tuple[dict[str, dict[str, float]], dict[str, float]]` — same return shape as `pik.solve_ik` (angles dict keyed by every bone in `digit.chain_bones`, error dict keyed by `digit.effector_bones`).

**Design notes for the implementer:**

The tip bone's hinge angle is derived from three world-space points: `target1_point` (= this digit's `chain_bones[-2]`'s target position, e.g. PIP), `target2_point` (= `chain_bones[-1]`'s target position, e.g. DIP — also the hinge joint's own location), and `tip_point` (e.g. TIP — has no corresponding bone). The physical model: a hinge bends the outgoing segment direction relative to the incoming segment direction, by a rotation about its one free axis. Concretely:

1. Solve `chain_bones[:-1]` (all but the last) via `pik.solve_ik` with `effector_bones=digit.effector_bones` and `target_points=np.array([target1_point, target2_point])` — this is a completely standard call, no changes to `solve_ik` needed.
2. Re-run `pin.forwardKinematics(fm.model, fm.data, q_solved)` (where `q_solved = pik.configuration_from_angles(fm, solved_angles)`) to read `R_ref = fm.data.oMi[fm.joint_of[digit.chain_bones[-2]]].rotation` — the world orientation immediately proximal to the tip bone's own local rotation.
3. `v_in_world = target2_point - target1_point`; `v_out_world = tip_point - target2_point`.
4. Transform both into the tip bone's own rest-orientation-corrected frame: `O_tip = pik._rest_orientation_matrix(fm.by_name[digit.chain_bones[-1]])`; `w_in = O_tip @ (R_ref.T @ v_in_world)`; `w_out = O_tip @ (R_ref.T @ v_out_world)`.
5. `theta = hinge_angle_from_vectors(w_in, w_out, digit.tip_hinge_axis)`.
6. Set `solved_angles[digit.chain_bones[-1]][digit.tip_hinge_axis] = theta` (the other two axes of that bone are already 0 from the IK step, since they're degenerate/dead per the confirmed rig data above), then clamp the whole `solved_angles` dict with `pik.clamp_angles(fm, solved_angles)`.
7. Return `(clamped_angles, final_error_from_step_1)` — the geometry step doesn't change the IK's own reported convergence error (it only ever affected an untargeted DOF).

- [ ] **Step 1: Write the failing test**

```python
# append to test_finger_ik.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: same pytest command, `-k solve_digit_chain`
Expected: FAIL with `AttributeError: module 'finger_ik' has no attribute 'solve_digit_chain'`

- [ ] **Step 3: Write minimal implementation**

Append to `finger_ik.py`:

```python
import pinocchio as pin


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
```

- [ ] **Step 4: Run test to verify it passes**

Run: same pytest command
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add ai_vision/pose_transfer_photo/finger_ik.py ai_vision/pose_transfer_photo/test_finger_ik.py
git commit -m "finger_ik: add solve_digit_chain combining IK with tip-hinge geometry"
```

---

### Task 4: Hand landmark extraction

**Files:**
- Create: `ai_vision/pose_transfer_photo/hand_landmarks.py`

**Interfaces:**
- Produces: `extract_hand_world_landmarks(image_path: str) -> dict[str, list[tuple[float, float, float]]]` — keys are `"Left"`/`"Right"` (only for hands MediaPipe actually detected; a dict with 0, 1, or 2 keys), each value a list of 21 `(x, y, z)` tuples in MediaPipe's metric hand-world-landmark space (same units/convention as `pose_transfer_photo.extract_world_landmarks`'s pose landmarks, per the design spec's "assume shared convention" decision).

**No unit test for this task** — this mirrors the existing, untested precedent of `pose_transfer_photo.extract_world_landmarks()` in the same file, which also isn't unit-tested: both call into MediaPipe's Tasks API against a real image and a downloaded model file, making them integration-only by nature. This function is exercised live in Task 6's validation.

- [ ] **Step 1: Write the implementation**

```python
# hand_landmarks.py
"""MediaPipe HandLandmarker extraction for pose_transfer_photo.py's
--fingers flag. Mirrors extract_world_landmarks()'s pattern in
pose_transfer_photo.py (auto-downloaded model, Tasks API, world landmarks).
"""
from __future__ import annotations

import os
import sys
import urllib.request

import cv2
import mediapipe as mp

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")


def _ensure_model() -> str:
    if not os.path.exists(_MODEL_PATH):
        print(f"Downloading hand landmarker model -> {_MODEL_PATH}")
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
    return _MODEL_PATH


def extract_hand_world_landmarks(image_path: str) -> dict[str, list[tuple[float, float, float]]]:
    """Decode an image and return each detected hand's 21 metric-scale world
    landmarks, keyed by MediaPipe's own handedness label ("Left"/"Right" --
    per MediaPipe's documented default convention this is the subject's own
    anatomical hand, matching pose_transfer_photo.py's existing assumption
    for body landmarks; see Task 6's live validation note if a run's
    resulting hand poses look mirrored).

    A hand MediaPipe doesn't detect (occluded, out of frame, low confidence)
    is simply absent from the returned dict -- not an error, since the
    caller (pose_transfer_photo.py) treats a missing hand as "skip this
    hand's fingers," matching how a missed body limb is already handled.

    Raises SystemExit only if the image itself cannot be loaded (same
    failure mode as extract_world_landmarks) -- zero hands detected is not
    an error here, unlike the body pose case, since photos are commonly
    framed to show a face/body clearly but crop or blur one or both hands.
    """
    img = cv2.imread(image_path)
    if img is None:
        sys.exit(f"Cannot load image: {image_path!r}")
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    HandLandmarker        = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    BaseOptions           = mp.tasks.BaseOptions
    RunningMode           = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=_ensure_model()),
        running_mode=RunningMode.IMAGE,
        num_hands=2,
    )

    with HandLandmarker.create_from_options(options) as landmarker:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_image)

    hands: dict[str, list[tuple[float, float, float]]] = {}
    for handedness, world_landmarks in zip(result.handedness, result.hand_world_landmarks):
        label = handedness[0].category_name  # "Left" or "Right"
        hands[label] = [(lm.x, lm.y, lm.z) for lm in world_landmarks]
    return hands
```

- [ ] **Step 2: Manually smoke-test against a real image**

Run (adjust the image path to any photo with at least one visible hand):

```bash
export PATH="/x/apps/miniforge3/envs/pinocchio-ik/Library/bin:$PATH"
cd ai_vision/pose_transfer_photo
"x:/apps/miniforge3/envs/pinocchio-ik/python.exe" -c "
from hand_landmarks import extract_hand_world_landmarks
hands = extract_hand_world_landmarks('d:/ghirp/Downloaded Content/body2.png')
print(list(hands.keys()))
for label, lms in hands.items():
    print(label, 'landmark count:', len(lms), 'wrist:', lms[0])
"
```

Expected: no traceback; prints 0, 1, or 2 hand labels depending on whether `body2.png` shows hands clearly enough (it's a body photo — this may well come back empty; if so, note that a hands-visible test photo is needed for Task 6 and flag it then rather than treating it as a Task 4 failure).

- [ ] **Step 3: Commit**

```bash
git add ai_vision/pose_transfer_photo/hand_landmarks.py
git commit -m "hand_landmarks: add MediaPipe HandLandmarker extraction"
```

---

### Task 5: Anchor hand landmarks to the solved wrist position

**Files:**
- Modify: `ai_vision/pose_transfer_photo/finger_ik.py`
- Test: `ai_vision/pose_transfer_photo/test_finger_ik.py`

**Interfaces:**
- Consumes: `pose_transfer_photo.to_daz_world(landmark, mp_hip_mid, daz_hip_world, unit_scale)` (pre-existing, unchanged) — imported into `finger_ik.py`.
- Produces: `anchor_hand_landmarks(hand_landmarks: list[tuple[float, float, float]], mp_hip_mid: tuple[float, float, float], daz_hip_world: tuple[float, float, float], unit_scale: float, wrist_world: tuple[float, float, float]) -> list[np.ndarray]` — applies the body pipeline's existing scale+rotation transform to all 21 landmarks, then translates the whole set so the transformed wrist landmark (index 0) lands exactly on `wrist_world`. Returns a list of 21 length-3 `np.ndarray`s in DAZ world space.

- [ ] **Step 1: Write the failing test**

```python
# append to test_finger_ik.py
from pose_transfer_photo import to_daz_world


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: same pytest command, `-k anchor_hand_landmarks`
Expected: FAIL with `AttributeError: module 'finger_ik' has no attribute 'anchor_hand_landmarks'`

- [ ] **Step 3: Write minimal implementation**

Append to `finger_ik.py`:

```python
from pose_transfer_photo import to_daz_world


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
```

- [ ] **Step 4: Run test to verify it passes**

Run: same pytest command
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add ai_vision/pose_transfer_photo/finger_ik.py ai_vision/pose_transfer_photo/test_finger_ik.py
git commit -m "finger_ik: anchor hand landmarks to the already-solved wrist position"
```

---

### Task 6: Wire `--fingers` into pose_transfer_photo.py

**Files:**
- Modify: `ai_vision/pose_transfer_photo/pose_transfer_photo.py`

**Interfaces:**
- Consumes: `finger_ik.DIGIT_CHAINS`, `finger_ik.solve_digit_chain`, `finger_ik.anchor_hand_landmarks`, `hand_landmarks.extract_hand_world_landmarks` (all from prior tasks).
- Produces: nothing further downstream — this is the final integration point.

**No unit test for this task** — CLI/orchestration wiring in a script that's already integration-tested by convention (see Task 4's note); validated live in this task's Step 2 and formally in Task 7.

- [ ] **Step 1: Add the flag, validation, and orchestration**

Add near the other flags (after `--no-feet`, in the `if __name__ == "__main__":` block):

```python
    parser.add_argument("--fingers", action="store_true",
                        help="Also pose fingers from the same photo via MediaPipe HandLandmarker "
                             "(requires --backend pinocchio and wrist targets enabled)")
```

Right after `args = parser.parse_args()`, add:

```python
    if args.fingers and args.backend != "pinocchio":
        sys.exit("--fingers requires --backend pinocchio (per-finger IK is too slow over the "
                  "HTTP stacked backend -- see finger_ik.py's module docstring).")
    if args.fingers and not args.hands:
        sys.exit("--fingers requires wrist targets to be enabled (fingers are anchored to the "
                  "solved wrist position) -- remove --no-hands.")
```

After the existing body-IK block (right after the `print(f"\nFinal per-limb error...")` loop, before the final `print(f"\nDone. ...")` line), add:

```python
    if args.fingers:
        import finger_ik as fik
        from hand_landmarks import extract_hand_world_landmarks

        print("\nExtracting hand landmarks for finger posing...")
        detected_hands = extract_hand_world_landmarks(args.image)

        post_solve_meta = {b["name"]: b for b in figure.bone_metadata()}
        finger_errors: dict[str, dict[str, float]] = {}
        all_solved_angles: dict[str, tuple[float, float, float]] = {}

        side_by_label = {"Left": "l", "Right": "r"}
        for hand_label, side in side_by_label.items():
            if hand_label not in detected_hands:
                print(f"  {hand_label} hand: SKIPPED (not detected)")
                continue

            wrist_bone = f"{side}_hand"
            wrist_world = _world(wrist_bone) if wrist_bone in by_name else (
                post_solve_meta[wrist_bone]["world_position"]["x"],
                post_solve_meta[wrist_bone]["world_position"]["y"],
                post_solve_meta[wrist_bone]["world_position"]["z"],
            )
            anchored = fik.anchor_hand_landmarks(
                detected_hands[hand_label], mp_hip_mid, daz_hip_world, unit_scale, wrist_world,
            )

            for digit_name in ("index", "mid", "ring", "pinky", "thumb"):
                digit = fik.DIGIT_CHAINS[f"{side}_{digit_name}"]
                digit_meta = [post_solve_meta[b] for b in digit.chain_bones]
                fm = fik.pik.build_figure_model(list(post_solve_meta.values()), digit.chain_bones)
                current_rot = figure.bone_rotations()
                initial_angles = {
                    b: {"x": current_rot[b][0], "y": current_rot[b][1], "z": current_rot[b][2]}
                    for b in digit.chain_bones
                }
                solved, err = fik.solve_digit_chain(
                    fm, digit,
                    target1_point=np.array(anchored[digit.target1_landmark]),
                    target2_point=np.array(anchored[digit.target2_landmark]),
                    tip_point=np.array(anchored[digit.tip_landmark]),
                    initial_angles=initial_angles,
                    max_iterations=args.max_iterations, tolerance=args.tolerance,
                    damping=args.damping, rest_pose_weight=args.rest_pose_weight,
                    max_step_degrees=args.step_degrees, debug=args.debug,
                )
                finger_errors[f"{side}_{digit_name}"] = err
                for bone, a in solved.items():
                    all_solved_angles[bone] = (a["x"], a["y"], a["z"])

        if all_solved_angles:
            with scene.undo("Apply photo finger pose"):
                figure.set_bone_rotations(all_solved_angles)

        if finger_errors:
            print("\nFinal per-digit error (scene units):")
            for name, err in finger_errors.items():
                worst = max(err.values())
                status = "OK" if worst <= args.tolerance else "NOT CONVERGED"
                print(f"  {name:10s} {worst:.3f}   [{status}]")
```

**Note for the implementer:** `fik.pik` refers to `pinocchio_ik` re-exported through `finger_ik`'s own `import pinocchio_ik as pik` — reaching it as `fik.pik.build_figure_model` avoids a second top-level import line in `pose_transfer_photo.py`; if this reads awkwardly during review, an equally fine alternative is adding `import pinocchio_ik as pik` alongside the `import finger_ik as fik` line above and calling `pik.build_figure_model` directly. Either is fine; keep whichever the reviewer prefers.

- [ ] **Step 2: Live smoke-test the wiring**

Find or take a test photo with at least one hand clearly visible and reasonably unoccluded (per Task 4 Step 2's note, `body2.png` may not qualify — check its `extract_hand_world_landmarks` output first; if empty, use a different photo for this step and record which one worked, since Task 7's live validation will need the same one).

```bash
export PATH="/x/apps/miniforge3/envs/pinocchio-ik/Library/bin:$PATH"
# zero the figure first (same as any other run)
"x:/apps/miniforge3/envs/pinocchio-ik/python.exe" -c "
from dazpy import DazClient, DazScene
from dazpy.poses import zero_figure
client = DazClient(); scene = DazScene(client)
zero_figure(scene.find_skeleton_by_label('Jason Cross'))
"
cd ai_vision/pose_transfer_photo
"x:/apps/miniforge3/envs/pinocchio-ik/python.exe" pose_transfer_photo.py <hands-visible-photo> \
  --figure "Jason Cross" --backend pinocchio --fingers --debug
```

Expected: no traceback; a "Final per-digit error" section prints for each detected hand's 5 digits, each `[OK]` or `[NOT CONVERGED]` (a few `NOT CONVERGED` on tight tolerance is fine — this step confirms the wiring runs end-to-end, not that convergence is perfect). If it crashes, the most likely causes worth checking first: (a) `by_name`/`post_solve_meta` bone-name typos, (b) `unit_scale`/`mp_hip_mid`/`daz_hip_world` not in scope at that point in `main` (they're defined earlier in the same `if __name__` block, above the body IK call — confirm nothing renamed them).

- [ ] **Step 3: Commit**

```bash
git add ai_vision/pose_transfer_photo/pose_transfer_photo.py
git commit -m "pose_transfer_photo: wire --fingers flag into the CLI and main flow"
```

---

### Task 7: Live validation and coordinate-alignment check

**Files:** none (validation only; may produce a follow-up bd note or issue if the coordinate-alignment assumption fails)

- [ ] **Step 1: Run the full pipeline with `--fingers` against a hands-visible photo**

Same command as Task 6 Step 2. Record the per-digit convergence errors.

- [ ] **Step 2: Visually inspect the resulting hand pose in DAZ Studio**

Check specifically for the risk the spec calls out: fingers curling or splaying in a *physically wrong direction* (not just imprecise) — e.g. fingers bending backward, or index/pinky swapped in apparent position. Distinguish this from ordinary IK residual (a finger that's slightly less curled than the photo but bending the *correct way* is fine and expected).

- [ ] **Step 3a: If orientation looks correct**

Record results (which photo, per-digit errors, screenshot or description of the pose) in bd issue notes (create a new bd issue for this feature if one doesn't already exist, e.g. under the same epic as `daz-script-server-hewu`, or ask the user which issue to attach notes to). Done — no further code changes needed from this plan.

- [ ] **Step 3b: If orientation looks wrong (fingers bend the wrong way / hands mirrored)**

This falsifies the "MediaPipe Pose and Hand world landmarks share a rotational convention" assumption from the spec. Do not attempt to patch this by guessing sign flips. Instead: file a bd issue documenting exactly what was observed (which digits, which direction, screenshot), and treat "add per-hand Kabsch/Procrustes alignment" (using WRIST + INDEX_MCP + PINKY_MCP as a stable corresponding triangle, per the spec's fallback plan) as new follow-up work requiring its own brainstorming pass — it's a real algorithmic addition, not a quick fix, and changes the anchoring approach `finger_ik.anchor_hand_landmarks` implements.

---

## Self-review notes (for whoever executes this plan)

- **Spec coverage:** digit chain table (Task 1) ✓, per-axis limit data confirmed live and baked into Task 1's test rather than re-derived ✓, hinge-from-geometry (Task 2/3) ✓, hand landmark extraction (Task 4) ✓, coordinate alignment + anchoring (Task 5) with staged live validation (Task 7) ✓, CLI flag + validation errors (Task 6) ✓, error handling / skip-on-missing-hand (Task 6) ✓, out-of-scope batch pipeline correctly excluded ✓.
- **Known loose end:** Task 6's orchestration snippet re-fetches `figure.bone_rotations()` and rebuilds a fresh `FigureModel` once per digit (10 times per run) rather than batching all digits into one model — this mirrors Task 3's per-digit-chain design (chosen for simplicity, since digits don't share intermediate bones) but is worth flagging to the reviewer as a possible spot to simplify further if it reads as repetitive; not a correctness issue.
