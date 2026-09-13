"""DAZ Studio Script Server example: apply a body pose from a photo to a Genesis 9 figure.

PURPOSE
-------
This script is a demonstration of what the DAZ Studio Script Server makes
possible.  It extracts 33 body landmarks from a source image using MediaPipe
PoseLandmarker, converts the wrist/ankle landmarks into DAZ Studio world-space
targets, and drives the figure's hands and feet toward those targets using a
custom multi-effector IK solve: one combined raw-DazScript call per iteration
(see `fundamentals/raw_script/` for the underlying pattern) perturbs every
bone across all four limb chains and reads back all four effector positions
in a single round trip, giving a full stacked Jacobian with cross-limb
coupling included; numpy then solves one damped-least-squares step for all
four targets *simultaneously*, with a null-space secondary objective that
biases unconstrained joints back toward the figure's starting pose.

It is an *example*, not a full-body motion-capture tool.  Only the four limb
effectors (both wrists, both ankles) are driven — elbows/knees follow from the
IK chain solve rather than being matched to the photo directly.  Torso and
head are never touched at all: the chains are deliberately rooted below the
shared `spine4`/`hip` bones (see `_CHAINS` below), so a target beyond the
remaining chain's reach reports NOT CONVERGED instead of quietly recruiting
the spine to compensate.  A real system would solve pelvis/spine orientation
explicitly and account for camera perspective.

An earlier version of this script called dazpy's own single-effector
`DazSkeleton.hand_to_target()`/`.foot_to_target()` once per limb, in
sequence — the same solver `ik_bone_to_target.py` exercises directly.  Their
auto-selected chains include `spine4` (shared by both hands) and `hip`
(shared by both feet, and in fact the skeleton's actual root), so each
limb's solve could disturb a root bone a previously-solved limb depended on,
and worse, a foot-reach solve rotating `hip` tipped the *entire* body —
torso, arms, head included — to gain a little extra leg reach. A
"relaxation passes" workaround (re-solving every limb 2-3 times) reduced the
first problem but not the second, and neither is fixable by iterating harder
since it's the chain composition itself, not the solver, doing the damage.
This version fixes both by construction: solving all four limbs jointly
removes the "previous limb" problem, and rooting the leg chains at `pelvis`
(a sibling of `spine1`, not its ancestor) instead of `hip` means a leg-reach
solve can no longer touch the spine at all.

IMPORTANT: reset the figure to a neutral pose (e.g. `dazpy.poses.zero_figure`,
or DAZ Studio's own Zero Pose) before each run.  Calibration reads the
figure's *current* shoulder-joint and hip world positions, so a leftover pose
from a previous run — or from manual posing — will throw off both the scale
and the IK starting point.

WHAT IT DEMONSTRATES
--------------------
  - Running MediaPipe PoseLandmarker inference in Python (auto-downloads the
    model file on first run), using pose_world_landmarks (metric, hip-centered)
  - Auto-calibrating scale and origin from the live figure's own shoulder width
    and hip position, so results roughly work across different figure heights
    without manual tuning
  - Solving all four limb targets simultaneously with a stacked-Jacobian
    damped-least-squares IK step (one combined raw-DazScript call per
    iteration computes the full cross-limb coupling in a single round trip;
    numpy solves the linear system in Python)
  - Choosing IK chain composition deliberately (excluding shared root bones)
    rather than just tuning solver parameters, when a shared root lets the
    solver "cheat" toward a lower position error at the cost of a wrong pose
  - A null-space secondary objective (rest-pose bias) that keeps DOFs the
    primary task doesn't need close to the starting pose instead of drifting
  - Wrapping the whole pose application in one named undo step
    (scene.undo(...)) so Ctrl+Z in DAZ Studio undoes it in a single step
  - Reporting per-iteration IK convergence for all four effectors at once

ENVIRONMENT SETUP
-----------------
1. DAZ Studio must be running with the DazScriptServer plugin loaded and its
   HTTP server active (default: 127.0.0.1:18811).  You can verify it is
   responding with:

       curl http://127.0.0.1:18811/health

2. Install the Python dependencies in a virtual environment:

       python -m venv .venv
       .venv\\Scripts\\activate          # Windows
       # source .venv/bin/activate     # macOS / Linux
       pip install requests mediapipe opencv-python numpy

3. Install or develop-install the dazpy SDK (from the repo root):

       pip install -e .

4. Open a scene in DAZ Studio containing a Genesis 9 figure (named "Jason
   Cross" by default — override with --figure), then run:

       python ai_vision/pose_transfer_photo/pose_transfer_photo.py photo.jpg

Usage:
    python pose_transfer_photo.py photo.jpg
    python pose_transfer_photo.py photo.jpg --figure "Jason Cross"
    python pose_transfer_photo.py photo.jpg --scale 1.2
    python pose_transfer_photo.py photo.jpg --no-feet
    python pose_transfer_photo.py photo.jpg --max-iterations 150 --damping 0.1 --debug

    # --backend pinocchio: local IK solve, zero HTTP round-trips per iteration.
    # Requires a separate conda-forge environment (not the default venv this
    # script otherwise runs in) -- see environment.yml in this directory, and
    # run via run_pinocchio.ps1 / run_pinocchio.sh rather than invoking this
    # interpreter directly (see pinocchio_ik.py's docstring for why). Worth
    # it once processing many photos x many figures; the model-build cost
    # per unique figure is then amortized (see pinocchio_ik.get_figure_model,
    # bd daz-script-server-hewu).
    ./run_pinocchio.ps1 photo.jpg --backend pinocchio    # or run_pinocchio.sh

    # --batch DIR: process every photo in a folder against one figure,
    # resetting to Zero Pose between each (see bd daz-script-server-hewu).
    # Any combination of --save-poses/--render/--export-mesh/--stats-csv
    # controls what gets written per photo under --output-dir.
    ./run_pinocchio.ps1 --batch photos/ --backend pinocchio \\
        --save-poses --stats-csv --output-dir batch_output/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request

import cv2
import mediapipe as mp
import numpy as np

from dazpy import DazClient, DazRenderSettings, DazScene
from dazpy.exceptions import DazBusyError
from dazpy.poses import zero_figure

# Plausible real-world shoulder width range (meters) used to sanity-check
# MediaPipe's metric pose estimate — see calibrate()'s docstring.
_PLAUSIBLE_SHOULDER_WIDTH_M = (0.20, 0.60)

# ── model setup ────────────────────────────────────────────────────────────────
# mediapipe >= 0.10 uses the Tasks API and requires a model file.
# We download it automatically on first use.
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)
_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pose_landmarker_lite.task")


def _ensure_model() -> str:
    if not os.path.exists(_MODEL_PATH):
        print(f"Downloading pose landmarker model → {_MODEL_PATH}")
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
    return _MODEL_PATH


# ── landmark index constants ───────────────────────────────────────────────────
# MediaPipe Pose canonical indices, anatomical left/right (the person's own
# left/right, matching DAZ Studio's l_*/r_* bone naming directly — no mirroring
# needed for the left/right mapping itself, only for axis sign below).

L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW,    R_ELBOW    = 13, 14
L_WRIST,    R_WRIST    = 15, 16
L_HIP,      R_HIP      = 23, 24
L_KNEE,     R_KNEE     = 25, 26
L_ANKLE,    R_ANKLE    = 27, 28


# ── image → landmarks ──────────────────────────────────────────────────────────

def extract_world_landmarks(image_path: str) -> list[tuple[float, float, float]]:
    """Decode an image and return 33 metric-scale, hip-centered body landmarks.

    Uses MediaPipe's Tasks API `pose_world_landmarks` output — approximate
    real-world meters, origin near the mid-hip point.  Downloads the
    pose_landmarker_lite.task model file to this directory on first run.

    Raises SystemExit if the image cannot be loaded or no pose is detected.
    """
    img = cv2.imread(image_path)
    if img is None:
        sys.exit(f"Cannot load image: {image_path!r}")

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    PoseLandmarker        = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    BaseOptions           = mp.tasks.BaseOptions
    RunningMode           = mp.tasks.vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=_ensure_model()),
        running_mode=RunningMode.IMAGE,
        num_poses=1,
    )

    with PoseLandmarker.create_from_options(options) as landmarker:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_image)

    if not result.pose_world_landmarks:
        sys.exit("No pose detected in image.")

    lms = result.pose_world_landmarks[0]
    return [(lm.x, lm.y, lm.z) for lm in lms]


# ── coordinate mapping ─────────────────────────────────────────────────────────

def _sub(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _dist(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return float(np.linalg.norm(np.array(a) - np.array(b)))


def calibrate(
    landmarks: list[tuple[float, float, float]],
    daz_shoulder_width: float,
    daz_hip_world: tuple[float, float, float],
    scale: float,
) -> tuple[float, tuple[float, float, float], tuple[float, float, float]]:
    """Derive a scale factor and origin mapping photo landmarks onto DAZ world space.

    Calibrates purely from shoulder width so results roughly track the target
    figure's actual size without manual tuning.  Axis signs are a fixed
    approximation (MediaPipe world landmarks: x right, y down, z toward
    camera → DAZ: x right, y up, z away from camera) — good enough for a
    frontal photo, but not guaranteed for every camera angle.  Use --scale to
    compensate if limbs look over- or under-extended.

    Returns:
        (unit_scale, mp_hip_mid, daz_hip_world) — unit_scale converts a meter
        offset in MediaPipe world-landmark space to DAZ scene units.
    """
    mp_shoulder_width = _dist(landmarks[L_SHOULDER], landmarks[R_SHOULDER])
    if mp_shoulder_width < 1e-6:
        sys.exit("Degenerate pose detection (zero shoulder width) — try another photo.")
    lo, hi = _PLAUSIBLE_SHOULDER_WIDTH_M
    if not (lo <= mp_shoulder_width <= hi):
        print(
            f"Warning: MediaPipe's metric shoulder-width estimate ({mp_shoulder_width:.3f} "
            f"'meters') is outside the plausible human range ({lo}-{hi}m). Its "
            "pose_world_landmarks depth/scale estimate is likely unreliable for this photo "
            "(partial body, extreme crop/angle, or low confidence). The applied pose may be "
            "way over- or under-scaled — tune --scale manually, or try a cleaner full-body "
            "frontal photo.",
            file=sys.stderr,
        )
    mp_hip_mid = (
        (landmarks[L_HIP][0] + landmarks[R_HIP][0]) / 2.0,
        (landmarks[L_HIP][1] + landmarks[R_HIP][1]) / 2.0,
        (landmarks[L_HIP][2] + landmarks[R_HIP][2]) / 2.0,
    )
    unit_scale = (daz_shoulder_width / mp_shoulder_width) * scale
    return unit_scale, mp_hip_mid, daz_hip_world


def to_daz_world(
    landmark: tuple[float, float, float],
    mp_hip_mid: tuple[float, float, float],
    daz_hip_world: tuple[float, float, float],
    unit_scale: float,
) -> tuple[float, float, float]:
    """Map one MediaPipe world landmark to a DAZ Studio world-space point."""
    offset = _sub(landmark, mp_hip_mid)
    # MediaPipe world landmarks: +x toward subject's right→image left, +y down, +z toward camera.
    # DAZ Studio: +x right, +y up, +z toward viewer. Flip y and z; x sign matches directly
    # because both conventions place the subject's anatomical right on their own +x.
    dx = offset[0] * unit_scale
    dy = -offset[1] * unit_scale
    dz = -offset[2] * unit_scale
    return (daz_hip_world[0] + dx, daz_hip_world[1] + dy, daz_hip_world[2] + dz)


# ── busy-retry helper ──────────────────────────────────────────────────────────
# A burst of rapid HTTP calls (one stacked-Jacobian solve per iteration) can
# catch DAZ Studio's main thread still finishing the previous call. Retry here
# instead of crashing the whole pose application over one transient hiccup.

def _call_with_busy_retry(fn, *, retries: int = 5, base_delay: float = 1.5):
    for attempt in range(retries):
        try:
            return fn()
        except DazBusyError:
            if attempt == retries - 1:
                raise
            wait = min(base_delay * (attempt + 1), 8.0)
            print(f"    (DAZ Studio busy, retrying in {wait:.1f}s...)")
            time.sleep(wait)


# ── stacked-Jacobian multi-effector IK ──────────────────────────────────────────
# Genesis 9 IK chains per limb (root -> effector's parent). These match what
# dazpy's own single-limb aligner resolves via FigureRigProfile.suggest_primary_chain()
# (see character/ik_bone_to_target.py --debug output) — hardcoded here (same
# precedent as bvh/bvh_bone_maps.py's per-generation bone tables) so the chains
# for all four limbs are known up front and can be merged into one combined
# perturbation script below.

# spine4 and hip are deliberately excluded from these chains (unlike dazpy's
# own hand_to_target()/foot_to_target(), whose auto-selected chains include
# them) — both are shared roots whose rotation reaches far beyond the limb
# being solved:
#   - spine4 is the hand chains' only source of extra reach beyond the
#     forearm, and the solver used that freely: a perfectly converged wrist
#     position with the whole spine wrenched back to get there.
#   - hip is worse: it's the skeleton's actual root (parent=None; even
#     spine1 is its child), so rotating it to help a foot reach farther
#     rocks the *entire* body — torso, arms, head included. pelvis, by
#     contrast, is a *sibling* of spine1 (both children of hip) rather than
#     its ancestor, so a leg chain rooted at pelvis can reach for a foot
#     target without moving the spine/head at all.
# Dropping both means a target beyond the remaining chain's reach honestly
# reports NOT CONVERGED instead of quietly recruiting the torso to compensate
# — matching the documented "torso/head aren't targeted" scope for real,
# instead of mostly matching it.
_CHAINS: dict[str, list[str]] = {
    "l_hand": ["l_shoulder", "l_upperarm", "l_forearm"],
    "r_hand": ["r_shoulder", "r_upperarm", "r_forearm"],
    "l_foot": ["pelvis", "l_thigh", "l_shin"],
    "r_foot": ["pelvis", "r_thigh", "r_shin"],
}


def _stacked_jacobian_script(
    figure_label: str, union_chain: list[str], effector_names: list[str], step_degrees: float
) -> str:
    """Build one DazScript call that perturbs every bone in `union_chain` (one
    axis at a time) and reads back all `effector_names` world positions after
    each perturbation — the server-side half of a stacked multi-effector
    Jacobian. Mirrors dazpy's own DazSkeleton.evaluate_pose_jacobian()
    (dazpy/_skeleton.py), generalized from one effector to several so the
    Python side can see how a bone shared across limb chains (e.g. `pelvis`,
    shared by both leg chains) affects *every* limb it touches, not just the
    one limb it happens to be perturbing for.
    """
    return f"""(function(){{
        var _skel=null,_skels=Scene.getSkeletonList();
        for(var _i=0;_i<_skels.length;_i++){{if(_skels[_i].getLabel()==={json.dumps(figure_label)}){{_skel=_skels[_i];break;}}}}
        if(!_skel) return null;
        var _chainNames={json.dumps(union_chain)};
        var _effNames={json.dumps(effector_names)};
        var _step={float(step_degrees)};
        var _all=_skel.getAllBones();
        var _map={{}};
        for(var i=0;i<_all.length;i++){{_map[_all[i].getName()]=_all[i];}}
        var _bones=[]; for(var i=0;i<_chainNames.length;i++){{
            var b=_map[_chainNames[i]]; if(!b) return {{error: "bone_not_found", name: _chainNames[i]}};
            _bones.push(b);
        }}
        var _effs=[]; for(var i=0;i<_effNames.length;i++){{
            var e=_map[_effNames[i]]; if(!e) return {{error: "bone_not_found", name: _effNames[i]}};
            _effs.push(e);
        }}
        function _pos(b){{var p=b.getWSPos(); return [p.x,p.y,p.z];}}
        var _base=[]; for(var e=0;e<_effs.length;e++){{_base.push(_pos(_effs[e]));}}
        var _columns=[];
        for(var bi=0; bi<_bones.length; bi++){{
            var b=_bones[bi];
            var ctrls=[b.getXRotControl(),b.getYRotControl(),b.getZRotControl()];
            for(var axis=0; axis<3; axis++){{
                var ctrl=ctrls[axis];
                var orig=ctrl.getValue();
                ctrl.setValue(orig+_step);
                var col=[];
                for(var e=0;e<_effs.length;e++){{
                    var p=_pos(_effs[e]);
                    col.push((p[0]-_base[e][0])/_step,(p[1]-_base[e][1])/_step,(p[2]-_base[e][2])/_step);
                }}
                ctrl.setValue(orig);
                _columns.push(col);
            }}
        }}
        return {{base: _base, columns: _columns}};
    }})()"""


def solve_stacked_ik(
    client: DazClient,
    figure,
    figure_label: str,
    effector_names: list[str],
    target_points: list[tuple[float, float, float]],
    *,
    max_iterations: int = 25,
    tolerance: float = 0.15,
    step_degrees: float = 1.0,
    damping: float = 0.25,
    rest_pose_weight: float = 0.15,
    debug: bool = False,
) -> dict[str, float]:
    """Solve all `effector_names` toward `target_points` simultaneously.

    One combined stacked-Jacobian call per iteration (see
    `_stacked_jacobian_script`) replaces what would otherwise be one
    `hand_to_target`/`foot_to_target` call per limb. `pelvis` is shared by
    both leg chains (see `_CHAINS`); because its effect on *both* feet is
    visible in the same Jacobian, a single damped-least-squares step per
    iteration accounts for that coupling directly — solving r_foot can no
    longer silently undo l_foot's progress, because both are solved for in
    the same step.

    With only 4 effectors (12 constraints) and up to 9 bones (27 rotation
    DOFs, with `_CHAINS` as currently defined), the system is still
    underdetermined: more than one joint configuration can hit the targets
    exactly, and plain DLS has no reason to prefer one zero-error solution
    over another. `rest_pose_weight` adds a secondary objective — pull
    unconstrained DOFs back toward the figure's *starting* rotations —
    projected through the Jacobian's null space so it never fights the
    primary hand/foot targets (same `rest_pose_weight` concept as the
    still-unused `dazpy.SolveOptions.rest_pose_weight` field, applied here
    for real). 0 disables it.

    Returns ``{effector_name: final_error}`` (scene units) for reporting.
    """
    union_chain: list[str] = []
    for name in effector_names:
        for bone in _CHAINS[name]:
            if bone not in union_chain:
                union_chain.append(bone)

    targets = np.array(target_points, dtype=float)  # (E, 3)
    n_dof = len(union_chain) * 3
    n_err = len(effector_names) * 3

    # Start from the figure's actual current rotations (not zero) so any
    # rotation already on these bones is preserved and refined, not clobbered.
    # Also doubles as the rest-pose bias target below.
    current_rot = figure.bone_rotations()
    current = {
        name: list(current_rot.get(name, (0.0, 0.0, 0.0)))
        for name in union_chain
    }
    rest_flat = np.array(
        [current[name][axis] for name in union_chain for axis in range(3)]
    )

    final_error = {name: float("inf") for name in effector_names}

    for iteration in range(max_iterations):
        script = _stacked_jacobian_script(figure_label, union_chain, effector_names, step_degrees)
        result = _call_with_busy_retry(lambda: client.execute(script).value)
        if result is None or result.get("error"):
            sys.exit(
                f"Error: stacked IK solve failed on {figure_label!r} "
                f"({result.get('name') if result else 'skeleton not found'})."
            )

        base = np.array(result["base"], dtype=float)  # (E, 3)
        error_vec = targets - base                    # (E, 3)
        for i, name in enumerate(effector_names):
            final_error[name] = float(np.linalg.norm(error_vec[i]))

        if debug:
            errs = "  ".join(f"{n}={final_error[n]:.3f}" for n in effector_names)
            print(f"    iter {iteration + 1:2d}/{max_iterations}: {errs}")

        if all(e <= tolerance for e in final_error.values()):
            break

        flat_error = error_vec.flatten()               # (E*3,)
        columns = np.array(result["columns"], dtype=float)  # (n_dof, E*3)
        J = columns.T                                   # (E*3, n_dof)

        # Damped least squares in the row space (n_dof > E*3, i.e. more DOFs
        # than target coordinates — the usual underdetermined-IK case): solve
        # (J J^T + damping*I) y = error, then delta = J^T y. Same formula as
        # dazpy's own single-chain solver (_interaction.py's
        # _damped_least_squares_step), generalized from a 3-row error vector
        # to E*3 rows so all four effectors are solved in one linear system.
        JJt = J @ J.T + damping * np.eye(n_err)
        try:
            JJt_inv = np.linalg.inv(JJt)
        except np.linalg.LinAlgError:
            break
        J_pinv = J.T @ JJt_inv                            # (n_dof, E*3)
        delta = J_pinv @ flat_error                        # (n_dof,) primary task

        if rest_pose_weight > 0:
            # Null-space secondary task: pull toward the starting pose without
            # touching the DOF combinations the primary task above already
            # claimed — (I - J_pinv @ J) projects a vector onto exactly the
            # subspace the targets don't constrain.
            current_flat = np.array(
                [current[name][axis] for name in union_chain for axis in range(3)]
            )
            null_space = np.eye(n_dof) - J_pinv @ J
            rest_pull = rest_pose_weight * (rest_flat - current_flat)
            delta = delta + null_space @ rest_pull

        max_delta = float(np.abs(delta).max()) if n_dof else 0.0
        if max_delta > step_degrees:
            delta *= step_degrees / max_delta

        for bone_idx, bone_name in enumerate(union_chain):
            for axis in range(3):
                current[bone_name][axis] += float(delta[bone_idx * 3 + axis])

        figure.set_bone_rotations({name: tuple(vals) for name, vals in current.items()})

    return final_error


def solve_pinocchio_ik(
    figure,
    figure_label: str,
    effector_names: list[str],
    target_points: list[tuple[float, float, float]],
    *,
    max_iterations: int = 300,
    tolerance: float = 0.15,
    damping: float = 0.2,
    rest_pose_weight: float = 0.15,
    max_step_degrees: float = 2.0,
    debug: bool = False,
) -> dict[str, float]:
    """Same contract as `solve_stacked_ik`, but the entire iterative solve runs
    locally against a Pinocchio kinematic model (see `pinocchio_ik.py`) --
    zero HTTP round-trips inside the loop, one `set_bone_rotations()` call at
    the end. Requires a real Pinocchio install (see `pinocchio_ik.py`'s module
    docstring for the Windows conda-forge setup; there is no usable PyPI wheel).
    """
    import pinocchio_ik as pik

    union_chain: list[str] = []
    for name in effector_names:
        for bone in _CHAINS[name]:
            if bone not in union_chain:
                union_chain.append(bone)
        if name not in union_chain:
            union_chain.append(name)  # effector itself is a pure position marker in the model

    meta = figure.bone_metadata()
    fm = pik.get_figure_model(figure_label, meta, union_chain)

    current_rot = figure.bone_rotations()
    initial_angles = {
        name: {"x": current_rot[name][0], "y": current_rot[name][1], "z": current_rot[name][2]}
        for name in union_chain
    }

    solved_angles, final_error = pik.solve_ik(
        fm, effector_names, np.array(target_points, dtype=float),
        initial_angles=initial_angles,
        max_iterations=max_iterations, tolerance=tolerance, damping=damping,
        rest_pose_weight=rest_pose_weight, max_step_degrees=max_step_degrees,
        debug=debug,
    )

    figure.set_bone_rotations({name: (a["x"], a["y"], a["z"]) for name, a in solved_angles.items()})
    return final_error


def process_photo(image_path: str, figure_label: str, scene: "DazScene", client: DazClient, args) -> dict:
    """Run the full single-photo pose-transfer pipeline against the given figure.

    Shared by both the single-image CLI path and `--batch` looping below --
    identical to what the script always did for one photo, just extracted so
    a batch run can call it once per image without duplicating this logic.
    Returns a result dict with per-effector and (if `--fingers`) per-digit
    convergence errors, for reporting/CSV output.
    """
    if args.fingers:
        import finger_ik as fik
        from hand_landmarks import extract_hand_world_landmarks

    try:
        figure = scene.find_skeleton_by_label(figure_label)
    except Exception:
        sys.exit(f"Error: figure {figure_label!r} not found in scene.")

    print(f"Extracting pose from {image_path!r}...")
    landmarks = extract_world_landmarks(image_path)

    # Calibrate against the figure's own rest-pose shoulder width and hip position,
    # read once before any IK is applied.
    metadata = figure.bone_metadata()
    by_name = {b["name"]: b for b in metadata}

    def _world(name: str) -> tuple[float, float, float]:
        w = by_name[name]["world_position"]
        return (w["x"], w["y"], w["z"])

    try:
        # l_upperarm/r_upperarm are the actual shoulder ball joints (~35 units apart on
        # Genesis 9). l_shoulder/r_shoulder are the clavicle roots, which sit close to the
        # spine (~8 units apart) and badly under-measure shoulder width if used here.
        daz_shoulder_width = _dist(_world("l_upperarm"), _world("r_upperarm"))
        daz_hip_world = _world("hip")
    except KeyError as exc:
        sys.exit(
            f"Bone {exc} not found on {figure_label!r} — this example targets Genesis 9 "
            "bone names (l_upperarm, r_upperarm, hip). Run figure.bones() to list this "
            "figure's actual bone names if it's a different generation."
        )

    unit_scale, mp_hip_mid, daz_hip_world = calibrate(
        landmarks, daz_shoulder_width, daz_hip_world, args.scale
    )
    print(f"Calibrated scale: {unit_scale:.4f} (scene units per photo unit)")

    effector_names: list[str] = []
    target_points: list[tuple[float, float, float]] = []
    if args.hands:
        effector_names += ["l_hand", "r_hand"]
        target_points += [
            to_daz_world(landmarks[L_WRIST], mp_hip_mid, daz_hip_world, unit_scale),
            to_daz_world(landmarks[R_WRIST], mp_hip_mid, daz_hip_world, unit_scale),
        ]
    if args.feet:
        effector_names += ["l_foot", "r_foot"]
        target_points += [
            to_daz_world(landmarks[L_ANKLE], mp_hip_mid, daz_hip_world, unit_scale),
            to_daz_world(landmarks[R_ANKLE], mp_hip_mid, daz_hip_world, unit_scale),
        ]

    if not effector_names:
        sys.exit("Nothing to do — both --no-hands and --no-feet were given.")

    for name, point in zip(effector_names, target_points):
        print(f"  {name:10s} target -> {point[0]:+7.2f}, {point[1]:+7.2f}, {point[2]:+7.2f}")

    print(f"\nSolving {len(effector_names)} limb targets on {figure_label!r} simultaneously "
          f"(backend={args.backend!r})...")
    with scene.undo("Apply photo pose"):
        if args.backend == "pinocchio":
            final_error = solve_pinocchio_ik(
                figure, figure_label, effector_names, target_points,
                max_iterations=args.max_iterations, tolerance=args.tolerance,
                damping=args.damping, rest_pose_weight=args.rest_pose_weight,
                max_step_degrees=args.step_degrees, debug=args.debug,
            )
        else:
            final_error = solve_stacked_ik(
                client, figure, figure_label, effector_names, target_points,
                max_iterations=args.max_iterations, tolerance=args.tolerance,
                step_degrees=args.step_degrees, damping=args.damping,
                rest_pose_weight=args.rest_pose_weight, debug=args.debug,
            )

    print("\nFinal per-limb error (scene units):")
    for name in effector_names:
        status = "OK" if final_error[name] <= args.tolerance else "NOT CONVERGED"
        print(f"  {name:10s} {final_error[name]:.3f}   [{status}]")

    finger_errors: dict[str, dict[str, float]] = {}
    if args.fingers:
        print("\nExtracting hand landmarks for finger posing...")
        detected_hands = extract_hand_world_landmarks(image_path)

        # Re-read bone_metadata() *after* the body IK solve above, but only
        # to learn each wrist's POSED world position (for anchoring landmarks
        # and as the "from" side of the frame-correction below) -- every
        # digit's own IK model is still built from the REST snapshot
        # (`by_name`, captured before the body solve), not this one. Building
        # from REST metadata is what keeps each digit's geometry (and its
        # universe-rooted parent assumption) correct regardless of the
        # wrist's post-solve rotation; see `finger_ik.wrist_world_rotation`'s
        # docstring for why post-solve metadata can't be used here.
        post_solve_meta = {b["name"]: b for b in figure.bone_metadata()}

        def _post_world(name: str) -> tuple[float, float, float]:
            w = post_solve_meta[name]["world_position"]
            return (w["x"], w["y"], w["z"])

        # Hoisted once: every digit below reads the same body-solved
        # rotations (fingers themselves haven't been touched yet at this
        # point, so this doesn't change across digits or hands).
        current_rot = figure.bone_rotations()

        all_solved_angles: dict[str, tuple[float, float, float]] = {}

        # MediaPipe HandLandmarker's handedness label assumes a mirrored
        # (selfie-style) input image (see hand_landmarks.py's docstring for
        # the verified citation); pose_transfer_photo.py loads photos
        # directly via cv2.imread with no mirroring, so the label is
        # anatomically reversed here -- "Left" is the subject's own right
        # hand, and vice versa.
        side_by_label = {"Left": "r", "Right": "l"}
        for hand_label, side in side_by_label.items():
            if hand_label not in detected_hands:
                print(f"  {hand_label} hand: SKIPPED (not detected)")
                continue

            wrist_bone = f"{side}_hand"
            wrist_rest_world = _world(wrist_bone)
            wrist_posed_world = _post_world(wrist_bone)
            anchored = fik.anchor_hand_landmarks(
                detected_hands[hand_label], mp_hip_mid, daz_hip_world, unit_scale, wrist_posed_world,
                wrist_rest_world, _world(f"{side}_index1"), _world(f"{side}_pinky1"),
            )

            # W_hand: the wrist's accumulated world ROTATION after the body
            # solve (bone_metadata() has no orientation field). Built from
            # REST metadata + the body-solved angles for this arm's chain, so
            # it reflects exactly what the body solve did to this wrist,
            # without re-deriving anything pinocchio_ik.py already computed.
            wrist_rotation = fik.wrist_world_rotation(
                list(by_name.values()), _CHAINS[wrist_bone], wrist_bone, current_rot,
            )

            for digit_name in ("index", "mid", "ring", "pinky", "thumb"):
                digit = fik.DIGIT_CHAINS[f"{side}_{digit_name}"]
                # Built from REST metadata (not post_solve_meta): each digit's
                # root bone (e.g. l_indexmetacarpal) has its real parent
                # (l_hand) outside this chain's bone set, so
                # build_figure_model roots it directly to the universe frame
                # with IDENTITY rotation. That's only exact if the excluded
                # parent's own world rotation is identity -- true at REST,
                # false once the body solve has rotated the wrist. Rather
                # than feed this model posed positions (which would still be
                # wrong, since the identity-rooting assumption is about
                # rotation, not position), the model stays in the wrist's
                # REST frame and the targets are the ones transformed to
                # match (below).
                fm = fik.pik.build_figure_model(list(by_name.values()), digit.chain_bones)
                initial_angles = {
                    b: {"x": current_rot[b][0], "y": current_rot[b][1], "z": current_rot[b][2]}
                    for b in digit.chain_bones
                }
                target1 = fik.rotate_target_into_rest_wrist_frame(
                    anchored[digit.target1_landmark], wrist_rest_world, wrist_posed_world, wrist_rotation,
                )
                target2 = fik.rotate_target_into_rest_wrist_frame(
                    anchored[digit.target2_landmark], wrist_rest_world, wrist_posed_world, wrist_rotation,
                )
                tip = fik.rotate_target_into_rest_wrist_frame(
                    anchored[digit.tip_landmark], wrist_rest_world, wrist_posed_world, wrist_rotation,
                )
                solved, err = fik.solve_digit_chain(
                    fm, digit,
                    target1_point=target1, target2_point=target2, tip_point=tip,
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

    summary = f"\nDone. Applied {len(effector_names)} limb targets to {figure_label!r}."
    if args.fingers:
        summary += f" Posed fingers for {len(finger_errors)} digit(s)."
    print(summary)

    return {
        "image": image_path,
        "figure": figure_label,
        "figure_obj": figure,
        "limb_error": final_error,
        "finger_error": finger_errors,
    }


# ── batch output actions ────────────────────────────────────────────────────
# Each of --save-poses / --render / --export-mesh / --stats-csv is independent
# and combinable -- pass any subset. All write into subdirectories of
# --output-dir named after the source photo's filename stem, so results from
# different photos (and different output kinds) never collide.

def _stem(image_path: str) -> str:
    return os.path.splitext(os.path.basename(image_path))[0]


def save_pose_preset(figure, image_path: str, output_dir: str) -> str:
    """Save the figure's current pose-channel angles as a small JSON file.

    Not a native DAZ Studio .duf pose preset -- writing one of those goes
    through content-library save dialogs that can pop a blocking modal (see
    this project's own crash notes on live DazScript dialogs), which isn't
    safe to drive unattended across a batch of photos. This captures the
    same information (every posed bone's X/Y/Z rotation) in a format any
    later step can re-apply via `figure.set_bone_rotations()` without going
    near DAZ Studio's UI.
    """
    poses_dir = os.path.join(output_dir, "poses")
    os.makedirs(poses_dir, exist_ok=True)
    path = os.path.join(poses_dir, f"{_stem(image_path)}.json")
    rotations = _call_with_busy_retry(figure.bone_rotations)
    with open(path, "w") as f:
        json.dump({name: list(xyz) for name, xyz in rotations.items()}, f, indent=2)
    return path


def render_photo(
    client: DazClient, image_path: str, output_dir: str, camera_labels: list[str] | None = None
) -> list[str]:
    """Render the current pose to disk, once per camera in *camera_labels*.

    With no *camera_labels*, renders once from the active viewport camera to
    <output-dir>/renders/<stem>.png (unchanged from before --camera existed).
    With one or more labels, renders once per camera to
    <output-dir>/renders/<camera-label>/<stem>.png so results from different
    cameras never collide. A render failure for one camera is a per-camera
    warning, not a batch-aborting error -- same resilience as a bad photo.
    """
    renders_dir = os.path.join(output_dir, "renders")
    settings = DazRenderSettings(client)
    paths = []

    for camera_label in (camera_labels or [None]):
        out_dir = os.path.join(renders_dir, camera_label) if camera_label else renders_dir
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"{_stem(image_path)}.png")

        def _do_render(_path=path, _camera_label=camera_label):
            settings.output_path = _path
            return settings.render(camera_label=_camera_label)

        outcome = _call_with_busy_retry(_do_render)
        if not outcome.success:
            camera_note = f" (camera {camera_label!r})" if camera_label else ""
            print(f"  WARNING: render failed for {image_path!r}{camera_note}")
        paths.append(outcome.output_path or path)

    return paths


def simulate_dforce(scene: "DazScene", memorized_pose: bool) -> None:
    """Run a dForce simulation against the current scene state.

    Blocking (`wait=True`) since the batch loop is inherently sequential per
    photo anyway -- there's no other work to overlap a sim with.
    """
    scene.run_dforce_simulation(memorized_pose=memorized_pose, wait=True)


def export_mesh(scene: "DazScene", image_path: str, output_dir: str) -> str:
    meshes_dir = os.path.join(output_dir, "meshes")
    os.makedirs(meshes_dir, exist_ok=True)
    path = os.path.join(meshes_dir, f"{_stem(image_path)}.obj")
    _call_with_busy_retry(lambda: scene.export_obj(path, selected_only=False))
    return path


_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")


def find_batch_images(batch_dir: str) -> list[str]:
    names = sorted(
        name for name in os.listdir(batch_dir)
        if name.lower().endswith(_IMAGE_EXTENSIONS)
    )
    return [os.path.join(batch_dir, name) for name in names]


def write_stats_csv(path: str, results: list[dict], failures: list[dict] | None = None) -> None:
    """Write one row per photo -- successes with per-limb error, failures (a
    photo `process_photo()` couldn't process at all, e.g. no pose detected)
    with empty limb columns and their failure reason in `note`, so a batch
    with some unprocessable photos still produces one complete, readable
    record of what happened to every photo instead of silently omitting
    the failed ones.
    """
    import csv
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    limb_names = sorted({name for r in results for name in r["limb_error"]})
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "figure"] + limb_names + ["converged", "note"])
        for r in results:
            row = [r["image"], r["figure"]]
            all_ok = True
            for name in limb_names:
                err = r["limb_error"].get(name)
                row.append(f"{err:.4f}" if err is not None else "")
                if err is not None and err > r.get("tolerance", 0.15):
                    all_ok = False
            row.append("yes" if all_ok else "no")
            row.append("")
            writer.writerow(row)
        for f_ in (failures or []):
            row = [f_["image"], f_["figure"]] + [""] * len(limb_names) + ["error", f_["reason"]]
            writer.writerow(row)


if __name__ == "__main__":
    # ── CLI ────────────────────────────────────────────────────────────────────────

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("image", nargs="?", default=None,
                        help="Path to source image. Omit when using --batch.")
    parser.add_argument("--batch", metavar="DIR", default=None,
                        help="Process every image (.png/.jpg/.jpeg/.webp) in DIR against "
                             "--figure, one after another, instead of a single --image. "
                             "The figure is reset to Zero Pose (dazpy.poses.zero_figure) "
                             "before each photo, same as manual between-run guidance. "
                             "Pair with --backend pinocchio to actually get the per-figure "
                             "model-caching payoff (see pinocchio_ik.get_figure_model) -- "
                             "the default 'stacked' backend re-pays its HTTP round-trip cost "
                             "on every photo regardless of batching.")
    parser.add_argument("--output-dir", metavar="DIR", default="batch_output",
                        help="Root directory for --save-poses/--render/--export-mesh/"
                             "--stats-csv output (default: 'batch_output'). Each kind writes "
                             "into its own subdirectory, one file per photo named after the "
                             "photo's filename stem.")
    parser.add_argument("--save-poses", action="store_true",
                        help="Save each photo's solved pose-channel angles as a JSON file "
                             "under <output-dir>/poses/. Combinable with any other --save-*/"
                             "--render/--export-mesh/--stats-csv flag.")
    parser.add_argument("--render", action="store_true",
                        help="Render each photo's posed figure to <output-dir>/renders/ "
                             "(from the active viewport camera). Combinable with the other "
                             "output flags. This is a real DAZ Studio render per photo -- "
                             "expect it to dominate wall-clock time in a large batch.")
    parser.add_argument("--camera", metavar="LABEL", action="append", default=None,
                        help="Render from this named camera (the label shown in the Scene "
                             "panel) instead of the active viewport camera. Repeatable -- "
                             "pass multiple times to render each photo from each camera. "
                             "Writes to <output-dir>/renders/<camera-label>/. Implies "
                             "--render; every --camera label is validated against the scene "
                             "before the batch starts.")
    parser.add_argument("--dforce", action="store_true",
                        help="Run a dForce simulation (clothing/hair) against the posed figure "
                             "before rendering/exporting, using the figure's live pose as the "
                             "sim's starting point. dForce sims can take minutes per photo -- "
                             "expect this to dominate wall-clock time in a large batch, more so "
                             "than --render. Implied by --dforce-memorize.")
    parser.add_argument("--dforce-memorize", action="store_true",
                        help="Like --dforce, but starts the simulation's pose-based bulge "
                             "corrections from DAZ Studio's memorized pose instead of the "
                             "figure's live pose (the 'Start Bulge from Memorized Pose' option "
                             "in the Simulation Settings pane). Implies --dforce.")
    parser.add_argument("--export-mesh", action="store_true",
                        help="Export each photo's posed figure to <output-dir>/meshes/ as "
                             "OBJ. Combinable with the other output flags.")
    parser.add_argument("--stats-csv", action="store_true",
                        help="Write one row per photo (per-limb convergence error + overall "
                             "converged yes/no) to <output-dir>/stats.csv. Combinable with "
                             "the other output flags; per-photo convergence is always also "
                             "printed to stdout regardless of this flag.")
    parser.add_argument("--figure", default="Jason Cross",
                        help="DAZ figure label (default: 'Jason Cross')")
    parser.add_argument("--backend", choices=["stacked", "pinocchio"], default="stacked",
                        help="'stacked' (default): original HTTP finite-difference solver, "
                             "one round-trip per iteration. 'pinocchio': local kinematic model "
                             "(see pinocchio_ik.py), zero round-trips during iteration, one "
                             "set_bone_rotations() push at the end -- requires the separate "
                             "'pinocchio-ik' conda-forge environment (see environment.yml); "
                             "run via run_pinocchio.ps1/.sh rather than this interpreter "
                             "directly, or you'll hit an import error with setup instructions.")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="Extra multiplier on top of auto-calibrated scale (default: 1.0)")
    parser.add_argument("--no-hands", dest="hands", action="store_false",
                        help="Skip wrist targets")
    parser.add_argument("--no-feet", dest="feet", action="store_false",
                        help="Skip ankle targets")
    parser.add_argument("--fingers", action="store_true",
                        help="Also pose fingers from the same photo via MediaPipe HandLandmarker "
                             "(requires --backend pinocchio and wrist targets enabled)")
    parser.add_argument("--max-iterations", type=int, default=150,
                        help="Max stacked-IK iterations for the whole simultaneous solve "
                             "(default: 150 — this is a real numerical solve, not a quick "
                             "lookup; large displacements from rest can need most of that)")
    parser.add_argument("--tolerance", type=float, default=0.15,
                        help="IK convergence distance in scene units, per effector (default: 0.15)")
    parser.add_argument("--step-degrees", type=float, default=1.0,
                        help="Max per-bone rotation change per iteration, in degrees (default: 1.0)")
    parser.add_argument("--damping", type=float, default=0.1,
                        help="Damped-least-squares damping factor (default: 0.1)")
    parser.add_argument("--rest-pose-weight", type=float, default=0.15,
                        help="Null-space bias pulling unconstrained joints back toward the "
                             "figure's starting pose, 0-1 (default: 0.15). Raise if limbs "
                             "look contorted despite low convergence error; 0 disables it.")
    parser.add_argument("--debug", action="store_true",
                        help="Print per-iteration convergence diagnostics")
    args = parser.parse_args()

    if bool(args.batch) == bool(args.image):
        sys.exit("Pass exactly one of a single image path or --batch DIR.")

    if args.fingers and args.backend != "pinocchio":
        sys.exit("--fingers requires --backend pinocchio (per-finger IK is too slow over the "
                  "HTTP stacked backend -- see finger_ik.py's module docstring).")
    if args.fingers and not args.hands:
        sys.exit("--fingers requires wrist targets to be enabled (fingers are anchored to the "
                  "solved wrist position) -- remove --no-hands.")

    if args.camera:
        args.render = True
    if args.dforce_memorize:
        args.dforce = True

    if args.batch:
        image_paths = find_batch_images(args.batch)
        if not image_paths:
            sys.exit(f"No {'/'.join(_IMAGE_EXTENSIONS)} images found in {args.batch!r}.")
        print(f"Batch: {len(image_paths)} photo(s) from {args.batch!r} -> {args.figure!r}")
    else:
        image_paths = [args.image]

    scene = DazScene()
    client = DazClient()

    if args.camera:
        known_labels = {cam.label for cam in scene.cameras()}
        unknown = [label for label in args.camera if label not in known_labels]
        if unknown:
            sys.exit(f"--camera label(s) not found in scene: {', '.join(map(repr, unknown))}")

    results: list[dict] = []
    failures: list[dict] = []
    for i, image_path in enumerate(image_paths, start=1):
        if args.batch:
            print(f"\n[{i}/{len(image_paths)}] {image_path}")
            figure = _call_with_busy_retry(lambda: scene.find_skeleton_by_label(args.figure))
            _call_with_busy_retry(lambda: zero_figure(figure))

        if args.batch:
            # process_photo() (via extract_world_landmarks()/calibrate()/etc.)
            # uses sys.exit() for per-photo problems like "no pose detected"
            # or "degenerate detection" -- correct for a single-image run
            # (a real error, exit the process), wrong for a batch: one
            # unprocessable photo (extreme crop/angle, occluded figure,
            # stylized art the pose model can't read) shouldn't abort every
            # photo after it. Catch it here, record why, and move on.
            try:
                result = process_photo(image_path, args.figure, scene, client, args)
            except SystemExit as exc:
                reason = str(exc.code) if exc.code is not None else "unknown error"
                print(f"  SKIPPED: {reason}")
                failures.append({"image": image_path, "figure": args.figure, "reason": reason})
                continue
        else:
            result = process_photo(image_path, args.figure, scene, client, args)

        result["tolerance"] = args.tolerance
        results.append(result)

        if args.save_poses:
            path = save_pose_preset(result["figure_obj"], image_path, args.output_dir)
            print(f"  saved pose -> {path}")
        if args.dforce:
            print(f"  simulating dForce ({'memorized' if args.dforce_memorize else 'live'} pose)...")
            _call_with_busy_retry(lambda: simulate_dforce(scene, args.dforce_memorize))
        if args.render:
            paths = render_photo(client, image_path, args.output_dir, args.camera)
            for path in paths:
                print(f"  rendered -> {path}")
        if args.export_mesh:
            path = export_mesh(scene, image_path, args.output_dir)
            print(f"  exported mesh -> {path}")

    if args.stats_csv:
        csv_path = os.path.join(args.output_dir, "stats.csv")
        write_stats_csv(csv_path, results, failures)
        print(f"\nStats written -> {csv_path}")

    if args.batch:
        converged = sum(
            1 for r in results if all(e <= args.tolerance for e in r["limb_error"].values())
        )
        print(f"\nBatch done: {converged}/{len(results)} converged, "
              f"{len(results) - converged}/{len(results)} not converged, "
              f"{len(failures)} skipped (no pose detected/other error), "
              f"out of {len(image_paths)} total photo(s).")
