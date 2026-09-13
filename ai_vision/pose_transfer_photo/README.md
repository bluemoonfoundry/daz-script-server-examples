# Pose Transfer (Photo)

**Level:** Advanced
**Category:** AI/Vision

## Overview

Extracts a body pose from a photo using MediaPipe PoseLandmarker, then drives a Genesis 9 figure's hands and feet toward the photo-derived world-space positions using a custom stacked-Jacobian, multi-effector IK solve — all four limbs solved simultaneously in one damped-least-squares step per iteration, with a null-space bias that keeps unconstrained joints close to the figure's starting pose. Elbows and knees follow from the IK chain solve; torso/head are never touched, by construction (see How It Works).

## What You'll Learn

- Building a custom multi-effector IK solve on top of the script server's raw-DazScript escape hatch (see `fundamentals/raw_script/`), when the SDK's own single-effector `hand_to_target()`/`foot_to_target()` (used directly in `character/ik_bone_to_target/`) isn't enough
- Why chain composition matters for IK quality: a shared root bone (`spine4`, `hip`) reached by more than one limb's chain lets the solver "cheat" by moving that shared bone instead of the limb itself — sometimes producing a lower position error at the cost of an anatomically wrong pose
- Adding a null-space secondary objective (rest-pose bias) to a damped-least-squares IK solve using numpy
- Auto-calibrating a photo's scale/origin against a live figure's own rest-pose measurements
- Wrapping a multi-limb pose application in one undo step
- Mapping MediaPipe FaceLandmarker's ARKit-style blendshape output onto a different rig's own morph vocabulary (Genesis 9's facs_bs_* set splits several bilateral ARKit categories into separate Left/Right and/or Upper/Lower morphs) via a small static lookup table, rather than assuming the two naming schemes line up 1:1

## Prerequisites

- DAZ Studio with DazScriptServer plugin running
- A scene with a Genesis 9 figure loaded (bone names `l_upperarm`, `r_upperarm`, `hip`, `pelvis`, `l_shoulder`/`r_shoulder`, `l_upperarm`/`r_upperarm`, `l_forearm`/`r_forearm`, `l_thigh`/`r_thigh`, `l_shin`/`r_shin`, `l_hand`, `r_hand`, `l_foot`, `r_foot` — Genesis 9's internal naming), **zeroed to a neutral pose** (e.g. via `dazpy.poses.zero_figure(figure)`, or DAZ Studio's own Zero Pose) before each run — see Limitations
- `dazpy` installed (`pip install dazpy`)

## Dependencies

```bash
pip install mediapipe opencv-python numpy
```

The above covers the default `--backend stacked`. `--backend pinocchio`
(local IK solve, needed for `--batch` to actually pay off across many
photos — see below) needs a **separate environment**, because it depends on
the real Pinocchio C++/Eigen library, which has no Windows PyPI wheel and
is only reliably available via conda-forge:

```bash
conda env create -f environment.yml   # one-time; also installs mediapipe/opencv/dazpy
```

Then run through that environment via the wrapper scripts in this
directory, rather than invoking your normal interpreter directly (see
`pinocchio_ik.py`'s module docstring for the DLL/PATH pitfalls this avoids):

```bash
./run_pinocchio.ps1 photo.jpg --backend pinocchio     # Windows
./run_pinocchio.sh  photo.jpg --backend pinocchio     # macOS/Linux
```

If you forget and run `--backend pinocchio` from the wrong interpreter, the
import failure now prints these same setup steps instead of a bare
`ModuleNotFoundError`.

## Usage

```bash
python pose_transfer_photo.py photo.jpg
python pose_transfer_photo.py photo.jpg --figure "Jason Cross"
python pose_transfer_photo.py photo.jpg --scale 1.2
python pose_transfer_photo.py photo.jpg --no-feet
python pose_transfer_photo.py photo.jpg --debug

# Batch: process every photo in a folder against one figure (auto-resets to
# Zero Pose before each), via the pinocchio-ik environment (see Dependencies
# above). Any combination of the four output flags is allowed.
./run_pinocchio.ps1 --batch photos/ --backend pinocchio \
    --save-poses --render --export-mesh --stats-csv --output-dir batch_output/

    # --expression-image PATH: apply a facial expression extracted from a
    # separate photo (MediaPipe FaceLandmarker blendshapes -> Genesis 9
    # facs_bs_*/facs_ctrl_* morphs), holding body pose fixed. Repeatable --
    # each path produces its own output variant against the SAME solved body
    # pose. Combine with --batch for a full N-photos x M-expressions grid.
    python pose_transfer_photo.py photo.jpg \
        --expression-image smile.jpg --expression-image surprised.jpg \
        --save-poses --render --output-dir batch_output/
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `image` | *(required)* | Path to source image (JPEG, PNG, or any format OpenCV supports) |
| `--figure LABEL` | `Jason Cross` | Target figure label |
| `--scale FLOAT` | `1.0` | Extra multiplier on top of the auto-calibrated scale — tune if limbs look over/under-extended |
| `--no-hands` | off | Skip wrist targets |
| `--no-feet` | off | Skip ankle targets |
| `--max-iterations INT` | `150` | Max iterations for the whole simultaneous solve — this is a real numerical solve, not a lookup; large displacements from rest can need most of it |
| `--tolerance FLOAT` | `0.15` | Per-effector convergence distance, scene units |
| `--step-degrees FLOAT` | `1.0` | Max per-bone rotation change per iteration, in degrees |
| `--damping FLOAT` | `0.1` | Damped-least-squares damping factor — lower converges faster but can overshoot/oscillate |
| `--rest-pose-weight FLOAT` | `0.15` | Null-space bias toward the starting pose, 0-1 — raise if limbs look contorted despite low error, `0` disables it |
| `--debug` | off | Print per-iteration convergence diagnostics |

## How It Works

1. MediaPipe PoseLandmarker extracts 33 metric-scale, hip-centered `pose_world_landmarks` from the photo.
2. The figure's own rest-pose shoulder width (`l_upperarm`/`r_upperarm` distance) and hip world position are read once (`bone_metadata()`), and used to auto-calibrate a scale factor and origin — so the mapping roughly tracks the target figure's actual size.
3. Wrist and ankle landmarks are converted to DAZ world-space points (fixed axis-sign approximation: MediaPipe `+y` down / `+z` toward camera → DAZ `+y` up / `+z` toward viewer).
4. All four targets are solved **simultaneously**, each iteration:
   - One combined raw-DazScript call (`_stacked_jacobian_script`) perturbs every bone across all four limb chains — `l_shoulder`/`l_upperarm`/`l_forearm` for the left hand, `pelvis`/`l_thigh`/`l_shin` for the left leg, etc. — one axis at a time, and reads back **all four** effector world positions after each perturbation. This is a single HTTP round trip and gives a full stacked Jacobian with cross-limb coupling terms included (how does rotating `pelvis` for the left foot affect the right foot's position too?).
   - Python (numpy) solves one damped-least-squares step for the whole 12-constraint / ~30-39-DOF system at once, then projects a secondary "pull toward the starting pose" objective through the Jacobian's null space (`--rest-pose-weight`) so DOFs the primary task doesn't need stay close to rest instead of drifting into an arbitrary zero-error configuration.
   - The combined delta is applied to all affected bones in one `set_bone_rotations()` call.
5. All iterations are wrapped in one `scene.undo("Apply photo pose")` block, so `Ctrl+Z` undoes the entire pose in a single step.

### Why not just call `hand_to_target()`/`foot_to_target()` per limb?

An earlier version did exactly that, once per limb in sequence. Two problems surfaced under live testing (see PR history): `hand_to_target()`'s auto-selected chain includes `spine4`, shared by both hands, and `foot_to_target()`'s includes `hip`/`pelvis`, shared by both feet — solving one limb could nudge a shared bone a previously-solved limb depended on, undoing part of its progress. Adding "relaxation passes" (re-solving every limb 2-3 times) helped but never fully converged. Solving jointly removes the "previous limb" problem entirely — but a second issue emerged: `hip` is the skeleton's actual root (`spine1` is also its child), so letting a foot-reach solve rotate it tipped the *entire* body, head included, to gain a few extra units of leg reach. The fix was architectural, not just more solver iterations: this example's chains exclude both `spine4` and `hip` (see `_CHAINS` in the script), rooting the leg chains at `pelvis` instead — a true sibling of `spine1`, not its ancestor — so a leg-reach solve can no longer touch the spine at all.

## Output

Console output shows the calibrated scale, each target point, and (with `--debug`) per-iteration convergence for all four effectors at once. Final per-limb error and `OK`/`NOT CONVERGED` status print at the end.

## Limitations

- **Reset the figure to a neutral pose before each run.** Calibration reads the figure's *current* shoulder-joint and hip world positions — if a previous pose is still applied, the scale/origin will be calibrated against a contorted rig instead of a clean rest pose.
- **The photo must show one person, unobstructed, full body.** MediaPipe only returns one pose (`num_poses=1`) and will silently lock onto whichever candidate it's most confident in — feeding it a multi-person image (e.g. a reference mood board / contact sheet) gets you *a* pose, just not necessarily the one you meant. If results look physically implausible (e.g. ankles ending up near head height), check what MediaPipe actually detected before assuming the math is wrong.
- Only wrist and ankle effectors are targeted — elbow/knee bend comes from the IK solve, not directly from the photo, so unusual poses (e.g. a sharply bent knee, or a seated/kneeling pose where legs fold under the body) may not match exactly, since this example doesn't solve knee position directly.
- Torso and head are never touched (see How It Works) — a photo pose that genuinely requires torso lean or twist to reach (arms held out wide, weight shifted hard to one side) will show up as `NOT CONVERGED` on the affected limb rather than a bent spine.
- This is a real (if small) numerical optimization, not a lookup: convergence can still need the full default 150 iterations for large displacements from rest, and each iteration is one HTTP round trip. `--debug` shows per-iteration error if you want to watch it converge (or stall).
- Axis-sign mapping assumes a roughly frontal photo; side-on or unusual camera angles may need `--scale` tuning or produce mirrored-looking results.

## Related Examples

- [ik_bone_to_target](../../character/ik_bone_to_target/) — the single-effector IK aligner this example's chains and DLS math are modeled on, demonstrated directly
- [expression_transfer](../expression_transfer/) — the equivalent workflow for facial expressions
- [raw_script](../../fundamentals/raw_script/) — the "drop to raw DazScript" pattern this example's stacked-Jacobian call is built on
- See main repository [README](../../README.md) for related examples
