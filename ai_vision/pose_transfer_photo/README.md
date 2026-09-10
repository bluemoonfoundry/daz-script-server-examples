# Pose Transfer (Photo)

**Level:** Advanced
**Category:** AI/Vision

## Overview

Extracts a body pose from a photo using MediaPipe PoseLandmarker, then drives a Genesis 9 figure's hands and feet toward the photo-derived world-space positions using dazpy's existing IK limb aligner. Elbows and knees follow from the IK chain solve; torso and head are left at rest.

## What You'll Learn

- Driving `DazSkeleton.hand_to_target()` / `.foot_to_target()` (the same IK aligner used by `character/ik_bone_to_target/`) from external Python vision output
- Auto-calibrating a photo's scale/origin against a live figure's own rest-pose measurements
- Wrapping a multi-limb pose application in one undo step

## Prerequisites

- DAZ Studio with DazScriptServer plugin running
- A scene with a Genesis 9 figure loaded (bone names `l_shoulder`, `r_shoulder`, `hip`, `l_hand`, `r_hand`, `l_foot`, `r_foot` — Genesis 9's internal naming)
- `dazpy` installed (`pip install dazpy`)

## Dependencies

```bash
pip install mediapipe opencv-python numpy
```

## Usage

```bash
python pose_transfer_photo.py photo.jpg
python pose_transfer_photo.py photo.jpg --figure "Jason Cross"
python pose_transfer_photo.py photo.jpg --scale 1.2
python pose_transfer_photo.py photo.jpg --no-feet
python pose_transfer_photo.py photo.jpg --debug
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `image` | *(required)* | Path to source image (JPEG, PNG, or any format OpenCV supports) |
| `--figure LABEL` | `Jason Cross` | Target figure label |
| `--scale FLOAT` | `1.0` | Extra multiplier on top of the auto-calibrated scale — tune if limbs look over/under-extended |
| `--no-hands` | off | Skip wrist targets |
| `--no-feet` | off | Skip ankle targets |
| `--max-iterations INT` | `15` | Max IK iterations per limb |
| `--tolerance FLOAT` | `0.15` | IK convergence distance in scene units |
| `--debug` | off | Print per-limb IK chain and convergence diagnostics |

## How It Works

1. MediaPipe PoseLandmarker extracts 33 metric-scale, hip-centered `pose_world_landmarks` from the photo.
2. The figure's own rest-pose shoulder width and hip world position are read once (`bone_metadata()`), and used to auto-calibrate a scale factor and origin — so the mapping roughly tracks the target figure's actual size.
3. Wrist and ankle landmarks are converted to DAZ world-space points (fixed axis-sign approximation: MediaPipe `+y` down / `+z` toward camera → DAZ `+y` up / `+z` toward viewer).
4. Each of the four target points is applied via `figure.hand_to_target(...)` / `figure.foot_to_target(...)`, DAZ Studio's built-in damped-least-squares IK aligner (same solver as `character/ik_bone_to_target/`).
5. All four limb solves are wrapped in one `scene.undo("Apply photo pose")` block, so `Ctrl+Z` undoes the entire pose in a single step.

## Output

Console output shows the calibrated scale, each limb's target point and IK convergence status (`OK` / `NOT CONVERGED`), and with `--debug`, the resolved bone chain and iteration/error diagnostics per limb.

## Limitations

- **Reset the figure to a neutral pose before each run.** Calibration reads the figure's *current* shoulder-joint (`l_upperarm`/`r_upperarm`) and hip world positions — if a previous pose is still applied, the scale/origin will be calibrated against a contorted rig instead of a clean rest pose.
- **The photo must show one person, unobstructed, full body.** MediaPipe only returns one pose (`num_poses=1`) and will silently lock onto whichever candidate it's most confident in — feeding it a multi-person image (e.g. a reference mood board / contact sheet) gets you *a* pose, just not necessarily the one you meant. If results look physically implausible (e.g. ankles ending up near head height), check what MediaPipe actually detected before assuming the math is wrong.
- Only wrist and ankle effectors are targeted — elbow/knee bend comes from the IK solve, not directly from the photo, so unusual poses (e.g. a sharply bent knee, or a seated/kneeling pose where legs fold under the body) may not match exactly, since this example doesn't solve knee position directly.
- Torso, spine, and head orientation are left at rest — this example doesn't attempt full-skeleton pose matching.
- Axis-sign mapping assumes a roughly frontal photo; side-on or unusual camera angles may need `--scale` tuning or produce mirrored-looking results.

## Related Examples

- [ik_bone_to_target](../../character/ik_bone_to_target/) — the underlying IK aligner, demonstrated directly
- [expression_transfer](../expression_transfer/) — the equivalent workflow for facial expressions
- See main repository [README](../../README.md) for related examples
