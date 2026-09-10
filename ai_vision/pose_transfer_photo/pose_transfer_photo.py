"""DAZ Studio Script Server example: apply a body pose from a photo to a Genesis 9 figure.

PURPOSE
-------
This script is a demonstration of what the DAZ Studio Script Server makes
possible.  It extracts 33 body landmarks from a source image using MediaPipe
PoseLandmarker, converts the wrist/ankle landmarks into DAZ Studio world-space
targets, and drives the figure's hands and feet toward those targets using the
IK aligner already built into dazpy (`DazSkeleton.hand_to_target` /
`.foot_to_target`, the same solver `ik_bone_to_target.py` exercises directly).

It is an *example*, not a full-body motion-capture tool.  Only the four limb
effectors (both wrists, both ankles) are driven — elbows/knees follow from the
IK chain solve rather than being matched to the photo directly, and the torso,
spine, and head are left at rest.  A real system would also solve pelvis/spine
orientation and account for camera perspective.  The goal here is to show that
the script server's existing IK aligner can be driven from arbitrary Python
vision output, not just from the interaction-recipe / hardcoded-target
call sites shown elsewhere in this repo.

WHAT IT DEMONSTRATES
--------------------
  - Running MediaPipe PoseLandmarker inference in Python (auto-downloads the
    model file on first run), using pose_world_landmarks (metric, hip-centered)
  - Auto-calibrating scale and origin from the live figure's own shoulder width
    and hip position, so results roughly work across different figure heights
    without manual tuning
  - Reusing DazSkeleton.hand_to_target()/.foot_to_target() — the same
    damped-least-squares IK aligner used by character/ik_bone_to_target.py —
    to drive both wrists and both ankles toward photo-derived world points
  - Wrapping the whole pose application in one named undo step
    (scene.undo(...)) so Ctrl+Z in DAZ Studio undoes it in a single step
  - Reporting per-limb IK convergence (iterations, final error) for debugging

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
    python pose_transfer_photo.py photo.jpg --debug
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import urllib.request

import cv2
import mediapipe as mp
import numpy as np

from dazpy import DazScene
from dazpy.exceptions import DazBusyError

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
# hand_to_target()/foot_to_target() rebuild a fresh rig profile (bone_metadata())
# on every call with no built-in retry, and each call itself fires up to
# max_iterations rapid HTTP round-trips. Four limbs back-to-back in one
# scene.undo() block can catch DAZ Studio's main thread still catching up from
# the previous limb's burst, raising StudioBusyError. Retry here instead of
# crashing the whole pose application over one transient hiccup.

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


if __name__ == "__main__":
    # ── CLI ────────────────────────────────────────────────────────────────────────

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("image", help="Path to source image")
    parser.add_argument("--figure", default="Jason Cross",
                        help="DAZ figure label (default: 'Jason Cross')")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="Extra multiplier on top of auto-calibrated scale (default: 1.0)")
    parser.add_argument("--no-hands", dest="hands", action="store_false",
                        help="Skip wrist targets")
    parser.add_argument("--no-feet", dest="feet", action="store_false",
                        help="Skip ankle targets")
    parser.add_argument("--max-iterations", type=int, default=15,
                        help="Max IK iterations per limb (default: 15)")
    parser.add_argument("--tolerance", type=float, default=0.15,
                        help="IK convergence distance in scene units (default: 0.15)")
    parser.add_argument("--debug", action="store_true",
                        help="Print per-limb IK convergence diagnostics")
    args = parser.parse_args()

    scene = DazScene()
    try:
        figure = scene.find_skeleton_by_label(args.figure)
    except Exception:
        sys.exit(f"Error: figure {args.figure!r} not found in scene.")

    print(f"Extracting pose from {args.image!r}...")
    landmarks = extract_world_landmarks(args.image)

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
            f"Bone {exc} not found on {args.figure!r} — this example targets Genesis 9 "
            "bone names (l_upperarm, r_upperarm, hip). Run figure.bones() to list this "
            "figure's actual bone names if it's a different generation."
        )

    unit_scale, mp_hip_mid, daz_hip_world = calibrate(
        landmarks, daz_shoulder_width, daz_hip_world, args.scale
    )
    print(f"Calibrated scale: {unit_scale:.4f} (scene units per photo unit)")

    targets: list[tuple[str, str, tuple[float, float, float]]] = []
    if args.hands:
        targets.append(("hand", "l_hand", to_daz_world(landmarks[L_WRIST], mp_hip_mid, daz_hip_world, unit_scale)))
        targets.append(("hand", "r_hand", to_daz_world(landmarks[R_WRIST], mp_hip_mid, daz_hip_world, unit_scale)))
    if args.feet:
        targets.append(("foot", "l_foot", to_daz_world(landmarks[L_ANKLE], mp_hip_mid, daz_hip_world, unit_scale)))
        targets.append(("foot", "r_foot", to_daz_world(landmarks[R_ANKLE], mp_hip_mid, daz_hip_world, unit_scale)))

    print(f"\nApplying pose to {args.figure!r} ({len(targets)} limb targets)...")
    with scene.undo("Apply photo pose"):
        for i, (kind, anchor, point) in enumerate(targets):
            if kind == "hand":
                result = _call_with_busy_retry(lambda: figure.hand_to_target(
                    point, source_anchor=anchor,
                    max_iterations=args.max_iterations, tolerance=args.tolerance,
                ))
            else:
                result = _call_with_busy_retry(lambda: figure.foot_to_target(
                    point, source_anchor=anchor,
                    max_iterations=args.max_iterations, tolerance=args.tolerance,
                ))
            status = "OK" if result.converged else "NOT CONVERGED"
            print(f"  {anchor:10s} -> {point[0]:+7.2f}, {point[1]:+7.2f}, {point[2]:+7.2f}   [{status}]")
            if args.debug:
                print(f"    chain={result.chain}")
                print(f"    iterations={result.iterations}  "
                      f"initial_error={result.initial_error}  final_error={result.final_error}")
            # Brief settle time so DAZ Studio's main thread catches up before the
            # next limb's rig-profile rebuild — reduces (but doesn't eliminate,
            # hence the retry above) StudioBusyError on back-to-back IK calls.
            if i < len(targets) - 1:
                time.sleep(0.5)

    print(f"\nDone. Applied {len(targets)} limb targets to {args.figure!r}.")
