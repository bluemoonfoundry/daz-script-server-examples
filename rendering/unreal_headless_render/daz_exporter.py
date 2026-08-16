"""DAZ Studio Script Server example: export a pose + morphs as an Unreal render job.

PURPOSE
-------
This script is a demonstration of what the DAZ Studio Script Server makes
possible. It reads the current bone orientations and active morph values of
a figure in the live DAZ Studio scene, converts the bone rotations from
DAZ's Y-up axis convention to Unreal's Z-up convention, and writes the
result to a JSON "render job" file that a headless Unreal Engine process can
consume to reproduce the pose on a matching SkeletalMeshActor.

It is an *example*, not a production asset pipeline. It assumes the target
Unreal skeleton shares bone names with the DAZ figure (e.g. exported via a
DAZ-to-Unreal bridge/retarget) and does not attempt to solve retargeting
between structurally different skeletons.

WHAT IT DEMONSTRATES
--------------------
  - Finding the currently-selected figure in the live scene
    (DazScene.primary_selection + DazScene.find_skeleton_by_label)
  - Reading every bone's local rotation as a quaternion in a single HTTP
    call (DazSkeleton.bone_rotations_quat)
  - Reading only the non-zero morph values in a single HTTP call
    (DazSkeleton.morph_values(nonzero_only=True))
  - Converting an axis convention (Y-up -> Z-up) for rotations with
    dazpy.math3.AxisRemap / Y_UP_TO_Z_UP, which composes correctly through
    quaternions without the gimbal/rotation-order pitfalls of remapping
    Euler triples directly

ENVIRONMENT SETUP
-----------------
1. DAZ Studio must be running with the DazScriptServer plugin loaded and its
   HTTP server active (default: 127.0.0.1:18811). Verify with:

       curl http://127.0.0.1:18811/health

2. Install the Python dependencies in a virtual environment:

       python -m venv .venv
       .venv\\Scripts\\activate          # Windows
       # source .venv/bin/activate     # macOS / Linux
       pip install -r requirements.txt

3. Install or develop-install the dazpy SDK (from the daz-script-server
   repo root):

       pip install -e .

4. Open a scene in DAZ Studio with the figure to export selected (or pass
   --figure/--label explicitly), then run:

       python daz_exporter.py --character-id BP_Character_Heroine \\
           --camera-name CineCamera_Shot005 \\
           --output-dir "C:/VNRenders/Scene01" \\
           --output-filename Scene01_Shot005.png

Usage:
    python daz_exporter.py --character-id BP_Character_Heroine --camera-name CineCamera_Shot005
    python daz_exporter.py --label "Genesis 9" --character-id BP_Character_Heroine --camera-name Cam1 --out temp/render_job.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from dazpy import DazScene
from dazpy.math3 import Quat, Y_UP_TO_Z_UP

SCHEMA_VERSION = 1


def build_render_job(
    figure,
    *,
    character_id: str,
    camera_name: str,
    output_dir: str,
    output_filename: str,
    resolution: tuple[int, int],
    warmup_frames: int,
) -> dict:
    """Read the figure's current pose/morphs and assemble a render_job dict.

    Bone rotations are converted from DAZ's Y-up axis convention to
    Unreal's Z-up convention before being written out, so
    ue_headless_render.py can apply them without any further conversion.
    """
    raw_bones = figure.bone_rotations_quat()
    bones_out = {}
    for name, q in raw_bones.items():
        remapped = Y_UP_TO_Z_UP.apply_quat(Quat.from_dict(q))
        bones_out[name] = remapped.to_dict()

    morphs_out = figure.morph_values(nonzero_only=True)

    return {
        "schema_version": SCHEMA_VERSION,
        "character_id": character_id,
        "bones": bones_out,
        "morphs": morphs_out,
        "render_config": {
            "camera_name": camera_name,
            "output_dir": output_dir,
            "output_filename": output_filename,
            "resolution": list(resolution),
            "warmup_frames": warmup_frames,
        },
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--figure", help="Figure's internal DazScript name (mutually exclusive with --label)")
    p.add_argument("--label", help="Figure's user-visible label (default: use the current scene selection)")
    p.add_argument("--character-id", required=True, help="Target actor name/label in the Unreal level")
    p.add_argument("--camera-name", required=True, help="CineCameraActor name in the Unreal level")
    p.add_argument("--output-dir", required=True, help="Directory Unreal's Movie Render Queue writes the PNG to")
    p.add_argument("--output-filename", required=True, help="Output PNG filename")
    p.add_argument("--resolution", nargs=2, type=int, default=[3840, 2160], metavar=("WIDTH", "HEIGHT"))
    p.add_argument("--warmup-frames", type=int, default=20, help="Frames to let Lumen/Path Tracer lighting converge before capture")
    p.add_argument("--out", default=os.path.join("temp", "render_job.json"), help="Path to write the render_job.json file")
    return p.parse_args()


def _fail(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    args = _parse_args()
    if args.figure and args.label:
        _fail("--figure and --label are mutually exclusive")

    scene = DazScene()
    try:
        if args.label:
            figure = scene.find_skeleton_by_label(args.label)
        elif args.figure:
            figure = scene.find_skeleton(args.figure)
        else:
            node = scene.primary_selection()
            if node is None:
                _fail("No figure selected in the scene, and no --figure/--label given.")
            figure = scene.find_skeleton_by_label(node.label)
    except Exception as exc:
        _fail(f"Could not find figure in scene: {exc}")

    job = build_render_job(
        figure,
        character_id=args.character_id,
        camera_name=args.camera_name,
        output_dir=args.output_dir,
        output_filename=args.output_filename,
        resolution=tuple(args.resolution),
        warmup_frames=args.warmup_frames,
    )

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(job, f, indent=2)

    print(f"Wrote {len(job['bones'])} bone(s) and {len(job['morphs'])} morph(s) -> {args.out}")
