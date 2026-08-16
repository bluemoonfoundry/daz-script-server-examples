# Unreal Headless Render

**Level:** Advanced
**Category:** Rendering

## Overview

End-to-end pipeline that captures a character's pose and morphs from a live DAZ Studio scene and renders a 4K still through Unreal Engine 5's Movie Render Queue (MRQ), running headlessly. Built for high-volume Visual Novel-style rendering where DAZ Studio's Iray renderer is too slow to use directly.

## What You'll Learn

- Reading a figure's full bone rotation set and active morph values in one HTTP call each (`DazSkeleton.bone_rotations_quat`, `DazSkeleton.morph_values`)
- Converting DAZ's Y-up axis convention to Unreal's Z-up convention for rotations, via quaternions (`dazpy.math3.Y_UP_TO_Z_UP`), rather than remapping Euler angles directly
- Bridging two separate applications (DAZ Studio, Unreal Engine) through a JSON job file and subprocess orchestration
- Driving Unreal's Movie Render Queue from Python in headless/commandlet mode

**SDK features used:**
- `DazScene.primary_selection()` / `find_skeleton_by_label()` / `find_skeleton()`
- `DazSkeleton.bone_rotations_quat()`
- `DazSkeleton.morph_values(nonzero_only=True)`
- `dazpy.math3.Quat`, `AxisRemap`, `Y_UP_TO_Z_UP`

## Prerequisites

- DAZ Studio with DazScriptServer plugin running
- `dazpy` installed (`pip install -e .` from the `daz-script-server` repo root)
- Unreal Engine 5 project with:
  - **Movie Render Queue** plugin enabled (Edit > Plugins > "Movie Render Queue")
  - **Python Editor Script Plugin** enabled (Edit > Plugins > "Python Editor Script Plugin")
  - `-ExecutePythonScript` startup Python execution allowed — this is on by default for the Python Editor Script Plugin; no extra project setting is normally required, but confirm under Project Settings > Plugins > Python that "Developer Mode" is on if you hit import errors for project-local modules like `morph_name_map.py`
  - A target `SkeletalMeshActor` whose skeleton shares bone names with the DAZ figure (e.g. exported via a DAZ-to-Unreal bridge), placed in the level
  - A minimal Level Sequence asset per shot/camera for MRQ to render (see Caveat 2 in `ue_headless_render.py`)
  - The Anim Blueprint / Animation Mode on that actor's `SkeletalMeshComponent` configured so nothing overwrites your manually-set bones/morphs before capture (see Caveat 1 in `ue_headless_render.py`)

## Dependencies

Install additional dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the full pipeline in one command

```bash
python render_orchestrator.py \
    --ue-cmd "C:/Program Files/Epic Games/UE_5.4/Engine/Binaries/Win64/UnrealEditor-Cmd.exe" \
    --uproject "C:/MyProject/MyProject.uproject" \
    --character-id BP_Character_Heroine \
    --camera-name CineCamera_Shot005 \
    --output-dir "C:/VNRenders/Scene01" \
    --output-filename Scene01_Shot005.png
```

Add `--dry-run` first to print the exact commands without touching DAZ Studio or launching Unreal, to confirm the wiring.

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--ue-cmd PATH` | *(required)* | Path to `UnrealEditor-Cmd.exe` |
| `--uproject PATH` | *(required)* | Path to the target `.uproject` |
| `--character-id NAME` | *(required)* | Actor label/name in the Unreal level |
| `--camera-name NAME` | *(required)* | `CineCameraActor` name in the level |
| `--output-dir PATH` | *(required)* | Directory MRQ writes the PNG to |
| `--output-filename NAME` | *(required)* | Output PNG filename |
| `--resolution W H` | `3840 2160` | Output resolution |
| `--warmup-frames N` | `20` | Frames to let Lumen/Path Tracer converge before capture |
| `--figure NAME` / `--label NAME` | — | Select a specific DAZ figure (default: current scene selection) |
| `--job-path PATH` | `temp/render_job.json` | Where the intermediate render job JSON is written |
| `--nullrhi` | off | Pass `-NullRHI` to Unreal — disables GPU rendering; only useful for smoke-testing the pipeline wiring, **not** for real renders |
| `--timeout SECONDS` | `600` | Max time to wait for the Unreal render process |
| `--dry-run` | off | Print the plan without executing anything |

`daz_exporter.py` and `ue_headless_render.py` can also be run independently — see their module docstrings for standalone usage.

## How It Works

1. **`daz_exporter.py`** connects to DAZ Studio, finds the figure (selection or `--figure`/`--label`), reads every bone's local rotation as a quaternion and every non-zero morph value in two HTTP round trips, remaps the bone rotations from DAZ's Y-up to Unreal's Z-up convention, and writes `render_job.json`.
2. **`render_orchestrator.py`** launches `UnrealEditor-Cmd.exe <uproject> -ExecutePythonScript="ue_headless_render.py" -RenderJobPath="..." -Unattended -nosplash`.
3. **`ue_headless_render.py`** (running inside Unreal's embedded Python) reads `render_job.json`, finds the matching `SkeletalMeshActor`, applies morph target values (via `morph_name_map.py`'s Daz→Unreal name table) and bone rotations, then configures and submits a Movie Render Queue job (PNG output, resolution, warm-up frames) and renders synchronously.
4. **`render_orchestrator.py`** waits for the Unreal process to exit and confirms the output PNG exists.

## Output

A single 4K (or `--resolution`-specified) PNG at `<output-dir>/<output-filename>`, plus `render_job.json` in `temp/` (or `--job-path`) as an inspectable intermediate artifact.

## Known Limitations / Caveats

- **Bone name matching only** — this example assumes the Unreal skeleton's bone names match the DAZ figure's exactly (true for a DAZ-to-Unreal bridge export). Structurally different skeletons need a bone name-mapping layer, which this example does not implement.
- **Morph name mapping is manual** — DAZ morph names rarely match Unreal morph target names even from a bridge export. Fill in `morph_name_map.py`'s `MORPH_NAME_MAP` for your mesh; unmapped morphs are skipped with a warning, never a hard failure.
- **`unreal.*` API surface is version-sensitive** — the exact bone-space enum and warm-up-frame-count property name used in `ue_headless_render.py` may need adjusting for your installed UE5 minor version; see Caveat 3 in that file's docstring.
- **Not a retargeting solution** — this pipeline transfers a literal pose (bone rotations + morph values), not a semantic retarget across different rigs/proportions.

## Related Examples

- [`character/pose_transfer`](../../character/pose_transfer/README.md) — the same bulk-bone-rotation read pattern, applied between two DAZ figures instead of exporting to Unreal
- [`character/animation_frame_dump`](../../character/animation_frame_dump/README.md) — capturing a full animation timeline instead of a single pose
- [`rendering/comfyui_enhance`](../comfyui_enhance/README.md) — another external-tool render pipeline orchestrated from DAZ Studio, with the same subprocess/orchestrator structure
- See main repository [README](../../README.md) for related examples
