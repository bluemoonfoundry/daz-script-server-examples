# Sprite Matrix

**Level:** Advanced
**Category:** Rendering

## Overview

Production pipeline for one sprite: given a scene with the sprite already loaded (a specific outfit, A-pose) and a JSON spec describing a matrix of pose x expression combinations, renders front and over-the-shoulder (back) camera angles for every combo -- beauty image plus Normal/Depth Iray Canvas passes -- then stylizes each render into a "graphic-novel naturalism" look via a ComfyUI multi-ControlNet (normal, depth, lineart) + LoRA img2img workflow -- all three passes conditioned through a single SDXL union ControlNet model, re-tagged per pass via `SetUnionControlNetType`. Both stages are resumable (skip-by-file-existence), so a plain rerun after a crash or partial failure is self-healing.

## What You'll Learn

- Practical implementation of sprite_matrix workflow
- SDK patterns for remote DAZ Studio control
- Production-ready code structure

**SDK features used:**
- `DazRenderSettings`
- `DazPose`
- `DazSceneState`
- `DazScene.find_skeleton_by_label()`

## Prerequisites

- DAZ Studio with DazScriptServer plugin running
- `dazpy` installed (`pip install dazpy`)
- Basic Python knowledge

## Dependencies

Install additional dependencies:
```bash
pip install -r requirements.txt
```

## Usage

```bash
pip install -r sprite_matrix/requirements.txt
```

## How It Works

[See code comments in the script for detailed implementation walkthrough]

## Output

[Example-specific output description]

## Related Examples

- See main repository [README](../../README.md) for related examples
