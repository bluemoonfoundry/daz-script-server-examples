# Scene To Usd

**Level:** Advanced
**Category:** Export

## Overview

Interrogates the live DAZ Studio scene through the HTTP API and writes a Pixar USD file — without touching the DAZ Studio UI, loading extra plugins, or modifying the scene.

## What You'll Learn

- Practical implementation of scene_to_usd workflow
- SDK patterns for remote DAZ Studio control
- Production-ready code structure

**SDK features used:**
- `DazGeometry.bounding_box_posed()`
- `DazGeometry.face_vertex_indices_all()`
- `DazGeometry.vertex_positions_all()`
- `DazGeometry.uv_positions_all()`
- `DazScene.skeletons()`

## Prerequisites

- DAZ Studio with DazScriptServer plugin running
- `dazpy` installed (`pip install dazpy`)
- Basic Python knowledge

## Dependencies

No additional dependencies beyond `dazpy`.

## Usage

```bash
pip install usd-core
```


### Arguments

| Argument | Default | Description |
|---|---|---|
| `--out FILE` | `scene.usda` | Output USD file (`.usda` = ASCII, `.usdc` = binary) |
| `--morphs` | off | Export active shape morphs as UsdSkel blend shapes |
| `--figure LABEL` | all figures | Export only the named figure |


## How It Works

[See code comments in the script for detailed implementation walkthrough]

## Output

[Example-specific output description]

## Related Examples

- See main repository [README](../../README.md) for related examples
