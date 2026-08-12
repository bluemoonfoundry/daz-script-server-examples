# Pose Interpolation

**Level:** Intermediate
**Category:** Animation

## Overview

Loads two state files produced by `character_state.py`, interpolates all bone rotations, morph values, and FACS properties across N steps using a configurable easing curve, and renders each frame.

## What You'll Learn

- Practical implementation of pose_interpolation workflow
- SDK patterns for remote DAZ Studio control
- Production-ready code structure

## Prerequisites

- DAZ Studio with DazScriptServer plugin running
- `dazpy` installed (`pip install dazpy`)
- Basic Python knowledge

## Dependencies

No additional dependencies beyond `dazpy`.

## Usage

```bash
ffmpeg -framerate 24 -i frame_%03d.png interpolation.mp4
```


### Arguments

| Argument | Default | Description |
|---|---|---|
| `--a FILE` | *(required)* | Start state JSON (from `character_state.py save`) |
| `--b FILE` | *(required)* | End state JSON |
| `--steps N` | `10` | Number of frames including start and end |
| `--ease NAME` | `ease_in_out` | Easing curve: `linear`, `ease_in`, `ease_out`, `ease_in_out`, `ease_in_cubic`, `ease_out_cubic`, `ease_in_out_cubic`, `bounce_out` |
| `--out DIR` | `y:/tmp/interpolation` | Output directory |
| `--width PX` | `1920` | Render width |
| `--height PX` | `1080` | Render height |


## How It Works

[See code comments in the script for detailed implementation walkthrough]

## Output

[Example-specific output description]

## Related Examples

- See main repository [README](../../README.md) for related examples
