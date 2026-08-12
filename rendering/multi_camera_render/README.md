# Multi Camera Render

**Level:** Beginner
**Category:** Rendering

## Overview

Iterates every camera in the scene (or a named subset) and renders from each one to `<out>/<camera_label>.png`.  Useful for storyboarding and covering multiple angles in a single run.

## What You'll Learn

- Practical implementation of multi_camera_render workflow
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
python rendering/multi_camera_render.py
python rendering/multi_camera_render.py --out C:/renders --width 1920 --height 1080
python rendering/multi_camera_render.py --cameras "Front" "Side" "Hero Shot"
```


### Arguments

| Argument | Default | Description |
|---|---|---|
| `--out DIR` | `y:/tmp/multicam` | Output directory |
| `--width PX` | `1920` | Render width |
| `--height PX` | `1080` | Render height |


## How It Works

[See code comments in the script for detailed implementation walkthrough]

## Output

[Example-specific output description]

## Related Examples

- See main repository [README](../../README.md) for related examples
