# Turntable

**Level:** Beginner
**Category:** Rendering

## Overview

Rotates a figure around its local Y axis in equal steps and renders each frame to a numbered PNG.  Existing X and Z rotations are preserved so a posed character stays posed throughout the spin.

## What You'll Learn

- Practical implementation of turntable workflow
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
ffmpeg -framerate 24 -i frame_%03d.png turntable.mp4
```


### Arguments

| Argument | Default | Description |
|---|---|---|
| `--figure LABEL` | `Genesis 9` | Figure label |
| `--steps N` | `36` | Number of frames for a full 360° rotation |
| `--out DIR` | `y:/tmp/turntable` | Output directory |
| `--width PX` | `1920` | Render width in pixels |


## How It Works

[See code comments in the script for detailed implementation walkthrough]

## Output

[Example-specific output description]

## Related Examples

- See main repository [README](../../README.md) for related examples
