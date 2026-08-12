# Webcam Expression Mirror

**Level:** Advanced
**Category:** AI/Vision

## Overview

Captures frames from your webcam, runs MediaPipe FaceLandmarker on each frame, and streams the resulting FACS morph values to a Genesis 9 figure at up to `--fps` updates per second.  The figure's expression updates live as your face moves.

## What You'll Learn

- Practical implementation of webcam_expression_mirror workflow
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
pip install mediapipe opencv-python numpy
```


### Arguments

| Argument | Default | Description |
|---|---|---|
| `--figure LABEL` | `Genesis 9` | Target figure label |
| `--scale FLOAT` | `1.0` | Global expression scale factor |
| `--camera N` | `0` | OpenCV camera index (try `1`, `2`, … for external webcams) |
| `--fps N` | `10` | Max DAZ Studio updates per second |
| `--smooth FLOAT` | `0.5` | EMA smoothing: `0` = raw/responsive, `0.9` = very smooth |


## How It Works

[See code comments in the script for detailed implementation walkthrough]

## Output

[Example-specific output description]

## Related Examples

- See main repository [README](../../README.md) for related examples
