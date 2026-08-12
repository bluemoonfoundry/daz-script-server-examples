# Expression Transfer

**Level:** Advanced
**Category:** AI/Vision

## Overview

Extracts a facial expression from a photo using MediaPipe FaceLandmarker, computes Action Unit (AU) magnitudes from landmark geometry entirely in Python, and applies the result to a Genesis 9 figure's FACS HD expression controls in a single HTTP call.

## What You'll Learn

- Practical implementation of expression_transfer workflow
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
| `image` | *(required unless `--list-properties`)* | Path to source image (JPEG, PNG, or any format OpenCV supports) |
| `--figure LABEL` | `Genesis 9` | Target figure label |
| `--scale FLOAT` | `1.0` | Global expression scale factor — reduce if morphs are over-driven |
| `--no-reset` | off | Blend onto the current expression instead of zeroing FACS first |
| `--list-properties` | off | List all numeric properties on the figure and exit |
| `--search TERM` | — | Filter `--list-properties` output by case-insensitive substring |


## How It Works

[See code comments in the script for detailed implementation walkthrough]

## Output

[Example-specific output description]

## Related Examples

- See main repository [README](../../README.md) for related examples
