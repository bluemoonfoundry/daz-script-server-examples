# Body Measurements

**Level:** Advanced
**Category:** Geometry

## Overview

Computes practical body measurements for a selected figure by pulling the posed mesh into Python, slicing it with horizontal planes, and measuring the torso contour at each slice.  The example targets Genesis 8, Genesis 8.1, and Genesis 9 figures, but the same approach works for other figures too. Each reported measurement includes both centimeters and inches.

## What You'll Learn

- Practical implementation of body_measurements workflow
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
pip install trimesh
```


### Arguments

| Argument | Default | Description |
|---|---|---|
| `--figure LABEL` | `Genesis 9` | Figure label as shown in the Scene panel |
| `--figure-type TYPE` | auto-detect | Force calibration entry: `G9F`, `G9M`, `G8F`, `G8M`, `G8.1F`, `G8.1M` |
| `--sample-step CM` | `0.5` | Slice spacing when searching for local min/max circumferences |
| `--search-window CM` | `5.0` | Half-width of the bust / underbust / low-hip search window |
| `--torso-only` | off | Narrow the bust search band (centroid-based torso loop selection is always active) |
| `--clothing` | off | Print bra and dress size estimates |
| `--pretty` | off | Render summary as compact tables |


## How It Works

[See code comments in the script for detailed implementation walkthrough]

## Output

[Example-specific output description]

## Related Examples

- See main repository [README](../../README.md) for related examples
