# Material Color Variations

**Level:** Intermediate
**Category:** Rendering

## Overview

Renders a node's material surface in multiple diffuse colours.  The original colour is saved before the loop and restored afterward — including if the run is interrupted.

## What You'll Learn

- Practical implementation of material_color_variations workflow
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
python rendering/material_color_variations.py --node "Cube" --material "Default"
python rendering/material_color_variations.py \
    --node "Shirt" --material "Fabric" \
    --colors "#C0392B" "#2980B9" "#27AE60" \
    --out C:/swatches --width 1920 --height 1080
```


### Arguments

| Argument | Default | Description |
|---|---|---|
| `--node LABEL` | `Genesis 9` | Scene node whose material to modify |
| `--material NAME` | `Torso` | Material surface name |
| `--colors HEX …` | 8-colour palette | Hex colours to render (`#RRGGBB`) |
| `--out DIR` | `y:/tmp/color_variations` | Output directory |
| `--width PX` | `1920` | Render width |


## How It Works

[See code comments in the script for detailed implementation walkthrough]

## Output

[Example-specific output description]

## Related Examples

- See main repository [README](../../README.md) for related examples
