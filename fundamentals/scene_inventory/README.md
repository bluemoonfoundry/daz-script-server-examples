# Scene Inventory

**Level:** Intermediate
**Category:** Fundamentals

## Overview

Collects a structured report for every node in the scene — type, label, world position, visibility, material names, vertex count, and (for figures) bone and morph counts.  Everything is gathered in a single DazScript call.

## What You'll Learn

- Practical implementation of scene_inventory workflow
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
python fundamentals/scene_inventory.py
python fundamentals/scene_inventory.py --out inventory.json --pretty
```


### Arguments

| Argument | Default | Description |
|---|---|---|
| `--out FILE` | stdout | Write JSON to this file instead of stdout |


## How It Works

[See code comments in the script for detailed implementation walkthrough]

## Output

[Example-specific output description]

## Related Examples

- See main repository [README](../../README.md) for related examples
