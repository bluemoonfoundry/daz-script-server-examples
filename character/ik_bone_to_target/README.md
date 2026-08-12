# Ik Bone To Target

**Level:** Advanced
**Category:** Character

## Overview

Moves a specified bone on one figure toward a named target node using a simple iterative IK approach — useful for quick posing, reach tests, and interaction scripting.  Optionally restores the original pose afterward.

## What You'll Learn

- Practical implementation of ik_bone_to_target workflow
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
python character/ik_bone_to_target.py --source MadisonG9 --bone r_hand --target HandTarget
python character/ik_bone_to_target.py --source BobG8 --bone lFoot --target AliceG8 --restore
python character/ik_bone_to_target.py --source MadisonG9 --bone r_hand --target HandTarget --dry-run
```


### Arguments

| Argument | Default | Description |
|---|---|---|
| `--source LABEL` | *(required)* | Figure whose bone should move |
| `--bone NAME` | *(required)* | Effector bone to move toward the target |
| `--target LABEL` | *(required)* | Target scene node (any node — figure, prop, null) |
| `--restore` | off | Restore the original bone rotation after printing the result |


## How It Works

[See code comments in the script for detailed implementation walkthrough]

## Output

[Example-specific output description]

## Related Examples

- See main repository [README](../../README.md) for related examples
