# Keyframe Baking

**Level:** Advanced
**Category:** Animation

## Overview

Reads the evaluated bone rotations and morph values of an animated figure at the current frame, then bakes the full play range to explicit keyframes in one HTTP call.  After baking, the animation no longer depends on IK rigs, expression controllers, or other drivers — useful before FBX/BVH export or after pushing a captured clip back to the timeline.

## What You'll Learn

- Practical implementation of keyframe_baking workflow
- SDK patterns for remote DAZ Studio control
- Production-ready code structure

**SDK features used:**
- `DazSkeleton.bone_rotations()`
- `DazSkeleton.morph_values(nonzero_only=True)`
- `DazSkeleton.bake_bone_rotations()`
- `DazSkeleton.bake_morphs()`
- `DazSkeleton.bake()`
- `DazScene.play_range()`

## Prerequisites

- DAZ Studio with DazScriptServer plugin running
- `dazpy` installed (`pip install dazpy`)
- Basic Python knowledge

## Dependencies

No additional dependencies beyond `dazpy`.

## Usage

```bash
python animation/keyframe_baking.py --figure "Genesis 9"
python animation/keyframe_baking.py --figure "Genesis 9" --morphs
python animation/keyframe_baking.py --figure "Genesis 9" --start 10 --end 90 --morphs
python animation/keyframe_baking.py --figure "Genesis 9" --preview
```


### Arguments

| Argument | Default | Description |
|---|---|---|
| `--figure LABEL` | `Genesis 9` | Figure label |
| `--start N` | play range start | First frame to bake |
| `--end N` | play range end | Last frame to bake |
| `--morphs` | off | Also bake morph channels alongside bone rotations |
| `--preview` | off | Print current bone/morph state without writing any keyframes |


## How It Works

[See code comments in the script for detailed implementation walkthrough]

## Output

[Example-specific output description]

## Related Examples

- See main repository [README](../../README.md) for related examples
