# Animation Mixing

**Level:** Advanced
**Category:** Animation

## Overview

Treats captured animation files (from `animation_frame_dump.py`) as editable clips.  All operations — clipping, crossfading, concatenation — run entirely in Python with no HTTP calls.  The result can be pushed back to a live figure in a single call when needed.

## What You'll Learn

- Practical implementation of animation_mixing workflow
- SDK patterns for remote DAZ Studio control
- Production-ready code structure

**SDK features used:**
- `DazAnimation.load()`
- `DazAnimation.clip()`
- `DazAnimation.blend()`
- `DazAnimation.append()`
- `DazAnimation.as_pose()`
- `DazAnimation.apply()`
- `len(anim)`
- `anim[i]`
- `DazPose.apply()`

## Prerequisites

- DAZ Studio with DazScriptServer plugin running
- `dazpy` installed (`pip install dazpy`)
- Basic Python knowledge

## Dependencies

No additional dependencies beyond `dazpy`.

## Usage

```bash
python animation/animation_mixing.py clip   --anim walk.json --start 10 --end 40 --out walk_loop.json
python animation/animation_mixing.py blend  --a walk.json --b run.json --t 0.5 --out trot.json
python animation/animation_mixing.py append --a intro.json --b main.json --out full.json
python animation/animation_mixing.py apply  --anim walk.json --frame 0 --figure "Genesis 9"
```


### Arguments

| Argument | Default | Description |
|---|---|---|
| `--anim FILE` | *(required)* | Source animation JSON |
| `--start N` | *(required)* | First scene frame to keep |
| `--end N` | *(required)* | Last scene frame to keep |
| `--out FILE` | *(required)* | Output JSON path |


## How It Works

[See code comments in the script for detailed implementation walkthrough]

## Output

[Example-specific output description]

## Related Examples

- See main repository [README](../../README.md) for related examples
