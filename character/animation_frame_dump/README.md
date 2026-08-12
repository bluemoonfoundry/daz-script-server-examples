# Animation Frame Dump

**Level:** Intermediate
**Category:** Character

## Overview

Scrubs through the timeline entirely inside DazScript — `Scene.setFrame()` advances the playhead server-side, so the entire animation is captured in a single HTTP call with no per-frame round-trips.

## What You'll Learn

- Practical implementation of animation_frame_dump workflow
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
python character/animation_frame_dump.py --figure "Genesis 9" --out anim.json
python character/animation_frame_dump.py --figure "Genesis 9" --out anim.json --morphs
```


### Arguments

| Argument | Default | Description |
|---|---|---|
| `--figure LABEL` | `Genesis 9` | Figure label |
| `--out FILE` | `anim.json` | Output JSON file |


## How It Works

[See code comments in the script for detailed implementation walkthrough]

## Output

[Example-specific output description]

## Related Examples

- See main repository [README](../../README.md) for related examples
