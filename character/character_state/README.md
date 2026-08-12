# Character State

**Level:** Intermediate
**Category:** Character

## Overview

Saves a character's complete state — shape morphs, expression / FACS controls, and bone rotations — to a JSON file.  Restores it on demand. Only non-default values are stored so the file stays compact.

## What You'll Learn

- Practical implementation of character_state workflow
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
python character/character_state.py save    --figure "Genesis 9" --out state.json
python character/character_state.py restore --figure "Genesis 9" --file state.json
```


### Arguments

| Argument | Default | Description |
|---|---|---|
| `--figure LABEL` | `Genesis 9` | Figure label as shown in the Scene panel |
| `--out FILE` | `state.json` | Output JSON file |


## How It Works

[See code comments in the script for detailed implementation walkthrough]

## Output

[Example-specific output description]

## Related Examples

- See main repository [README](../../README.md) for related examples
