# Scene Save Copy

**Level:** Beginner
**Category:** Fundamentals

## Overview

Saves a copy of the current scene to a new path — the Python equivalent of DAZ Studio's "Save a Copy As…" menu option — without changing the scene's internal filename or dirty flag.

## What You'll Learn

- Practical implementation of scene_save_copy workflow
- SDK patterns for remote DAZ Studio control
- Production-ready code structure

**SDK features used:**
- `DazScene.save_copy()`
- `DazScene.filename()`
- `DazScene.needs_save()`

## Prerequisites

- DAZ Studio with DazScriptServer plugin running
- `dazpy` installed (`pip install dazpy`)
- Basic Python knowledge

## Dependencies

No additional dependencies beyond `dazpy`.

## Usage

```bash
python fundamentals/scene_save_copy.py --dest C:/backups/scene_v2.duf
python fundamentals/scene_save_copy.py --dest C:/backups/scene_v2.duf --compare
python fundamentals/scene_save_copy.py --dest C:/backups/scene_v2.duf --dry-run
```


### Arguments

| Argument | Default | Description |
|---|---|---|
| `--dest PATH` | *(required)* | Absolute destination path on the DAZ Studio host |
| `--compare` | off | After saving, print source/copy sizes and whether the files are byte-identical |
| `--dry-run` | off | Print the strategy that would be used without writing anything |
| `--host HOST` | `127.0.0.1` | Server host |
| `--port PORT` | `18811` | Server port |


## How It Works

[See code comments in the script for detailed implementation walkthrough]

## Output

[Example-specific output description]

## Related Examples

- See main repository [README](../../README.md) for related examples
