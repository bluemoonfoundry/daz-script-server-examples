# Scene Event Monitor

**Level:** Intermediate
**Category:** Fundamentals

## Overview

Connects to the `GET /scene/events` SSE stream and reacts to live DAZ Studio activity — node additions, selection changes, time scrubs, scene loads, renders starting/finishing — without polling or modifying the scene.

## What You'll Learn

- Practical implementation of scene_event_monitor workflow
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
python fundamentals/scene_event_monitor.py monitor
python fundamentals/scene_event_monitor.py monitor --filter node,selection
python fundamentals/scene_event_monitor.py monitor --filter render --quiet
```


### Arguments

| Argument | Subcommand | Default | Description |
|---|---|---|---|
| `--filter CATEGORIES` | monitor, log | all | Comma-separated category subset: `node`, `skeleton`, `light`, `camera`, `selection`, `scene`, `time`, `render` |
| `--quiet` | monitor | off | Print only event type names, not full formatted lines |
| `--out FILE` | log | *(required)* | Output JSONL file (appended, not overwritten) |
| `--type EVENT_TYPE` | wait-for | *(required)* | Exact event type to wait for, e.g. `render.finished`, `node.added` |
| `--timeout SECS` | wait-for | `300` | Give up after this many seconds |


## How It Works

[See code comments in the script for detailed implementation walkthrough]

## Output

[Example-specific output description]

## Related Examples

- See main repository [README](../../README.md) for related examples
