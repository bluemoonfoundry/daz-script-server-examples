# Capture Viewport

**Level:** Beginner
**Category:** Rendering

## Overview

Captures the active DAZ Studio 3D viewport to a file without triggering an iRay render.  Three modes cover the common use cases:

## What You'll Learn

- Practical implementation of capture_viewport workflow
- SDK patterns for remote DAZ Studio control
- Production-ready code structure

**SDK features used:**
- `DazViewport.capture()`
- `DazViewport.capture_sprite()`
- `DazViewport.is_available()`
- `DazViewport.get_size()`

## Prerequisites

- DAZ Studio with DazScriptServer plugin running
- `dazpy` installed (`pip install dazpy`)
- Basic Python knowledge

## Dependencies

No additional dependencies beyond `dazpy`.

## Usage

```bash
pip install rembg
```


### Arguments

| Argument | Default | Description |
|---|---|---|
| `--mode {raw,clean,sprite}` | `clean` | Capture mode |
| `--output PATH` | *(required)* | Output file path (`.png` recommended) |
| `--backdrop R,G,B` | *(unchanged)* | Override viewport background colour during capture |
| `--no-alpha-matting` | off | Sprite mode: disable alpha matting (faster, lower edge quality) |
| `--daz-url URL` | `http://127.0.0.1:18811` | DAZ Studio script server URL |
| `--dry-run` | off | Print the plan without executing |


## How It Works

[See code comments in the script for detailed implementation walkthrough]

## Output

[Example-specific output description]

## Related Examples

- See main repository [README](../../README.md) for related examples
