# Comfyui Enhance

**Level:** Advanced
**Category:** Rendering

## Overview

End-to-end pipeline that captures the active DAZ Studio viewport and submits the result to a local [ComfyUI](https://github.com/comfyanonymous/ComfyUI) instance for img2img enhancement via a photorealistic diffusion model.

## What You'll Learn

- Practical implementation of comfyui_enhance workflow
- SDK patterns for remote DAZ Studio control
- Production-ready code structure

**SDK features used:**
- `DazViewport.capture()`
- `DazClient`

## Prerequisites

- DAZ Studio with DazScriptServer plugin running
- `dazpy` installed (`pip install dazpy`)
- Basic Python knowledge

## Dependencies

Install additional dependencies:
```bash
pip install -r requirements.txt
```

## Usage

```bash
pip install requests watchdog Pillow
```


### Arguments

| Argument | Default | Description |
|---|---|---|
| `--output PATH` | *(required)* | Output path for the enhanced image |
| `--checkpoint NAME` | *(from config.py)* | ComfyUI checkpoint filename override |
| `--denoise FLOAT` | `0.45` | Denoise strength (0.0 = no change, 1.0 = full generation) |
| `--no-watch` | off | Skip file watcher; use `--snapshot-path` directly |
| `--snapshot-path PATH` | — | Existing snapshot to submit (requires `--no-watch`) |
| `--dry-run` | off | Print plan without executing |


## How It Works

[See code comments in the script for detailed implementation walkthrough]

## Output

[Example-specific output description]

## Related Examples

- See main repository [README](../../README.md) for related examples
