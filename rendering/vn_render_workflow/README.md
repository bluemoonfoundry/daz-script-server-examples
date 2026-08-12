# Vn Render Workflow

**Level:** Advanced
**Category:** Rendering

## Overview

Four patterns for VN (visual novel) render pipelines.  VN production generates many renders of the same characters in different expressions, costumes, or compositions.  Run `--pattern 0|A|B|C` to execute just one pattern, or omit it to run all four in sequence.

## What You'll Learn

- Practical implementation of vn_render_workflow workflow
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
python rendering/vn_render_workflow.py
python rendering/vn_render_workflow.py --pattern A
python rendering/vn_render_workflow.py --pattern B --figure "Hero" --out C:/vn/renders
python rendering/vn_render_workflow.py --pattern C --figure "Alice" --figure2 "Bob"
```


### Arguments

| Argument | Default | Description |
|---|---|---|
| `--pattern {0,A,B,C}` | all | Run only this pattern |
| `--out DIR` | `y:/tmp/vn` | Output directory |
| `--figure LABEL` | `Genesis 9` | Primary figure label |
| `--figure2 LABEL` | `Genesis 9.1` | Second figure (Pattern C only) |
| `--width PX` | `1920` | Render width |


## How It Works

[See code comments in the script for detailed implementation walkthrough]

## Output

[Example-specific output description]

## Related Examples

- See main repository [README](../../README.md) for related examples
