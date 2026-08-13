# BVH Import

**Level:** Advanced
**Category:** BVH

## Overview

Parses a BVH motion capture file entirely in Python, retargets the bone rotations to the DAZ Studio internal naming for the detected generation, and drives the live rig one frame at a time over HTTP. Demonstrates pure-Python BVH parsing and frame-by-frame animation application via HTTP.

## What You'll Learn

- Practical implementation of bvh_import workflow
- SDK patterns for remote DAZ Studio control
- Production-ready code structure

**SDK features used:**
- BVH file parsing and retargeting
- Euler rotation conversion
- Frame-by-frame animation application
- Batch bone rotation updates

## Prerequisites

- DAZ Studio with DazScriptServer plugin running
- `dazpy` installed (`pip install dazpy`)
- Basic Python knowledge

## Dependencies

No additional dependencies beyond `dazpy`.

## Usage

```bash
python bvh/bvh_import/bvh_import.py
```

## How It Works

[See code comments in the script for detailed implementation walkthrough]

## Output

[Example-specific output description]

## Related Examples

- See main repository [README](../../README.md) for related examples
