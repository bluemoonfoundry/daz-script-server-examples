# BVH Bone Maps

**Level:** Advanced
**Category:** BVH

## Overview

Provides bone-name translation tables for retargeting mocap data from standard BVH conventions to DAZ Studio's per-generation internal bone names. Demonstrates a two-tier translation architecture mapping BVH convention names to canonical keys, then to DAZ internal bone names.

## What You'll Learn

- Practical implementation of bvh_bone_maps workflow
- SDK patterns for remote DAZ Studio control
- Production-ready code structure

**SDK features used:**
- Bone name mapping and translation
- Multi-generation support (Genesis 1-9)
- Mocap retargeting conventions

## Prerequisites

- DAZ Studio with DazScriptServer plugin running
- `dazpy` installed (`pip install dazpy`)
- Basic Python knowledge

## Dependencies

No additional dependencies beyond `dazpy`.

## Usage

```bash
python bvh/bvh_bone_maps/bvh_bone_maps.py
```

## How It Works

[See code comments in the script for detailed implementation walkthrough]

## Output

[Example-specific output description]

## Related Examples

- See main repository [README](../../README.md) for related examples
