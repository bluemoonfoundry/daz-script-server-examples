# BVH Discover

**Level:** Intermediate
**Category:** BVH

## Overview

Connects to a running DAZ Studio instance, reads every bone name from a specified figure, and emits a ready-to-paste CANONICAL_TO_DAZ block for bvh_bone_maps.py. Lets you verify or extend bone-name mapping tables for any DAZ generation without manually browsing the rig.

## What You'll Learn

- Practical implementation of bvh_discover workflow
- SDK patterns for remote DAZ Studio control
- Production-ready code structure

**SDK features used:**
- Bone querying and enumeration
- DAZ generation auto-detection
- Bone name validation

## Prerequisites

- DAZ Studio with DazScriptServer plugin running
- `dazpy` installed (`pip install dazpy`)
- Basic Python knowledge

## Dependencies

No additional dependencies beyond `dazpy`.

## Usage

```bash
python bvh/bvh_discover/bvh_discover.py
```

## How It Works

[See code comments in the script for detailed implementation walkthrough]

## Output

[Example-specific output description]

## Related Examples

- See main repository [README](../../README.md) for related examples
