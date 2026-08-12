# Scene Introspection

**Level:** Beginner
**Category:** Fundamentals

## Overview

Read-only dump of the entire scene hierarchy and world-space transforms. Output is JSON and can be piped to `jq` or redirected to a file.

## What You'll Learn

- Practical implementation of scene_introspection workflow
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
python fundamentals/scene_introspection.py
python fundamentals/scene_introspection.py | jq '.tree[0]'
```

## How It Works

[See code comments in the script for detailed implementation walkthrough]

## Output

[Example-specific output description]

## Related Examples

- See main repository [README](../../README.md) for related examples
