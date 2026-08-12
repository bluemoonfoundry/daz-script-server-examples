# Batch Operations

**Level:** Intermediate
**Category:** Fundamentals

## Overview

Shows how `Batch` bundles multiple independent DazScript reads into a single HTTP round-trip.  Reading label, bone count, and morph count for N figures normally costs 3N calls; with `Batch` it costs 1.

## What You'll Learn

- Practical implementation of batch_operations workflow
- SDK patterns for remote DAZ Studio control
- Production-ready code structure

**SDK features used:**
- `Batch`
- `Batch.add()`
- `BatchFuture.value`
- `DazScene.skeletons()`

## Prerequisites

- DAZ Studio with DazScriptServer plugin running
- `dazpy` installed (`pip install dazpy`)
- Basic Python knowledge

## Dependencies

No additional dependencies beyond `dazpy`.

## Usage

```bash
python fundamentals/batch_operations.py
python fundamentals/batch_operations.py --compare
```


### Arguments

| Argument | Default | Description |
|---|---|---|
| `--compare` | off | Also run the per-call version and print the call-count comparison |


## How It Works

[See code comments in the script for detailed implementation walkthrough]

## Output

[Example-specific output description]

## Related Examples

- See main repository [README](../../README.md) for related examples
