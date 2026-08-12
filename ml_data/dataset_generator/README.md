# Dataset Generator

**Level:** Intermediate
**Category:** ML/Data

## Overview

Randomises a set of expression morphs on a Genesis 9 figure and renders each variation to a numbered PNG.  A JSON sidecar is written alongside the images so the dataset is fully reproducible.  Suitable as a starting point for generating LoRA training data.

## What You'll Learn

- Practical implementation of dataset_generator workflow
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
python ml_data/dataset_generator.py
python ml_data/dataset_generator.py --count 100 --out C:/dataset --size 512
```


### Arguments

| Argument | Default | Description |
|---|---|---|
| `--count N` | `10` | Number of randomised renders to produce |
| `--out DIR` | `y:/tmp/` | Output directory |


## How It Works

[See code comments in the script for detailed implementation walkthrough]

## Output

[Example-specific output description]

## Related Examples

- See main repository [README](../../README.md) for related examples
