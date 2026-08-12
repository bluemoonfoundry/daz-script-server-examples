# Geometry Analysis

**Level:** Intermediate
**Category:** Geometry

## Overview

Retrieves a figure's mesh metadata, computes axis-aligned bounding boxes for both the rest and posed mesh, lists face and material groups with their face counts, and demonstrates Python-side utilities such as quad-to-triangle conversion and `Vec3`-wrapped vertex access.

## What You'll Learn

- Practical implementation of geometry_analysis workflow
- SDK patterns for remote DAZ Studio control
- Production-ready code structure

**SDK features used:**
- `DazGeometry.mesh_info()`
- `DazGeometry.bounding_box()`
- `DazGeometry.bounding_box_posed()`
- `DazGeometry.face_group_faces()`
- `DazGeometry.material_group_faces()`
- `DazGeometry.face_vertex_indices_all()`
- `DazGeometry.vertex_positions_all()`
- `DazGeometry.triangulate()`
- `DazGeometry.as_vec3()`
- `BoundingBox.center`
- `BoundingBox.size`
- `BoundingBox.volume`
- `BoundingBox.contains()`
- `Vec3.distance()`

## Prerequisites

- DAZ Studio with DazScriptServer plugin running
- `dazpy` installed (`pip install dazpy`)
- Basic Python knowledge

## Dependencies

No additional dependencies beyond `dazpy`.

## Usage

```bash
python geometry/geometry_analysis.py --figure "Genesis 9"
python geometry/geometry_analysis.py --figure "Genesis 9" --groups
python geometry/geometry_analysis.py --figure "Genesis 9" --triangulate --out tris.json
```


### Arguments

| Argument | Default | Description |
|---|---|---|
| `--figure LABEL` | `Genesis 9` | Figure label |
| `--groups` | off | Print face group and material group details with face counts |
| `--triangulate` | off | Fetch all faces and export an all-triangle mesh |
| `--out FILE` | `triangles.json` | Output path when `--triangulate` is used |


## How It Works

[See code comments in the script for detailed implementation walkthrough]

## Output

[Example-specific output description]

## Related Examples

- See main repository [README](../../README.md) for related examples
