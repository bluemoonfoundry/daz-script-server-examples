# daz-script-server Examples

Complete, categorized examples for [daz-script-server](https://github.com/bluemoonfoundry/daz-script-server) and the `dazpy` Python SDK.

[![License](https://img.shields.io/badge/license-AGPL%20v3-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8+-blue)](https://www.python.org/)

## Overview

This repository contains production-ready examples demonstrating how to control DAZ Studio remotely via the DazScriptServer plugin and dazpy Python SDK. Examples range from beginner tutorials to advanced production pipeline implementations.

## Prerequisites

- **DAZ Studio 4.5+** with [DazScriptServer plugin](https://github.com/bluemoonfoundry/daz-script-server) installed and running
- **Python 3.8+**
- **dazpy SDK:** `pip install dazpy`

## Quick Start

1. Start DAZ Studio with DazScriptServer plugin active (check status in DAZ Studio Panes → Daz Script Server)
2. Clone this repo: `git clone https://github.com/bluemoonfoundry/daz-script-server-examples.git`
3. Install dazpy: `pip install dazpy`
4. Navigate to an example: `cd fundamentals/raw_script/`
5. Install example dependencies (if requirements.txt exists): `pip install -r requirements.txt`
6. Run the example: `python raw_script.py`

## Examples by Category

| Example | Category | Level | Description | Dependencies |
|---------|----------|-------|-------------|--------------|
| [batch_operations](fundamentals/batch_operations/) | fundamentals | Intermediate | Shows how `Batch` bundles multiple independent DazScript reads into a single HTTP round-trip.  Readi... | None |
| [raw_script](fundamentals/raw_script/) | fundamentals | Beginner | Drop down to raw DazScript when the typed SDK doesn't expose what you need. Executes an IIFE against... | None |
| [scene_event_monitor](fundamentals/scene_event_monitor/) | fundamentals | Intermediate | Connects to the `GET /scene/events` SSE stream and reacts to live DAZ Studio activity — node additio... | None |
| [scene_introspection](fundamentals/scene_introspection/) | fundamentals | Beginner | Read-only dump of the entire scene hierarchy and world-space transforms. Output is JSON and can be p... | None |
| [scene_inventory](fundamentals/scene_inventory/) | fundamentals | Intermediate | Collects a structured report for every node in the scene — type, label, world position, visibility, ... | None |
| [scene_save_copy](fundamentals/scene_save_copy/) | fundamentals | Beginner | Saves a copy of the current scene to a new path — the Python equivalent of DAZ Studio's "Save a Copy... | None |
| [animation_frame_dump](character/animation_frame_dump/) | character | Intermediate | Scrubs through the timeline entirely inside DazScript — `Scene.setFrame()` advances the playhead ser... | None |
| [character_state](character/character_state/) | character | Intermediate | Saves a character's complete state — shape morphs, expression / FACS controls, and bone rotations — ... | None |
| [ik_bone_to_target](character/ik_bone_to_target/) | character | Advanced | Moves a specified bone on one figure toward a named target node using a simple iterative IK approach... | None |
| [pose_transfer](character/pose_transfer/) | character | Intermediate | Reads every bone's local Euler rotation from a source figure in one pass, then applies matching rota... | None |
| [animation_mixing](animation/animation_mixing/) | animation | Advanced | Treats captured animation files (from `animation_frame_dump.py`) as editable clips.  All operations ... | None |
| [keyframe_baking](animation/keyframe_baking/) | animation | Advanced | Reads the evaluated bone rotations and morph values of an animated figure at the current frame, then... | None |
| [pose_interpolation](animation/pose_interpolation/) | animation | Intermediate | Loads two state files produced by `character_state.py`, interpolates all bone rotations, morph value... | None |
| [body_measurements](geometry/body_measurements/) | geometry | Advanced | Computes practical body measurements for a selected figure by pulling the posed mesh into Python, sl... | Yes |
| [geometry_analysis](geometry/geometry_analysis/) | geometry | Intermediate | Retrieves a figure's mesh metadata, computes axis-aligned bounding boxes for both the rest and posed... | None |
| [scene_to_usd](export/scene_to_usd/) | export | Advanced | Interrogates the live DAZ Studio scene through the HTTP API and writes a Pixar USD file — without to... | Yes |
| [batch_render_morph_variations](rendering/batch_render_morph_variations/) | rendering | Intermediate | Renders a small, hardcoded matrix of expression morph combinations. Intended as a minimal starting t... | None |
| [capture_viewport](rendering/capture_viewport/) | rendering | Beginner | Captures the active DAZ Studio 3D viewport to a file without triggering an iRay render.  Three modes... | Yes |
| [comfyui_enhance](rendering/comfyui_enhance/) | rendering | Advanced | End-to-end pipeline that captures the active DAZ Studio viewport and submits the result to a local [... | Yes |
| [material_color_variations](rendering/material_color_variations/) | rendering | Intermediate | Renders a node's material surface in multiple diffuse colours.  The original colour is saved before ... | None |
| [multi_camera_render](rendering/multi_camera_render/) | rendering | Beginner | Iterates every camera in the scene (or a named subset) and renders from each one to `<out>/<camera_l... | None |
| [sprite_matrix](rendering/sprite_matrix/) | rendering | Advanced | Production pipeline for one sprite: given a scene with the sprite already loaded (a specific outfit,... | Yes |
| [turntable](rendering/turntable/) | rendering | Beginner | Rotates a figure around its local Y axis in equal steps and renders each frame to a numbered PNG.  E... | None |
| [vn_render_workflow](rendering/vn_render_workflow/) | rendering | Advanced | Four patterns for VN (visual novel) render pipelines.  VN production generates many renders of the s... | None |
| [dataset_generator](ml_data/dataset_generator/) | ml_data | Intermediate | Randomises a set of expression morphs on a Genesis 9 figure and renders each variation to a numbered... | None |
| [expression_transfer](ai_vision/expression_transfer/) | ai_vision | Advanced | Extracts a facial expression from a photo using MediaPipe FaceLandmarker, computes Action Unit (AU) ... | Yes |
| [webcam_expression_mirror](ai_vision/webcam_expression_mirror/) | ai_vision | Advanced | Captures frames from your webcam, runs MediaPipe FaceLandmarker on each frame, and streams the resul... | Yes |

## Categories

- **fundamentals/** — Core SDK patterns, scene inspection, batching, events
- **character/** — Pose, state, IK, animation dumps
- **animation/** — Keyframe baking, clip mixing, interpolation
- **geometry/** — Mesh analysis, body measurements
- **export/** — USD export, format conversion
- **rendering/** — Turntable, multi-cam, batch renders, VN workflows, external pipelines
- **ml_data/** — Dataset generation for ML training
- **ai_vision/** — MediaPipe expression transfer, webcam mirroring
- **bvh/** — BVH motion-capture import (in development)

## Skill Level Guide

- **Beginner:** Single HTTP call patterns, basic SDK usage, clear 1:1 mapping to DAZ Studio concepts. Start here if new to dazpy.
- **Intermediate:** Multiple API calls, state management, batch operations, moderate Python complexity. Requires understanding of DAZ Studio object model.
- **Advanced:** External integrations, complex pipelines, async patterns, production-scale techniques. Assumes Python proficiency and pipeline architecture experience.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on submitting new examples.

## License

AGPL v3 (matches parent repository)

## Links

- [daz-script-server repository](https://github.com/bluemoonfoundry/daz-script-server)
- [dazpy PyPI package](https://pypi.org/project/dazpy/)
- [DazScriptServer plugin documentation](https://github.com/bluemoonfoundry/daz-script-server#readme)
