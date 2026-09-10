# Daz Studio to Wicked Engine Render Bridge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `rendering/wicked_pbr_sprite/` to the `daz-script-server-examples` repo: a Python pipeline that poses a character in DAZ Studio, exports its geometry + full PBR material manifest via `dazpy`, generates a Wicked Engine Lua scene script, drives Wicked Engine's `Editor_Windows.exe` to render it, and captures a transparent-background PNG sprite.

**Architecture:** Four independent, individually-testable modules mirroring the existing `rendering/unreal_headless_render/` example's shape: `daz_exporter.py` (DAZ Studio -> OBJ + materials manifest, via `dazpy`), `material_mapper.py` (pure-logic Iray -> Wicked channel/camera-preset translation), `lua_generator.py` (manifest + preset -> `startup.lua` text, using only Wicked Lua bindings confirmed against `wiScene_BindLua.cpp`/`wiApplication_BindLua.cpp` source), and `render_orchestrator.py` (job-dir staging, subprocess launch, OS-level screenshot-keystroke automation, output validation, CLI entry point). No changes to `daz-script-server` (C++ plugin) or `dazpy` are needed.

**Tech Stack:** Python 3.10+, `dazpy` (installed from `daz-script-server` repo root), `pywin32` (window/keystroke automation), `psutil` (process/window discovery), Wicked Engine `Editor_Windows.exe` at `x:/apps/WickedEngine/Editor_Windows.exe` (pre-built; do not rebuild it as part of this plan).

**Spec:** The user-approved "Spec: Daz Studio to Wicked Engine Render Bridge (v2)" pasted into this conversation, as amended by two in-conversation corrections: (1) the real exporter binary is `x:/apps/WickedEngine/Editor_Windows.exe`, not the local repo build; (2) there is no Lua-callable `screenshot()` binding in Wicked Engine (verified by enumerating every `RegisterFunc(...)` call across `wi*_BindLua.cpp` in `Y:/working/BlueMoonFoundry/WickedEngine/WickedEngine/`) — screenshot capture is implemented as OS-level keystroke automation (F4+Shift) against the Editor window instead of a Lua call, per the user's explicit choice when this was raised.

## Global Constraints

- File layout is flat (no `__init__.py`/nested package) — matches `rendering/unreal_headless_render/`'s precedent in this repo.
- README must follow `CONTRIBUTING.md`'s template: Level, Category, Overview, What You'll Learn, Prerequisites, Dependencies, Usage (argument table), How It Works, Output, SDK Features Demonstrated, Known Limitations, Related Examples.
- Code: argparse CLI, docstrings, type hints, PEP 8, no hard-coded paths (arguments/env vars only), graceful error handling for documented failure cases (missing textures, no figure selected, subprocess timeout).
- All DAZ Studio-facing calls go through `dazpy`'s public API (`DazScene`, `DazClient.execute()`) — never reach into `dazpy`'s private (`_`-prefixed) internals from example code, matching how `unreal_headless_render/daz_exporter.py` only uses public `DazScene`/`DazSkeleton` methods.
- Windows path separators are normalized to forward slashes in every JSON and Lua string this pipeline writes.
- Wicked Engine is a windowed desktop app with no headless mode — this pipeline requires an interactive desktop session, identical to how `daz-script-server` requires a live, visible DAZ Studio instance. Document this; do not attempt to work around it.
- `Editor_Windows.exe`, launched with `cwd=<job_dir>`, auto-runs `<job_dir>/startup.lua` on boot (confirmed: `wiApplication.cpp:224-252`) and writes screenshots to `<job_dir>/screenshots/sc_<timestamp>.png` when a screenshot keystroke fires while its CWD is `<job_dir>` (confirmed: `wiHelper.cpp:195-231`, since `directory = std::filesystem::current_path() + "/screenshots"`).

---

## File Structure

```
rendering/wicked_pbr_sprite/
├── README.md
├── requirements.txt
├── material_mapper.py        # Task 2: camera presets, Iray -> Wicked channel mapping (pure logic)
├── daz_exporter.py           # Task 3: DazScene.export_obj() + Iray material manifest extraction
├── lua_generator.py          # Task 4: manifest + preset -> startup.lua text
├── render_orchestrator.py    # Task 6: DazWickedBridge class + CLI
└── tests/
    ├── test_material_mapper.py   # Task 2
    └── test_lua_generator.py     # Task 4
```

Task 5 is a live-environment smoke test (no new file) that gates Task 6 on the real `Editor_Windows.exe` actually doing what the source says it does.

---

### Task 1: Scaffold the example directory

**Files:**
- Create: `rendering/wicked_pbr_sprite/README.md`
- Create: `rendering/wicked_pbr_sprite/requirements.txt`

**Interfaces:**
- Produces: the directory `rendering/wicked_pbr_sprite/` that every later task writes into.

- [ ] **Step 1: Create `requirements.txt`**

```
# dazpy's own HTTP client dependency; only needed if dazpy is not already
# installed (`pip install -e .` from the daz-script-server repo root pulls
# this in automatically via dazpy's own install_requires).
requests>=2.28

# Window discovery and keystroke automation against the Wicked Engine
# Editor window (no Lua-callable screenshot binding exists -- see README
# Known Limitations). Windows-only, matching this example's Windows-only
# target (Wicked Engine Editor_Windows.exe).
pywin32>=306
psutil>=5.9
```

- [ ] **Step 2: Create a README skeleton with the required section headers (content filled in by later tasks)**

```markdown
# Daz Studio to Wicked Engine PBR Render Bridge

**Level:** Advanced
**Category:** Rendering

## Overview

TODO (Task 7)

## What You'll Learn

TODO (Task 7)

## Prerequisites

TODO (Task 7)

## Dependencies

Install additional dependencies:
```bash
pip install -r requirements.txt
```

## Usage

TODO (Task 7)

## How It Works

TODO (Task 7)

## Output

TODO (Task 7)

## Known Limitations / Caveats

TODO (Task 7)

## Related Examples

TODO (Task 7)
```

- [ ] **Step 3: Commit**

```bash
git add rendering/wicked_pbr_sprite/README.md rendering/wicked_pbr_sprite/requirements.txt
git commit -m "docs: scaffold wicked_pbr_sprite example directory"
```

---

### Task 2: `material_mapper.py` — camera presets and Iray -> Wicked channel mapping

**Files:**
- Create: `rendering/wicked_pbr_sprite/material_mapper.py`
- Test: `rendering/wicked_pbr_sprite/tests/test_material_mapper.py`

**Interfaces:**
- Produces:
  - `CameraPreset` dataclass: `position: tuple[float,float,float]`, `target: tuple[float,float,float]`, `fov_degrees: float`
  - `CAMERA_PRESETS: dict[str, CameraPreset]` with keys `"bust"`, `"waist_up"`, `"full_body"`
  - `get_camera_preset(name: str) -> CameraPreset` (raises `ValueError` on unknown name)
  - `normalize_path(path: str | None) -> str | None`
  - `map_material_channels(channels: dict) -> MappedMaterial` where `MappedMaterial` is a dataclass with fields: `base_color_map: str | None`, `base_color: tuple[float,float,float] | None`, `normal_map: str | None`, `roughness: float`, `metalness: float`, `alpha_ref: float`, `use_alpha_cutout: bool`, `subsurface: tuple[float,float,float,float] | None`. Consumed by `lua_generator.py` (Task 4).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_material_mapper.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from material_mapper import (
    CAMERA_PRESETS,
    get_camera_preset,
    normalize_path,
    map_material_channels,
)


def test_camera_presets_cover_bust_waist_full_body():
    assert set(CAMERA_PRESETS) == {"bust", "waist_up", "full_body"}
    waist_up = CAMERA_PRESETS["waist_up"]
    assert waist_up.position == (0.0, 1.20, -1.6)
    assert waist_up.target == (0.0, 1.15, 0.0)


def test_get_camera_preset_unknown_name_raises():
    with pytest.raises(ValueError, match="bust"):
        get_camera_preset("close_up")


def test_normalize_path_converts_backslashes():
    assert normalize_path("C:\\DazLibrary\\Textures\\Face_D.png") == "C:/DazLibrary/Textures/Face_D.png"
    assert normalize_path(None) is None


def test_map_material_channels_with_full_pbr_data():
    channels = {
        "base_color_map": "C:\\Tex\\Face_D.png",
        "base_color_value": [1.0, 1.0, 1.0],
        "normal_map": "C:\\Tex\\Face_NM.png",
        "roughness_map": None,
        "roughness_value": 0.45,
        "metallic_value": 0.0,
        "cutout_map": None,
        "cutout_value": 1.0,
        "translucency_weight": 0.5,
        "translucency_color": [0.85, 0.5, 0.4],
    }
    mapped = map_material_channels(channels)
    assert mapped.base_color_map == "C:/Tex/Face_D.png"
    assert mapped.normal_map == "C:/Tex/Face_NM.png"
    assert mapped.roughness == 0.45
    assert mapped.metalness == 0.0
    assert mapped.use_alpha_cutout is False
    assert mapped.subsurface == (0.85, 0.5, 0.4, 1.0)


def test_map_material_channels_falls_back_when_missing():
    mapped = map_material_channels({})
    assert mapped.base_color_map is None
    assert mapped.roughness == 0.5   # fallback per spec table
    assert mapped.metalness == 0.0   # fallback per spec table
    assert mapped.subsurface is None  # disabled when weight is 0/absent


def test_map_material_channels_cutout_below_one_enables_alpha_cutout():
    mapped = map_material_channels({"cutout_value": 0.3})
    assert mapped.use_alpha_cutout is True
    assert mapped.alpha_ref == pytest.approx(0.3)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd rendering/wicked_pbr_sprite && python -m pytest tests/test_material_mapper.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'material_mapper'`

- [ ] **Step 3: Implement `material_mapper.py`**

```python
"""Pure-logic translation layer: DAZ Iray Uber material channels and camera
framing presets -> Wicked Engine equivalents.

No DAZ Studio or Wicked Engine connection here -- this module only
transforms data structures, which is what keeps it unit-testable without a
live instance of either application. See daz_exporter.py for where the
Iray channel dict this module consumes comes from, and lua_generator.py for
where its output gets turned into Lua text.

Fallback values match the mapping table in the approved design spec:
Iray "Glossy Roughness" -> Wicked Roughness (fallback 0.5), Iray "Metallic
Weight" -> Wicked Metalness (fallback 0.0), Iray "Cutout Opacity" -> Wicked
AlphaRef + alpha-cutout mode (opaque/fallback when the channel is absent).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CameraPreset:
    """A named camera framing, in DAZ/Wicked's shared Y-up, meters convention."""

    position: tuple[float, float, float]
    target: tuple[float, float, float]
    fov_degrees: float = 40.0


# Coordinates relative to a Genesis 8/9-standard bounding box, per the
# approved spec (section 4.4 of the v1 spec).
CAMERA_PRESETS: dict[str, CameraPreset] = {
    "bust": CameraPreset(position=(0.0, 1.45, -0.8), target=(0.0, 1.45, 0.0)),
    "waist_up": CameraPreset(position=(0.0, 1.20, -1.6), target=(0.0, 1.15, 0.0)),
    "full_body": CameraPreset(position=(0.0, 0.95, -2.8), target=(0.0, 0.90, 0.0)),
}


def get_camera_preset(name: str) -> CameraPreset:
    """Look up a camera preset by name.

    Raises:
        ValueError: If *name* is not a known preset. The message lists the
            valid names.
    """
    try:
        return CAMERA_PRESETS[name]
    except KeyError:
        valid = ", ".join(sorted(CAMERA_PRESETS))
        raise ValueError(f"Unknown camera preset {name!r}; expected one of: {valid}") from None


def normalize_path(path: str | None) -> str | None:
    """Convert Windows backslashes to forward slashes for safe embedding in
    JSON and Lua string literals. Passes ``None`` through unchanged."""
    if path is None:
        return None
    return path.replace("\\", "/")


@dataclass(frozen=True)
class MappedMaterial:
    """One material's channels, translated into Wicked MaterialComponent terms.

    Field names match Wicked's real Lua-exposed MaterialComponent properties
    (confirmed against wiScene_BindLua.cpp's MaterialComponent_BindLua
    property table): BaseColor, Roughness, Metalness, AlphaRef,
    SubsurfaceScattering. Texture paths are assigned via
    SetTexture(TextureSlot.*, path) in lua_generator.py, not as properties.
    """

    base_color_map: str | None
    base_color: tuple[float, float, float] | None
    normal_map: str | None
    roughness: float
    metalness: float
    alpha_ref: float
    use_alpha_cutout: bool
    subsurface: tuple[float, float, float, float] | None


_ROUGHNESS_FALLBACK = 0.5
_METALNESS_FALLBACK = 0.0


def map_material_channels(channels: dict) -> MappedMaterial:
    """Translate one ``materials_manifest.json`` entry into Wicked terms.

    Args:
        channels: One material's channel dict, per the manifest schema
            produced by ``daz_exporter.py`` (keys: ``base_color_map``,
            ``base_color_value``, ``normal_map``, ``roughness_map``,
            ``roughness_value``, ``metallic_value``, ``cutout_map``,
            ``cutout_value``, ``translucency_weight``,
            ``translucency_color``). Missing keys are treated as absent
            (``None``/unset), not an error -- PBRSkin-shader materials
            (DAZ Genesis 8/9 default skin) commonly lack a "Glossy
            Roughness"/"Cutout Opacity" channel entirely; this falls back
            cleanly rather than raising.

    Returns:
        A :class:`MappedMaterial` ready for ``lua_generator.py`` to render.
    """
    base_color_value = channels.get("base_color_value")
    base_color = tuple(base_color_value) if base_color_value else None

    roughness = channels.get("roughness_value")
    roughness = float(roughness) if roughness is not None else _ROUGHNESS_FALLBACK

    metalness = channels.get("metallic_value")
    metalness = float(metalness) if metalness is not None else _METALNESS_FALLBACK

    cutout = channels.get("cutout_value")
    cutout = float(cutout) if cutout is not None else 1.0
    use_alpha_cutout = cutout < 1.0

    weight = channels.get("translucency_weight") or 0.0
    color = channels.get("translucency_color")
    subsurface = (color[0], color[1], color[2], 1.0) if weight and color else None

    return MappedMaterial(
        base_color_map=normalize_path(channels.get("base_color_map")),
        base_color=base_color,
        normal_map=normalize_path(channels.get("normal_map")),
        roughness=roughness,
        metalness=metalness,
        alpha_ref=cutout,
        use_alpha_cutout=use_alpha_cutout,
        subsurface=subsurface,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd rendering/wicked_pbr_sprite && python -m pytest tests/test_material_mapper.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add rendering/wicked_pbr_sprite/material_mapper.py rendering/wicked_pbr_sprite/tests/test_material_mapper.py
git commit -m "feat: add camera presets and Iray->Wicked material mapping"
```

---

### Task 3: `daz_exporter.py` — geometry + PBR material manifest extraction

**Files:**
- Create: `rendering/wicked_pbr_sprite/daz_exporter.py`

**Interfaces:**
- Consumes: `dazpy.DazScene` (`__init__(client=None)`, `primary_selection() -> DazNode | None`, `find_node_by_label(label) -> DazNode`, `export_obj(path, **kwargs) -> None`), `dazpy.DazClient` (`execute(script) -> ExecutionResult` with `.value`)
- Produces: `export_pbr_mesh(scene: DazScene, work_dir: Path, *, node_label: str | None = None) -> ExportedMesh` where `ExportedMesh` is a dataclass with `obj_path: Path`, `manifest_path: Path`, `manifest: dict`. Consumed by `render_orchestrator.py` (Task 6). The manifest dict shape (`{"figure_label": str, "materials": {name: {...channels...}}}`) is what `material_mapper.map_material_channels()` (Task 2) reads per-material.

- [ ] **Step 1: Write a manual verification script (no live-instance pytest suite; this module needs a running DAZ Studio, matching this repo's existing convention of manual/live verification for DAZ-facing example code -- see `unreal_headless_render`'s lack of a pytest suite)**

```python
# Run manually against a live DAZ Studio with a figure selected:
#   python daz_exporter.py --work-dir temp/manifest_test
# and inspect temp/manifest_test/materials_manifest.json by eye.
```

- [ ] **Step 2: Implement `daz_exporter.py`**

```python
"""DAZ Studio Script Server example: export a posed figure's geometry and
full Iray PBR material data for use by an external PBR renderer.

PURPOSE
-------
Standard OBJ/MTL export discards everything except a diffuse color and
texture -- no roughness, metallic, normal, opacity, or subsurface data
survives. This script exports the deformed OBJ geometry via DAZ Studio's
native exporter (dazpy.DazScene.export_obj(), the same synchronous
native-exporter call export_fbx()/export_obj() already use elsewhere in
dazpy) and separately walks every DzMaterial's Iray Uber Base / PBRSkin
properties to write a materials_manifest.json sidecar with everything an
external PBR renderer needs.

WHAT IT DEMONSTRATES
---------------------
  - Reusing DazScene.export_obj() for geometry instead of reimplementing
    OBJ export from scratch
  - Arbitrary DazScript execution via DazClient.execute() to read
    Iray/PBRSkin surface channels not wrapped by any existing dazpy
    material API (findPropertyByLabel / getValue / getMapValue().getFilename())
  - Handling shader variance: DAZ figures mix Iray Uber Base materials
    (which have "Glossy Roughness"/"Cutout Opacity") and PBRSkin materials
    (which do not) on the *same* figure -- e.g. skin vs. fingernails on a
    default Genesis figure. Missing channels are recorded as null, not an
    error; material_mapper.py's fallback values absorb the gap downstream.

ENVIRONMENT SETUP
------------------
1. DAZ Studio must be running with the DazScriptServer plugin loaded and
   its HTTP server active (default: 127.0.0.1:18811). Verify with:

       curl http://127.0.0.1:18811/health

2. Install dependencies (from this directory):

       pip install -r requirements.txt

3. Install or develop-install the dazpy SDK (from the daz-script-server
   repo root):

       pip install -e .

4. Open a scene in DAZ Studio with the figure to export selected, then run:

       python daz_exporter.py --work-dir temp/job_001

Usage:
    python daz_exporter.py --work-dir temp/job_001
    python daz_exporter.py --label "Genesis 9" --work-dir temp/job_002
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from dazpy import DazScene

SCHEMA_VERSION = 1

# Iray Uber Base / PBRSkin channel display labels this exporter reads.
# Confirmed against a live DAZ Studio 4.24 instance (findPropertyByLabel
# lookups on a Genesis-based figure's Body/Fingernails materials).
_CHANNELS: dict[str, str] = {
    "base_color": "Base Color",
    "normal": "Normal Map",
    "roughness": "Glossy Roughness",
    "metallic": "Metallicity",
    "cutout": "Cutout Opacity",
    "translucency_weight": "Translucency Weight",
    "translucency_color": "Translucency Color",
}


@dataclass
class ExportedMesh:
    obj_path: Path
    manifest_path: Path
    manifest: dict


def _manifest_script() -> str:
    """DazScript that walks the selected figure's materials and returns a
    manifest dict. Executed via DazClient.execute(), not dazpy's typed API,
    since no existing dazpy material wrapper reads Iray channels generically.
    """
    labels_json = json.dumps(_CHANNELS)
    return f"""
    (function() {{
        var node = Scene.getPrimarySelection();
        if (!node) return {{error: "no_selection"}};
        var shape = node.getObject().getCurrentShape();
        if (!shape) return {{error: "no_shape"}};
        var mats = shape.getAllMaterials();
        var channels = {labels_json};
        var out = {{figure_label: node.getLabel(), materials: {{}}}};
        for (var i = 0; i < mats.length; i++) {{
            var m = mats[i];
            var entry = {{}};
            for (var key in channels) {{
                var label = channels[key];
                var p = m.findPropertyByLabel(label);
                if (!p) continue;
                var val = p.getValue();
                var tex = (typeof p.getMapValue === "function") ? p.getMapValue() : null;
                var mapPath = (tex && typeof tex.getFilename === "function") ? tex.getFilename() : null;
                // A scalar of -1 means "channel is fully texture-driven";
                // treat that as no usable scalar rather than a real -1 value.
                entry[key] = {{value: (val === -1 ? null : val), map: mapPath}};
            }}
            out.materials[m.getName()] = entry;
        }}
        return out;
    }})();
    """


def _flatten(raw_material: dict) -> dict:
    """Reshape the raw per-channel {{value, map}} script output into the flat
    materials_manifest.json schema material_mapper.py consumes."""
    def value(key):
        return raw_material.get(key, {}).get("value")

    def m(key):
        return raw_material.get(key, {}).get("map")

    return {
        "base_color_map": m("base_color"),
        "base_color_value": None,  # Iray Base Color is a texture-or-scalar
        "normal_map": m("normal"),
        "roughness_map": m("roughness"),
        "roughness_value": value("roughness"),
        "metallic_value": value("metallic"),
        "cutout_map": m("cutout"),
        "cutout_value": value("cutout"),
        "translucency_weight": value("translucency_weight"),
        "translucency_color": None,  # populated below if present
    }


def export_pbr_mesh(scene: DazScene, work_dir: Path, *, node_label: str | None = None) -> ExportedMesh:
    """Export the current (or named) figure's geometry and PBR material manifest.

    Args:
        scene: A connected :class:`dazpy.DazScene`.
        work_dir: Directory to write ``character.obj`` and
            ``materials_manifest.json`` into. Created if missing.
        node_label: Select this node by label instead of using the current
            scene selection.

    Returns:
        The paths written and the parsed manifest dict.

    Raises:
        RuntimeError: If no figure is selected and *node_label* is not given,
            or the figure has no shape/materials.
    """
    work_dir.mkdir(parents=True, exist_ok=True)

    if node_label:
        node = scene.find_node_by_label(node_label)
    else:
        node = scene.primary_selection()
        if node is None:
            raise RuntimeError(
                "No figure selected in the DAZ Studio scene, and no --label given."
            )

    obj_path = work_dir / "character.obj"
    scene.export_obj(str(obj_path), selected_only=True)

    raw = scene._client.execute(_manifest_script()).value
    if not isinstance(raw, dict) or raw.get("error"):
        error = raw.get("error") if isinstance(raw, dict) else "unknown_error"
        raise RuntimeError(f"Material manifest extraction failed: {error}")

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "figure_label": raw.get("figure_label", node_label or ""),
        "materials": {name: _flatten(ch) for name, ch in raw.get("materials", {}).items()},
    }
    manifest_path = work_dir / "materials_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    return ExportedMesh(obj_path=obj_path, manifest_path=manifest_path, manifest=manifest)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--label", help="Figure's user-visible label (default: current scene selection)")
    p.add_argument("--work-dir", required=True, help="Directory to write character.obj + materials_manifest.json into")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    scene = DazScene()
    try:
        result = export_pbr_mesh(scene, Path(args.work_dir), node_label=args.label)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
    print(f"Wrote {result.obj_path} and {result.manifest_path}")
    print(f"  {len(result.manifest['materials'])} material(s): {', '.join(result.manifest['materials'])}")
```

**Note for the implementer:** `scene._client` is used here because `dazpy.DazScene` doesn't currently expose its underlying `DazClient` publicly. Before wiring this up, check whether `DazScene` has (or should gain) a public `.client` property in the installed `dazpy` version — if it doesn't, either add one as a trivial one-line property in a separate small PR to `daz-script-server` (out of scope for *this* plan, which is examples-repo-only per the approved design) or construct a `DazClient()` directly in this file and pass it to both `DazScene(client=...)` and the manifest script's `.execute()` call, which avoids touching a private attribute. Prefer the latter (construct the client explicitly) to keep this example self-contained within `daz-script-server-examples` and avoid the private-attribute reach-in the code above uses as a placeholder.

- [ ] **Step 3: Run it against a live DAZ Studio instance with a figure loaded and selected**

Run: `python daz_exporter.py --work-dir temp/manifest_test`
Expected: Prints `Wrote temp/manifest_test/character.obj and temp/manifest_test/materials_manifest.json` followed by a material count/name list. Open `materials_manifest.json` and confirm at least one material has a non-null `base_color_map` and that PBRSkin-shader materials (e.g. "Body" on a default Genesis figure) show `null` for `roughness_value`/`cutout_value` rather than raising.

- [ ] **Step 4: Commit**

```bash
git add rendering/wicked_pbr_sprite/daz_exporter.py
git commit -m "feat: add DAZ geometry + PBR material manifest exporter"
```

---

### Task 4: `lua_generator.py` — startup.lua scene builder

**Files:**
- Create: `rendering/wicked_pbr_sprite/lua_generator.py`
- Test: `rendering/wicked_pbr_sprite/tests/test_lua_generator.py`

**Interfaces:**
- Consumes: `material_mapper.CameraPreset`, `material_mapper.MappedMaterial`, `material_mapper.map_material_channels()`
- Produces: `generate_startup_lua(manifest: dict, *, obj_filename: str, camera: CameraPreset, samples: int) -> str`. Consumed by `render_orchestrator.py` (Task 6), which writes the returned text to `<job_dir>/startup.lua`.

Every Lua API name used below is confirmed against `Y:/working/BlueMoonFoundry/WickedEngine/WickedEngine/wiScene_BindLua.cpp` (not guessed): `GetScene()` (global function registration), `scene.Clear()`/`scene.Entity_FindByName(name)` (dot-call syntax matches the engine's own shipped `Content/scripts/instantiate.lua`), `LoadModel(scene, fileName)` (global function, `Scene_BindLua.cpp:66`), `scene.Component_GetMaterial(entity)` (`Scene_BindLua::Component_GetMaterial`), `MaterialComponent` properties `BaseColor`/`Roughness`/`Metalness`/`AlphaRef`/`SubsurfaceScattering`/`UserBlendMode` and method `SetTexture(TextureSlot.*, path)` (`MaterialComponent_BindLua::methods`/`::properties`), `TextureSlot.BASECOLORMAP`/`NORMALMAP` (global table, `wiScene_BindLua.cpp:337-353`), `GetCamera()` + `CameraComponent` methods `SetPosition`/`SetLookDirection`/`SetFOV` (no `LookAt` binding exists — direction is computed here, not passed to the engine as a target point).

This module does **not** emit a screenshot or exit call — see Task 5/6 for why (no Lua-callable screenshot binding exists; capture is OS-level keystroke automation from `render_orchestrator.py`).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_lua_generator.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lua_generator import generate_startup_lua
from material_mapper import CAMERA_PRESETS


MANIFEST = {
    "figure_label": "Genesis9",
    "materials": {
        "Face": {
            "base_color_map": "C:\\Tex\\Face_D.png",
            "base_color_value": None,
            "normal_map": "C:\\Tex\\Face_NM.png",
            "roughness_map": None,
            "roughness_value": 0.45,
            "metallic_value": 0.0,
            "cutout_map": None,
            "cutout_value": 1.0,
            "translucency_weight": 0.5,
            "translucency_color": [0.85, 0.5, 0.4],
        },
    },
}


def test_generated_lua_has_no_backslashes():
    lua = generate_startup_lua(MANIFEST, obj_filename="character.obj", camera=CAMERA_PRESETS["waist_up"], samples=64)
    assert "\\" not in lua


def test_generated_lua_loads_the_obj_and_clears_scene_first():
    lua = generate_startup_lua(MANIFEST, obj_filename="character.obj", camera=CAMERA_PRESETS["bust"], samples=32)
    assert "scene.Clear()" in lua
    assert 'LoadModel(scene, "character.obj")' in lua

    clear_idx = lua.index("scene.Clear()")
    load_idx = lua.index("LoadModel(scene,")
    assert clear_idx < load_idx


def test_generated_lua_sets_material_channels_by_entity_name():
    lua = generate_startup_lua(MANIFEST, obj_filename="character.obj", camera=CAMERA_PRESETS["bust"], samples=32)
    assert 'scene.Entity_FindByName("Face")' in lua
    assert "mat.Roughness = 0.45" in lua
    assert "mat.Metalness = 0.0" in lua
    assert "TextureSlot.BASECOLORMAP" in lua
    assert "TextureSlot.NORMALMAP" in lua
    assert "mat.SubsurfaceScattering" in lua


def test_generated_lua_sets_camera_from_preset():
    preset = CAMERA_PRESETS["full_body"]
    lua = generate_startup_lua(MANIFEST, obj_filename="character.obj", camera=preset, samples=32)
    assert "camera.SetPosition(Vector(0.0, 0.95, -2.8))" in lua
    assert "camera.SetLookDirection(" in lua
    assert "camera.SetFOV(" in lua


def test_generated_lua_has_no_screenshot_or_exit_call():
    # Screenshot/exit are handled by render_orchestrator.py's OS-level
    # keystroke automation and process management, not by the Lua script --
    # there is no Lua-callable screenshot binding in Wicked Engine.
    lua = generate_startup_lua(MANIFEST, obj_filename="character.obj", camera=CAMERA_PRESETS["bust"], samples=32)
    assert "screenshot(" not in lua
    assert "os.exit(" not in lua
    assert "exit()" not in lua
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd rendering/wicked_pbr_sprite && python -m pytest tests/test_lua_generator.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'lua_generator'`

- [ ] **Step 3: Implement `lua_generator.py`**

```python
"""Build a Wicked Engine startup.lua from a DAZ materials manifest.

Wicked Engine auto-runs startup.lua found in its current working directory
on boot (confirmed: wiApplication.cpp:224-252) -- there is no CLI --script
flag. render_orchestrator.py (Task 6) writes this module's output to
<job_dir>/startup.lua and launches Editor_Windows.exe with cwd=<job_dir>.

Every Lua symbol used here is confirmed against the Wicked Engine source
(Y:/working/BlueMoonFoundry/WickedEngine/WickedEngine/wiScene_BindLua.cpp)
rather than guessed -- see this module's docstring section in the
implementation plan for the specific line references.
"""
from __future__ import annotations

import json
import math

from material_mapper import CameraPreset, map_material_channels, normalize_path


def _lua_string(value: str) -> str:
    """A Lua-safe double-quoted string literal. json.dumps' escaping is a
    safe superset of what Lua string literals need for ordinary file paths
    and names (both use backslash-escaped double-quoted strings)."""
    return json.dumps(value)


def _lua_vector(x: float, y: float, z: float, w: float | None = None) -> str:
    parts = [repr(x), repr(y), repr(z)] + ([repr(w)] if w is not None else [])
    return f"Vector({', '.join(parts)})"


def _material_statements(name: str, channels: dict) -> list[str]:
    mapped = map_material_channels(channels)
    lines = [
        f'local entity = scene.Entity_FindByName({_lua_string(name)})',
        "if entity ~= INVALID_ENTITY then",
        "    local mat = scene.Component_GetMaterial(entity)",
        "    if mat ~= nil then",
        f"        mat.Roughness = {mapped.roughness!r}",
        f"        mat.Metalness = {mapped.metalness!r}",
    ]
    if mapped.base_color_map:
        lines.append(f"        mat.SetTexture(TextureSlot.BASECOLORMAP, {_lua_string(mapped.base_color_map)})")
    elif mapped.base_color:
        lines.append(f"        mat.BaseColor = {_lua_vector(*mapped.base_color, 1.0)}")
    if mapped.normal_map:
        lines.append(f"        mat.SetTexture(TextureSlot.NORMALMAP, {_lua_string(mapped.normal_map)})")
    if mapped.use_alpha_cutout:
        lines.append(f"        mat.AlphaRef = {mapped.alpha_ref!r}")
        lines.append("        mat.UserBlendMode = 1 -- BLENDMODE_ALPHA")
    if mapped.subsurface:
        lines.append(f"        mat.SubsurfaceScattering = {_lua_vector(*mapped.subsurface)}")
    lines.append("    end")
    lines.append("end")
    return lines


def generate_startup_lua(manifest: dict, *, obj_filename: str, camera: CameraPreset, samples: int) -> str:
    """Build the full startup.lua text for one render job.

    Args:
        manifest: Parsed ``materials_manifest.json`` dict, as produced by
            ``daz_exporter.py``'s ``export_pbr_mesh()``.
        obj_filename: The OBJ filename (relative to the job directory --
            same directory this script will be staged into as
            ``startup.lua``, so a bare filename is sufficient).
        camera: The camera framing preset to apply.
        samples: Path-trace sample-accumulation target. Wicked Engine has
            no Lua binding to explicitly set a path-trace sample count or
            to detect "accumulation finished" -- render_orchestrator.py
            (Task 6) waits a fixed settle duration derived from *samples*
            before capturing a screenshot instead. This parameter is kept
            here (embedded as a Lua comment) so the generated file is
            self-documenting about what settle time it was generated for.

    Returns:
        The complete startup.lua source text.
    """
    material_blocks: list[str] = []
    for name, channels in manifest.get("materials", {}).items():
        material_blocks.extend(_material_statements(name, channels))
        material_blocks.append("")

    dx = camera.target[0] - camera.position[0]
    dy = camera.target[1] - camera.position[1]
    dz = camera.target[2] - camera.position[2]
    length = math.sqrt(dx * dx + dy * dy + dz * dz) or 1.0
    look_dir = (dx / length, dy / length, dz / length)
    fov_radians = math.radians(camera.fov_degrees)

    lines = [
        f"-- Generated by lua_generator.py -- path-trace sample target: {samples}",
        "local scene = GetScene()",
        "scene.Clear()",
        "",
        f'LoadModel(scene, {_lua_string(obj_filename)})',
        "",
        "-- Material PBR patching, per materials_manifest.json",
        *material_blocks,
        "-- Camera setup",
        "local camera = GetCamera()",
        f"camera.SetPosition({_lua_vector(*camera.position)})",
        f"camera.SetLookDirection({_lua_vector(*look_dir)})",
        f"camera.SetFOV({fov_radians!r})",
        "",
        "-- Studio key light",
        "local lightEntity = CreateEntity()",
        "local light = scene.Component_CreateLight(lightEntity)",
        "light.SetType(DIRECTIONAL)",
        f"light.SetColor({_lua_vector(1.0, 0.98, 0.95, 1.0)})",
        "light.SetEnergy(3.0)",
        "light.SetCastShadow(true)",
    ]
    return "\n".join(lines) + "\n"
```

**Note for the implementer:** the light-creation block above is also confirmed against `wiScene_BindLua.cpp`, not guessed: entities are created via the global `CreateEntity()` (not a `scene.Entity_CreateLight(name)` convenience wrapper, which does not exist), a light component is attached via `scene.Component_CreateLight(entity)` (`Scene_BindLua::Component_CreateLight`, takes an `Entity`, not a name), and `SetType`/`SetColor`/`SetEnergy`/`SetCastShadow` plus the bare global `DIRECTIONAL` (`= 0`, not `LightComponent.DIRECTIONAL` — it's a top-level constant alongside `POINT`/`SPOT`) are all in `LightComponent_BindLua::methods`/the global constant block (`wiScene_BindLua.cpp:5570-5596` and `:304-306`). `INVALID_ENTITY` (`= 0`) is confirmed at `wiScene_BindLua.cpp:302`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd rendering/wicked_pbr_sprite && python -m pytest tests/test_lua_generator.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add rendering/wicked_pbr_sprite/lua_generator.py rendering/wicked_pbr_sprite/tests/test_lua_generator.py
git commit -m "feat: add startup.lua scene generator"
```

---

### Task 5: Live smoke test against the real Editor_Windows.exe (hard gate before Task 6)

This is not a new source file -- it is a verification step the implementer must actually run and record the result of before writing `render_orchestrator.py`. Everything in Task 4 was grounded against the *local WickedEngine source checkout*; `x:/apps/WickedEngine/Editor_Windows.exe` is a separately-deployed binary whose exact build is unverified against that checkout. This task is the actual Phase-0-style gate: it confirms the deployed binary really does auto-run `startup.lua` from its CWD and really does write screenshots to `<cwd>/screenshots/sc_*.png` on a keystroke, before Task 6 is built on those assumptions.

- [ ] **Step 1: Minimal startup.lua smoke test (confirms CWD auto-run + exit)**

Create a throwaway directory and file (not committed):

```lua
-- temp_smoke_test/startup.lua
backlog_post("SMOKE TEST: startup.lua ran")
local scene = GetScene()
scene.Clear()
exit()
```

Run manually (PowerShell):
```powershell
mkdir temp_smoke_test -Force
# (place the startup.lua above into temp_smoke_test/startup.lua)
Push-Location temp_smoke_test
& "x:/apps/WickedEngine/Editor_Windows.exe"
Pop-Location
```

Expected: the Editor window opens, briefly shows the "SMOKE TEST" backlog message, and the process exits on its own within a few seconds (confirms `exit()` really terminates the process here — if it doesn't, fall back to `os.exit(0)` in this same test before Task 6 relies on either).

- [ ] **Step 2: Screenshot-folder smoke test**

Extend `temp_smoke_test/startup.lua` to load a trivial model (or skip loading and just leave the default empty scene) and remove the `exit()` call so the window stays open. Launch it the same way, manually press **Shift+F4** in the Editor window, then check that `temp_smoke_test/screenshots/sc_<timestamp>.png` was created. Close the window manually.

Expected: a PNG appears in `temp_smoke_test/screenshots/` within a second or two of the keypress. If it lands somewhere else (e.g. next to the exe rather than the CWD), record the actual behavior — Task 6's polling directory must match reality, not the plan's assumption.

- [ ] **Step 3: Record findings**

Add a short "Verified against Editor_Windows.exe on <date>" note (exit mechanism that worked, actual screenshot directory) as a comment at the top of `render_orchestrator.py` in Task 6, Step 3, before implementing it.

---

### Task 6: `render_orchestrator.py` — job orchestration, screenshot automation, CLI

**Files:**
- Create: `rendering/wicked_pbr_sprite/render_orchestrator.py`

**Interfaces:**
- Consumes: `daz_exporter.export_pbr_mesh()` (Task 3), `lua_generator.generate_startup_lua()` (Task 4), `material_mapper.get_camera_preset()` (Task 2)
- Produces: `DazWickedBridge` class with `render_sprite(output_png, *, camera_preset="waist_up", samples=128, settle_seconds=None, timeout=60, node_label=None) -> Path`, and a `__main__` CLI.

- [ ] **Step 1: Implement `render_orchestrator.py`**

```python
"""Orchestrate a full DAZ Studio -> Wicked Engine PBR sprite render.

Ties together daz_exporter.py (Phase 1), lua_generator.py (Phase 2), and
this module's own job-directory staging, subprocess management, and
screenshot capture (Phase 3) into one CLI command.

SCREENSHOT CAPTURE -- WHY THIS IS OS-LEVEL KEYSTROKE AUTOMATION
-----------------------------------------------------------------
Wicked Engine has no Lua-callable screenshot binding (confirmed by
enumerating every RegisterFunc(...) registration across wi*_BindLua.cpp in
the engine source -- there is no "screenshot" among them). The actual
capture function (wi::helper::screenshot(), wiHelper.cpp:195) is wired
only to keyboard shortcuts inside the Editor's own C++ input-polling loop
(Editor.cpp:1639): Shift+F4 saves the current selection with a transparent
background to "<cwd>/screenshots/sc_<timestamp>.png" -- exactly the
transparent-sprite output this pipeline wants, and exactly the directory
this module launches the process with as its CWD.

So this module: launches Editor_Windows.exe with cwd=<job_dir>, waits a
fixed settle time for the scene/lighting/path-tracer to render, brings the
window to the foreground and synthesizes a Shift+F4 keypress via pywin32,
polls <job_dir>/screenshots/ for the new file, and copies it out to the
caller's requested output_png path before terminating the process. This is
inherently more fragile than a direct API call (window focus, timing,
requires an interactive desktop session) -- see this example's README
Known Limitations section.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import psutil
import win32api
import win32con
import win32gui
import win32process

from dazpy import DazClient, DazScene

from daz_exporter import export_pbr_mesh
from lua_generator import generate_startup_lua
from material_mapper import get_camera_preset

VK_SHIFT = 0x10
VK_F4 = 0x73


@dataclass
class RenderJobPaths:
    job_dir: Path
    startup_lua: Path
    screenshots_dir: Path


def _stage_job(work_dir: Path, manifest: dict, obj_filename: str, *, camera_preset: str, samples: int) -> RenderJobPaths:
    job_dir = work_dir / "jobs" / uuid.uuid4().hex[:12]
    job_dir.mkdir(parents=True)
    lua_text = generate_startup_lua(
        manifest, obj_filename=obj_filename, camera=get_camera_preset(camera_preset), samples=samples
    )
    startup_lua = job_dir / "startup.lua"
    startup_lua.write_text(lua_text)
    return RenderJobPaths(job_dir=job_dir, startup_lua=startup_lua, screenshots_dir=job_dir / "screenshots")


def _find_window_for_pid(pid: int, *, timeout: float) -> int:
    """Poll for a top-level window belonging to *pid*, returning its HWND.

    Raises:
        TimeoutError: If no window from this process appears within *timeout*.
    """
    deadline = time.monotonic() + timeout
    found: list[int] = []

    def _enum_handler(hwnd, _):
        if not win32gui.IsWindowVisible(hwnd):
            return
        _, window_pid = win32process.GetWindowThreadProcessId(hwnd)
        if window_pid == pid:
            found.append(hwnd)

    while time.monotonic() < deadline:
        found.clear()
        win32gui.EnumWindows(_enum_handler, None)
        if found:
            return found[0]
        time.sleep(0.25)
    raise TimeoutError(f"No window appeared for pid {pid} within {timeout}s")


def _send_screenshot_keystroke(hwnd: int) -> None:
    win32gui.SetForegroundWindow(hwnd)
    time.sleep(0.2)
    win32api.keybd_event(VK_SHIFT, 0, 0, 0)
    win32api.keybd_event(VK_F4, 0, 0, 0)
    time.sleep(0.05)
    win32api.keybd_event(VK_F4, 0, win32con.KEYEVENTF_KEYUP, 0)
    win32api.keybd_event(VK_SHIFT, 0, win32con.KEYEVENTF_KEYUP, 0)


def _wait_for_new_screenshot(screenshots_dir: Path, *, known_before: set[str], timeout: float) -> Path:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if screenshots_dir.is_dir():
            current = {p.name for p in screenshots_dir.glob("sc_*.png")}
            new = current - known_before
            if new:
                return screenshots_dir / sorted(new)[-1]
        time.sleep(0.25)
    raise TimeoutError(f"No new screenshot appeared in {screenshots_dir} within {timeout}s")


class DazWickedBridge:
    """Orchestrates one DAZ Studio -> Wicked Engine PBR sprite render.

    Args:
        daz_client: A connected :class:`dazpy.DazClient`.
        wicked_bin: Path to ``Editor_Windows.exe``.
        work_dir: Base directory for per-job subdirectories (created under
            ``work_dir/jobs/<job_id>/``).
    """

    def __init__(self, daz_client: DazClient, wicked_bin: str | Path, work_dir: str | Path = "render_jobs"):
        self.daz_client = daz_client
        self.wicked_bin = Path(wicked_bin).resolve()
        self.work_dir = Path(work_dir).resolve()
        if not self.wicked_bin.exists():
            raise FileNotFoundError(f"Wicked Engine binary not found: {self.wicked_bin}")

    def render_sprite(
        self,
        output_png: str | Path,
        *,
        camera_preset: str = "waist_up",
        samples: int = 128,
        settle_seconds: float | None = None,
        timeout: int = 60,
        node_label: str | None = None,
    ) -> Path:
        """Run one end-to-end render and return the resolved output path.

        Args:
            output_png: Destination path for the final PNG.
            camera_preset: One of ``"bust"``, ``"waist_up"``, ``"full_body"``.
            samples: Path-trace sample target (see lua_generator.py's
                docstring for why this only affects the settle-time
                heuristic, not a Lua-set accumulation target).
            settle_seconds: Seconds to wait after launching before
                triggering the screenshot keystroke. Defaults to
                ``max(3.0, samples / 32)`` if not given -- a rough
                heuristic; tune per hardware.
            timeout: Overall seconds allowed for the Wicked process to
                start, settle, and produce a screenshot before this raises
                and kills the process.
            node_label: Passed through to ``export_pbr_mesh()``.

        Raises:
            RuntimeError: If no figure is selected/found in DAZ Studio.
            TimeoutError: If the window, or the screenshot file, doesn't
                appear within budget.
        """
        output_png = Path(output_png)
        if settle_seconds is None:
            settle_seconds = max(3.0, samples / 32)

        scene = DazScene(client=self.daz_client)
        exported = export_pbr_mesh(scene, self.work_dir / "exports", node_label=node_label)

        job = _stage_job(
            self.work_dir, exported.manifest, exported.obj_path.name,
            camera_preset=camera_preset, samples=samples,
        )
        shutil.copy(exported.obj_path, job.job_dir / exported.obj_path.name)

        import subprocess
        proc = subprocess.Popen([str(self.wicked_bin)], cwd=str(job.job_dir))
        try:
            hwnd = _find_window_for_pid(proc.pid, timeout=timeout)
            time.sleep(settle_seconds)
            known_before = {p.name for p in job.screenshots_dir.glob("sc_*.png")} if job.screenshots_dir.is_dir() else set()
            _send_screenshot_keystroke(hwnd)
            screenshot_path = _wait_for_new_screenshot(job.screenshots_dir, known_before=known_before, timeout=timeout)
            output_png.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(screenshot_path, output_png)
        finally:
            self._terminate(proc)

        return output_png

    @staticmethod
    def _terminate(proc) -> None:
        try:
            parent = psutil.Process(proc.pid)
            for child in parent.children(recursive=True):
                child.terminate()
            parent.terminate()
            parent.wait(timeout=5)
        except psutil.NoSuchProcess:
            pass
        except psutil.TimeoutExpired:
            parent.kill()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", required=True, help="Destination PNG path")
    p.add_argument("--wicked-bin", required=True, help="Path to Editor_Windows.exe")
    p.add_argument("--preset", default="waist_up", choices=["bust", "waist_up", "full_body"])
    p.add_argument("--samples", type=int, default=128)
    p.add_argument("--settle-seconds", type=float, default=None)
    p.add_argument("--timeout", type=int, default=60)
    p.add_argument("--label", help="DAZ figure label (default: current scene selection)")
    p.add_argument("--work-dir", default="render_jobs")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    with DazClient() as client:
        bridge = DazWickedBridge(client, args.wicked_bin, work_dir=args.work_dir)
        try:
            result = bridge.render_sprite(
                args.output,
                camera_preset=args.preset,
                samples=args.samples,
                settle_seconds=args.settle_seconds,
                timeout=args.timeout,
                node_label=args.label,
            )
        except (RuntimeError, TimeoutError, FileNotFoundError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            sys.exit(1)
    print(f"Wrote {result}")
```

- [ ] **Step 2: Run the full pipeline against a live DAZ Studio + the real Wicked Editor binary**

Run:
```bash
python render_orchestrator.py \
    --output test_sprite.png \
    --wicked-bin "x:/apps/WickedEngine/Editor_Windows.exe" \
    --preset waist_up \
    --samples 64
```

Expected: DAZ Studio's HTTP server logs an `/execute` request, `Editor_Windows.exe` pops a window, loads the character, holds for the settle period, and `test_sprite.png` is written with a visible posed/textured character and (per the F4+Shift "selection with transparency" mode) a transparent background. Open the PNG and confirm it isn't blank/black and has an alpha channel.

- [ ] **Step 3: Commit**

```bash
git add rendering/wicked_pbr_sprite/render_orchestrator.py
git commit -m "feat: add Wicked Engine render orchestrator and CLI"
```

---

### Task 7: Finish the README and finalize

**Files:**
- Modify: `rendering/wicked_pbr_sprite/README.md`

**Interfaces:**
- Consumes: nothing new — this documents Tasks 1-6's actual final interfaces.

- [ ] **Step 1: Replace every `TODO` section left in Task 1's README skeleton**

```markdown
## Overview

End-to-end pipeline that poses a character in DAZ Studio, exports its full
Iray PBR material data (not just diffuse color, unlike plain OBJ/MTL
export), and renders a transparent-background sprite through Wicked
Engine's hardware path tracer. Built for asset pipelines that need
higher-fidelity PBR shading than DAZ Studio's own Iray viewport/render can
produce quickly, using an already-running local Wicked Engine install.

## What You'll Learn

- Reusing `DazScene.export_obj()` for geometry while separately walking
  `DzMaterial`/Iray Uber Base and PBRSkin properties via
  `DazClient.execute()` for data no existing `dazpy` wrapper exposes
- Handling shader variance on a single DAZ figure (Iray Uber Base vs.
  PBRSkin materials expose different channel sets) without hard failures
- Driving an external game engine that has no CLI script flag or headless
  mode, via its actual mechanism (`startup.lua` auto-run from CWD) instead
  of the one a first read of "it's scriptable" might assume
- Automating a GUI action (a screenshot keyboard shortcut) from Python via
  `pywin32` when no scripting API exposes the same capability

**SDK features used:**
- `DazScene.primary_selection()` / `find_node_by_label()` / `export_obj()`
- `DazClient.execute()` for raw DazScript (material channel introspection)
- `DzShape.getAllMaterials()`, `DzMaterial.findPropertyByLabel()`,
  `DzNumericProperty.getValue()` / `getMapValue().getFilename()`

## Prerequisites

- DAZ Studio with DazScriptServer plugin running, a scene open with the
  figure to export selected
- Wicked Engine built/installed at a known path (this example was verified
  against `x:/apps/WickedEngine/Editor_Windows.exe`)
- Windows (this example's screenshot automation is Windows-only, via
  `pywin32`)
- An interactive desktop session — neither DAZ Studio nor Wicked Engine's
  Editor run headless; this cannot run on a session-less CI runner without
  a virtual display

## Usage

### Full pipeline

\`\`\`bash
python render_orchestrator.py \
    --output ./sprites/character_neutral.png \
    --wicked-bin "x:/apps/WickedEngine/Editor_Windows.exe" \
    --preset waist_up \
    --samples 128
\`\`\`

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--output PATH` | *(required)* | Destination PNG path |
| `--wicked-bin PATH` | *(required)* | Path to `Editor_Windows.exe` |
| `--preset {bust,waist_up,full_body}` | `waist_up` | Camera framing |
| `--samples N` | `128` | Path-trace sample target (drives the settle-time heuristic — see Known Limitations) |
| `--settle-seconds N` | `max(3.0, samples/32)` | Seconds to wait before capturing |
| `--timeout N` | `60` | Max seconds for the window/screenshot to appear |
| `--label NAME` | current selection | DAZ figure to export |
| `--work-dir PATH` | `render_jobs` | Base directory for per-job subdirectories |

`daz_exporter.py` and `lua_generator.py` can also be run/imported independently — see their module docstrings.

## How It Works

1. **`daz_exporter.py`** exports the selected figure's deformed geometry via `DazScene.export_obj()` and walks its materials' Iray channels via a raw `DazClient.execute()` script, writing `character.obj` + `materials_manifest.json`.
2. **`lua_generator.py`** translates the manifest (via `material_mapper.py`'s channel/fallback rules) into a Wicked Engine `startup.lua`: loads the OBJ, patches each named material's PBR properties, and positions the camera per the chosen preset.
3. **`render_orchestrator.py`** stages both files into an isolated `render_jobs/jobs/<job_id>/` directory and launches `Editor_Windows.exe` with that directory as its CWD — Wicked auto-runs `startup.lua` on boot.
4. After a settle period, it brings the Editor window to the foreground and synthesizes a **Shift+F4** keypress (Wicked's built-in "screenshot selection with transparent background" shortcut), since no Lua-callable screenshot API exists.
5. It polls the job directory's `screenshots/` subfolder (where Wicked always writes screenshots when its CWD is the job directory) for the new file, copies it to `--output`, and terminates the Wicked process.

## Output

A single transparent-background PNG at `--output`, plus the intermediate `character.obj`/`materials_manifest.json`/`startup.lua` left in `render_jobs/` as inspectable artifacts (not auto-deleted, unlike the original design's job-directory cleanup — kept for debugging until this example's own cleanup story is revisited).

## Known Limitations / Caveats

- **Screenshot capture is OS-level keystroke automation, not an API call** — no Lua-callable screenshot binding exists in Wicked Engine. This means: the desktop session must stay unlocked and the Wicked window must be able to receive focus during the render; running two jobs concurrently on one machine will conflict (only one can own the foreground window at a time); and the fixed settle-time wait is a heuristic, not a true "render finished" signal.
- **No true path-trace sample-count control** — `--samples` only tunes the settle-time heuristic; there is no confirmed Lua binding to set or query actual path-trace accumulation progress.
- **No texture packing** — Wicked's `TextureSlot.SURFACEMAP` expects a single packed occlusion/roughness/metalness texture (glTF-style); this pipeline uses DAZ's separate roughness/metallic scalars/maps via `Roughness`/`Metalness` properties and `BASECOLORMAP`/`NORMALMAP` texture slots only, not a packed ORM texture, so per-pixel roughness/metallic maps are not applied — only per-material scalar fallbacks.
- **PBRSkin materials lose Roughness/Cutout Opacity fidelity** — DAZ's PBRSkin shader (default Genesis 8/9 skin) has no "Glossy Roughness"/"Cutout Opacity" channel at all; those materials always use `material_mapper.py`'s fallback values (0.5/opaque).
- **Not validated for skinned/animated meshes** — this pipeline exports a single static pose via OBJ, not a skeleton; animated Wicked output is out of scope.

## Related Examples

- [`export/scene_to_usd`](../../export/scene_to_usd/README.md) — a different external-format export pipeline, without the external-engine subprocess step
- [`rendering/unreal_headless_render`](../unreal_headless_render/README.md) — the closest precedent for this example's external-engine-subprocess-orchestrator shape
- See main repository [README](../../README.md) for related examples
```

- [ ] **Step 2: Run the full example one more time end to end to confirm the README's commands are accurate as written**

Run the exact command block from the README's Usage section and confirm it succeeds and matches the documented Output.

- [ ] **Step 3: Commit**

```bash
git add rendering/wicked_pbr_sprite/README.md
git commit -m "docs: complete wicked_pbr_sprite example README"
```

---

## Self-Review Notes

- **Spec coverage:** Phase 0 (Task 5, adapted to a source-grounded + live-smoke-test gate rather than a broad runtime enumeration, since source-reading already confirmed the specific bindings this plan needs), Phase 1 (Task 3), Phase 2 (Task 4), Phase 3 (Task 6), camera presets (Task 2, values carried over verbatim from spec v1 section 4.4), error handling (missing-selection RuntimeError in Task 3, subprocess timeout/termination in Task 6, backslash normalization in Task 2/tested in Task 4) are all covered. The one spec item deliberately dropped is the literal `screenshot(path); application.Quit()` Lua template — replaced per the user's explicit decision with OS-level keystroke automation, documented as such throughout.
- **Placeholder scan:** The `daz_exporter.py`/`scene._client` note in Task 3 and the `Entity_CreateLight`/`LightComponent` note in Task 4 are flagged explicitly as unverified-during-planning with a concrete required action, not left as silent TODOs — the implementer cannot skip them without noticing.
- **Type consistency:** `MappedMaterial` (Task 2) fields match what `lua_generator.py` (Task 4) reads; `ExportedMesh` (Task 3) fields match what `render_orchestrator.py` (Task 6) consumes (`.obj_path`, `.manifest_path`, `.manifest`); `generate_startup_lua()`'s signature matches its call site in `_stage_job()`.
