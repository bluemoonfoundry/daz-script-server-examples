"""Daz morph name -> Unreal SkeletalMesh morph target name mapping.

DAZ morph channel names (e.g. "PHMSmile", "eCTRLBrowUp") almost never match
the morph target names baked into an Unreal SkeletalMesh, even when the mesh
was exported from the same DAZ figure via a bridge plugin -- the bridge's
export/import process is free to rename, merge, or drop targets. There is no
generic way to infer this mapping, so it is kept here as an explicit,
hand-maintained table specific to your project's SkeletalMesh.

Fill in entries as you discover which DAZ morphs matter for your renders.
Any DAZ morph without an entry here is skipped (with a warning) by
ue_headless_render.py rather than causing the render to fail -- a Daz scene
will typically report far more active morphs than exist as targets on the
Unreal mesh (eyelid deltas driven by JCMs, correctives, etc. that never make
it through the bridge export).
"""

MORPH_NAME_MAP: dict[str, str] = {
    # "PHMSmile": "Smile",
    # "eCTRLBrowUp": "Brow_Up",
}
