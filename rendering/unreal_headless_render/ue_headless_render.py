"""Unreal Engine 5 headless render script: apply a DAZ pose and render a still via MRQ.

PURPOSE
-------
This script runs *inside* UnrealEditor-Cmd.exe's embedded Python
interpreter (invoked via `-ExecutePythonScript="ue_headless_render.py"`,
see render_orchestrator.py). It reads the render_job.json file produced by
daz_exporter.py, applies the exported bone rotations and morph target
values to a matching SkeletalMeshActor in the currently-open level, then
renders a single 4K still through Unreal's Movie Render Queue (MRQ).

This is example code, not a production render-farm script. Two things in
particular are project-specific and cannot be solved generically here --
read the callouts below before relying on this:

CAVEAT 1 -- Anim Mode / Anim Blueprint override
------------------------------------------------
Setting bone transforms or morph target values on a SkeletalMeshComponent
only "sticks" for a render if nothing else is driving that bone/morph every
tick. If the target actor's Anim Blueprint (or a Control Rig, or a running
animation) writes to the same bones/morphs after this script runs, your
manual pose will be silently overwritten before MRQ captures a frame. In
your Unreal project, either:
  - Set the component's Animation Mode to "Use Animation Blueprint" with an
    AnimBP that does NOT drive the bones/morphs you're posing here (e.g. one
    that only handles physics/cloth), or
  - Set Animation Mode to "Use Animation Asset" with no asset assigned
    (pure single-frame pose), or
  - Drive the pose through a Control Rig / Level Sequence track instead of
    this script's direct component calls -- more robust, but out of scope
    for this example.
This is a one-time Unreal project setup decision, not something this script
can detect or fix for you.

CAVEAT 2 -- MRQ requires a Level Sequence
------------------------------------------
Movie Render Queue renders a Level Sequence, not "the current viewport" --
there is no MRQ concept of rendering a single unsequenced still. This
script expects a minimal Level Sequence asset (can be a single frame) to
already exist per shot/camera, referenced by
`render_config.level_sequence_path` in render_job.json (defaults to
`/Game/Cinematics/<camera_name>_Sequence` if omitted -- adjust the
DEFAULT_SEQUENCE_PATH_TEMPLATE constant below to match your project's
naming convention).

CAVEAT 3 -- API surface varies by engine version
--------------------------------------------------
The exact `unreal.*` class/method/enum names below (especially
`set_bone_transform_by_name`'s BoneSpace enum, and the warm-up-frame
setting's exact property name) have moved between UE5 minor versions.
Before running against your project, verify each call against your
installed engine's Python API reference (in the Unreal Python console:
`help(unreal.SkeletalMeshComponent.set_bone_transform_by_name)`, etc.) and
adjust as needed -- this cannot be verified without a running UE5 instance.

Usage (invoked by render_orchestrator.py, not run standalone):
    UnrealEditor-Cmd.exe MyProject.uproject -ExecutePythonScript="ue_headless_render.py" -RenderJobPath="C:/.../render_job.json" -Unattended -nosplash
"""

from __future__ import annotations

import json
import sys

try:
    import unreal
except ImportError:  # pragma: no cover - only importable inside UE5's embedded Python
    unreal = None

from morph_name_map import MORPH_NAME_MAP

DEFAULT_SEQUENCE_PATH_TEMPLATE = "/Game/Cinematics/{camera_name}_Sequence"


def _log(msg: str) -> None:
    print(f"[ue_headless_render] {msg}")


def _get_command_line_token(key: str) -> str | None:
    """Extract a `-Key="value"` token from Unreal's full process command line.

    UE's `-ExecutePythonScript=` does not populate sys.argv with the rest of
    the engine command line, so extra parameters (like the render job path)
    must be passed as their own `-Key=value` tokens and pulled out of
    unreal.SystemLibrary.get_command_line() instead.
    """
    cmdline = unreal.SystemLibrary.get_command_line()
    marker = f"-{key}="
    idx = cmdline.find(marker)
    if idx == -1:
        return None
    rest = cmdline[idx + len(marker):]
    if rest.startswith('"'):
        end = rest.find('"', 1)
        return rest[1:end] if end != -1 else rest[1:]
    end = rest.find(" ")
    return rest if end == -1 else rest[:end]


def _find_actor_by_name(character_id: str):
    actor_subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    for actor in actor_subsystem.get_all_level_actors():
        if actor.get_actor_label() == character_id or actor.get_name() == character_id:
            return actor
    return None


def apply_morphs(skel_comp, morphs: dict) -> None:
    applied, skipped = 0, []
    for daz_name, value in morphs.items():
        ue_name = MORPH_NAME_MAP.get(daz_name)
        if ue_name is None:
            skipped.append(daz_name)
            continue
        skel_comp.set_morph_target(unreal.Name(ue_name), float(value))
        applied += 1
    _log(f"Applied {applied} morph target(s); skipped {len(skipped)} unmapped: {skipped}")


def apply_bones(skel_comp, bones: dict) -> None:
    applied, skipped = 0, []
    for bone_name, quat in bones.items():
        fname = unreal.Name(bone_name)
        if skel_comp.get_bone_index(fname) == -1:
            skipped.append(bone_name)
            continue
        rotation = unreal.Quat(quat["x"], quat["y"], quat["z"], quat["w"])
        transform = unreal.Transform(rotation=rotation)
        # BoneSpace enum member name/behavior varies by engine version -- see
        # CAVEAT 3 above. This applies the rotation in the bone's own local
        # space, matching how DazBone.local_rotation was read.
        skel_comp.set_bone_transform_by_name(fname, transform, unreal.BoneSpaces.WORLD_SPACE)
        applied += 1
    if skipped:
        _log(f"Applied {applied} bone rotation(s); skipped {len(skipped)} not found on target mesh: {skipped}")
    else:
        _log(f"Applied {applied} bone rotation(s).")


def configure_and_render(job: dict, sequence_path: str) -> None:
    render_config = job["render_config"]

    subsystem = unreal.get_editor_subsystem(unreal.MoviePipelineQueueSubsystem)
    queue = subsystem.get_queue()
    queue.delete_all_jobs()

    pipeline_job = queue.allocate_new_job(unreal.MoviePipelineExecutorJob)
    editor_world = unreal.EditorLevelLibrary.get_editor_world()
    pipeline_job.map = unreal.SoftObjectPath(editor_world.get_path_name())
    pipeline_job.sequence = unreal.SoftObjectPath(sequence_path)
    pipeline_job.job_name = render_config["output_filename"]

    config = pipeline_job.get_configuration()
    config.find_or_add_setting_by_class(unreal.MoviePipelineDeferredPassBase)

    output_setting = config.find_or_add_setting_by_class(unreal.MoviePipelineOutputSetting)
    output_setting.output_directory = unreal.DirectoryPath(render_config["output_dir"])
    output_setting.file_name_format = render_config["output_filename"].rsplit(".", 1)[0]
    width, height = render_config["resolution"]
    output_setting.output_resolution = unreal.IntPoint(width, height)

    aa_setting = config.find_or_add_setting_by_class(unreal.MoviePipelineAntiAliasingSetting)
    aa_setting.render_warm_up_count = render_config.get("warmup_frames", 20)

    camera_actor = _find_actor_by_name(render_config["camera_name"])
    if camera_actor is None:
        _log(f"WARNING: camera '{render_config['camera_name']}' not found by name; MRQ will use the sequence's bound camera.")

    executor = unreal.MoviePipelinePIEExecutor()
    subsystem.render_queue_with_executor_instance(executor)
    _log(f"Render submitted -> {render_config['output_dir']}/{render_config['output_filename']}")


def main() -> None:
    if unreal is None:
        print("ERROR: this script must be run inside UnrealEditor-Cmd.exe's embedded Python.", file=sys.stderr)
        sys.exit(1)

    job_path = _get_command_line_token("RenderJobPath")
    if not job_path:
        _log("ERROR: -RenderJobPath=\"...\" was not found on the command line.")
        sys.exit(1)

    with open(job_path) as f:
        job = json.load(f)

    actor = _find_actor_by_name(job["character_id"])
    if actor is None:
        _log(f"ERROR: actor '{job['character_id']}' not found in the current level.")
        sys.exit(1)

    skel_comp = actor.get_component_by_class(unreal.SkeletalMeshComponent)
    if skel_comp is None:
        _log(f"ERROR: actor '{job['character_id']}' has no SkeletalMeshComponent.")
        sys.exit(1)

    apply_morphs(skel_comp, job.get("morphs", {}))
    apply_bones(skel_comp, job.get("bones", {}))

    sequence_path = job["render_config"].get(
        "level_sequence_path",
        DEFAULT_SEQUENCE_PATH_TEMPLATE.format(camera_name=job["render_config"]["camera_name"]),
    )
    configure_and_render(job, sequence_path)


if __name__ == "__main__":
    main()
