"""DAZ Studio -> Unreal Engine 5 headless render pipeline orchestrator.

PURPOSE
-------
This is the single entry point for the full pipeline: it runs
daz_exporter.py to capture the current DAZ Studio pose/morphs into
render_job.json, then launches UnrealEditor-Cmd.exe headlessly to apply
that job and render a 4K still via Movie Render Queue (ue_headless_render.py),
then confirms the output file was produced.

Usage:
    python render_orchestrator.py --ue-cmd "C:/Program Files/Epic Games/UE_5.4/Engine/Binaries/Win64/UnrealEditor-Cmd.exe" \\
        --uproject "C:/MyProject/MyProject.uproject" \\
        --character-id BP_Character_Heroine --camera-name CineCamera_Shot005 \\
        --output-dir "C:/VNRenders/Scene01" --output-filename Scene01_Shot005.png

    python render_orchestrator.py --dry-run --ue-cmd ... --uproject ... --character-id ... --camera-name ... --output-dir ... --output-filename ...
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ue-cmd", required=True, metavar="PATH", help="Path to UnrealEditor-Cmd.exe")
    p.add_argument("--uproject", required=True, metavar="PATH", help="Path to the target .uproject file")
    p.add_argument("--job-path", default=str(THIS_DIR / "temp" / "render_job.json"), help="Where daz_exporter.py writes render_job.json")

    # daz_exporter.py pass-through args
    p.add_argument("--figure", help="Figure's internal DazScript name")
    p.add_argument("--label", help="Figure's user-visible label (default: current scene selection)")
    p.add_argument("--character-id", required=True, help="Target actor name/label in the Unreal level")
    p.add_argument("--camera-name", required=True, help="CineCameraActor name in the Unreal level")
    p.add_argument("--output-dir", required=True, help="Directory MRQ writes the PNG to")
    p.add_argument("--output-filename", required=True, help="Output PNG filename")
    p.add_argument("--resolution", nargs=2, type=int, default=[3840, 2160], metavar=("WIDTH", "HEIGHT"))
    p.add_argument("--warmup-frames", type=int, default=20)

    p.add_argument("--nullrhi", action="store_true", help="Pass -NullRHI to Unreal (disables GPU rendering -- only useful for wiring smoke tests, NOT real renders)")
    p.add_argument("--timeout", type=float, default=600.0, help="Seconds to wait for the Unreal render process")
    p.add_argument("--dry-run", action="store_true", help="Print the plan without running anything")
    return p.parse_args()


def _fail(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    args = _parse_args()

    exporter_cmd = [
        sys.executable, str(THIS_DIR / "daz_exporter.py"),
        "--character-id", args.character_id,
        "--camera-name", args.camera_name,
        "--output-dir", args.output_dir,
        "--output-filename", args.output_filename,
        "--resolution", str(args.resolution[0]), str(args.resolution[1]),
        "--warmup-frames", str(args.warmup_frames),
        "--out", args.job_path,
    ]
    if args.figure:
        exporter_cmd += ["--figure", args.figure]
    if args.label:
        exporter_cmd += ["--label", args.label]

    ue_cmd = [
        args.ue_cmd, args.uproject,
        f'-ExecutePythonScript="{THIS_DIR / "ue_headless_render.py"}"',
        f'-RenderJobPath="{args.job_path}"',
        "-Unattended", "-nosplash",
    ]
    if args.nullrhi:
        ue_cmd.append("-NullRHI")

    output_path = os.path.join(args.output_dir, args.output_filename)

    if args.dry_run:
        print("=== Dry run ===")
        print(f"  1. Export pose:  {' '.join(exporter_cmd)}")
        print(f"  2. Render (UE):  {' '.join(ue_cmd)}")
        print(f"  3. Expect output at: {output_path}")
        print("  (dry run -- no actions taken)")
        return

    # ── 1. Export pose/morphs from DAZ Studio ─────────────────────────────────
    print("Exporting pose from DAZ Studio...", flush=True)
    result = subprocess.run(exporter_cmd)
    if result.returncode != 0:
        _fail("daz_exporter.py failed -- is DAZ Studio running with DazScriptServer active?")
    print(f"OK ({args.job_path})")

    # ── 2. Launch Unreal headlessly and render ────────────────────────────────
    print("Launching Unreal Engine (headless)...", flush=True)
    t0 = time.monotonic()
    try:
        result = subprocess.run(ue_cmd, timeout=args.timeout)
    except subprocess.TimeoutExpired:
        _fail(f"Unreal did not finish within {args.timeout:.0f}s.")
    elapsed = time.monotonic() - t0
    if result.returncode != 0:
        _fail(f"UnrealEditor-Cmd.exe exited with code {result.returncode} after {elapsed:.1f}s.")
    print(f"OK ({elapsed:.1f}s)")

    # ── 3. Verify output ───────────────────────────────────────────────────────
    if not os.path.isfile(output_path):
        _fail(f"Unreal exited cleanly but no output file was found at {output_path}.")

    print(f"\nRender complete: {output_path}")


if __name__ == "__main__":
    main()
