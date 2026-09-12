<#
.SYNOPSIS
    Runs pose_transfer_photo.py against the pinocchio-ik conda-forge
    environment, without requiring a manual `conda activate` step first.

.DESCRIPTION
    `--backend pinocchio` needs the real Pinocchio C++/Eigen library, which
    only ships prebuilt on Windows via conda-forge (see environment.yml's
    header comment and pinocchio_ik.py's module docstring for why). Calling
    that environment's python.exe by raw path works but is easy to get
    subtly wrong -- numpy's BLAS DLL lives under the env's Library/bin,
    and if that's not on PATH, matrix multiplication segfaults the
    interpreter with NO Python traceback at all (it just vanishes). `conda
    run -n pinocchio-ik` sets this up correctly by itself, so this script
    is a thin wrapper around that instead of reimplementing env activation.

    All arguments after the script name are forwarded verbatim to
    pose_transfer_photo.py -- this script does not add or change any of
    that script's own CLI flags.

.EXAMPLE
    ./run_pinocchio.ps1 photo.jpg --backend pinocchio

.EXAMPLE
    ./run_pinocchio.ps1 --batch photos/ --backend pinocchio --save-poses --stats-csv

.NOTES
    If `conda` isn't already resolvable on PATH (a normal "Miniforge Prompt"
    shell has this; a plain PowerShell window usually doesn't), set
    $env:PINOCCHIO_IK_CONDA_ROOT to your miniforge/miniconda install root
    once (e.g. in your PowerShell profile) rather than editing this script:

        $env:PINOCCHIO_IK_CONDA_ROOT = "X:\apps\miniforge3"
#>

$ErrorActionPreference = "Stop"

function Resolve-CondaExe {
    $existing = Get-Command conda -ErrorAction SilentlyContinue
    if ($existing) { return $existing.Source }

    $root = $env:PINOCCHIO_IK_CONDA_ROOT
    if (-not $root) {
        # Falls back to where this project's own pinocchio-ik env has been
        # observed installed (see bd daz-script-server-hewu) -- a starting
        # guess, not a guarantee; override via $env:PINOCCHIO_IK_CONDA_ROOT
        # if your install lives elsewhere.
        $root = "X:\apps\miniforge3"
    }
    $condaBat = Join-Path $root "condabin\conda.bat"
    if (Test-Path $condaBat) { return $condaBat }

    throw (
        "Could not find a conda executable. Either put conda on PATH " +
        "(e.g. run this from a Miniforge/Anaconda Prompt), or set " +
        "`$env:PINOCCHIO_IK_CONDA_ROOT to your install root. Tried: $condaBat"
    )
}

function Test-PinocchioEnv {
    param([string]$CondaExe)
    $envList = & $CondaExe env list 2>&1
    if ($LASTEXITCODE -ne 0 -or -not ($envList -match "pinocchio-ik")) {
        throw (
            "The 'pinocchio-ik' conda environment doesn't exist yet. Create " +
            "it once with:`n`n    conda env create -f environment.yml`n`n" +
            "(run from this directory: $PSScriptRoot)"
        )
    }
}

$condaExe = Resolve-CondaExe
Test-PinocchioEnv -CondaExe $condaExe

$scriptPath = Join-Path $PSScriptRoot "pose_transfer_photo.py"
& $condaExe run -n pinocchio-ik --no-capture-output --cwd $PSScriptRoot `
    python $scriptPath @args
exit $LASTEXITCODE
