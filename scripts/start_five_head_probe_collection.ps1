<#
.SYNOPSIS
Full five_head_probe Training collection (6,059 frames; the 32 pilot frames are reused) and offline Gate retraining.

.DESCRIPTION
Modes
  Check    zero-cost preflight: inventory, sealed sources, pilot reuse, credential files, no running collector.
  Run      preflight, then paid collection into the fixed run directory. Resumes automatically when the
           directory already exists. On completion trains the 54-feature Gate and the 42-feature ablation.
  Status   read-only progress / ETA / budget of the run directory (safe while Run is active).
  Recover  after a hard interruption (closed window, power loss): quarantine uncertain targets so Run can
           resume. Dry run unless -Apply.
  Train    offline training only (requires a completed run; no API calls).

Do not edit scripts\collect_five_head_probe_pilot.py while a run is in progress: its hash is frozen in the
run's plan.json and --resume refuses a drifted plan. Status/Recover/launcher edits are safe.

.EXAMPLE
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_five_head_probe_collection.ps1 -Mode Check
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_five_head_probe_collection.ps1 -Mode Run
#>
param(
    [ValidateSet('Check', 'Run', 'Status', 'Recover', 'Train')][string]$Mode = 'Check',
    [ValidateRange(1, 8)][int]$Workers = 8,
    [switch]$Apply
)
$ErrorActionPreference = 'Stop'
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch { }
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$py = Join-Path $projectRoot '.venv-p2\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $py)) { throw "Interpreter not found: $py (this repo runs on .venv-p2)" }
$collector = Join-Path $PSScriptRoot 'collect_five_head_probe_pilot.py'
$statusTool = Join-Path $PSScriptRoot 'five_head_probe_collection_status.py'
$recoverTool = Join-Path $PSScriptRoot 'recover_five_head_probe_interrupted.py'
$trainer = Join-Path $PSScriptRoot 'train_gate_with_phase.py'
$pilot = Join-Path $projectRoot 'artifacts\training\gate\five_head_probe_pilot_32_20260916_r1'
$run = Join-Path $projectRoot 'artifacts\training\gate\five_head_probe_full_20260916_r1'
$gateOut = Join-Path $projectRoot 'artifacts\training\gate\five_head_probe_gate_20260916_r1'
$ablationOut = Join-Path $projectRoot 'artifacts\training\gate\five_head_probe_gate_base42_20260916_r1'
$logDir = Join-Path $projectRoot 'logs\five_head_probe'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$launcherLog = Join-Path $logDir ('launcher_{0}_{1}.log' -f $Mode.ToLower(), (Get-Date -Format 'yyyyMMdd_HHmmss'))

function Write-Log([string]$Message) {
    $line = '[{0}] {1}' -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $Message
    Write-Host $line
    Add-Content -LiteralPath $launcherLog -Value $line
}

function Invoke-Python([string[]]$PyArgs) {
    Write-Log ('python -X utf8 -u ' + ($PyArgs -join ' '))
    # Native stderr (tqdm progress bar, tracebacks) must not become terminating errors when redirected.
    # Out-Host keeps python's stdout out of this function's return value, so only the exit code is returned.
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try { & $py -X utf8 -u @PyArgs | Out-Host } finally { $ErrorActionPreference = $previous }
    return $LASTEXITCODE
}

function Get-RunningCollectors {
    try {
        return @(Get-CimInstance Win32_Process | Where-Object {
            $_.Name -like 'python*' -and $_.CommandLine -like '*collect_five_head_probe_pilot*' })
    } catch {
        Write-Log "Process check unavailable: $($_.Exception.Message)"
        return @()
    }
}

function Invoke-Preflight {
    foreach ($secret in @('.env.reviewer_routes.local', 'docs\DS-API.txt', 'docs\GLM-API.txt')) {
        if (-not (Test-Path -LiteralPath (Join-Path $projectRoot $secret))) { throw "Credential file missing: $secret" }
    }
    if (-not (Test-Path -LiteralPath (Join-Path $pilot 'receipt.json'))) { throw "Pilot receipt missing: $pilot" }
    $running = Get-RunningCollectors
    if ($running.Count -gt 0) {
        throw ('A collector is already running (PIDs ' + (($running | ForEach-Object { $_.ProcessId }) -join ', ') + '). Never start a second one.')
    }
    Write-Log 'Preflight (no paid requests): inventory, sealed sources and pilot reuse.'
    $code = Invoke-Python @($collector, '--output', $run, '--check-only', '--all-training', '--reuse-pilot', $pilot)
    if ($code -ne 0) { throw 'Preflight failed. Nothing was started.' }
}

function Invoke-Train {
    if (-not (Test-Path -LiteralPath (Join-Path $run 'receipt.json'))) { throw "Collection is not complete ($run\receipt.json missing); nothing to train." }
    if (Test-Path -LiteralPath $gateOut) { Write-Log "Exists, skipping 54-feature training: $gateOut" }
    else {
        $code = Invoke-Python @($trainer, '--unified-cache', $run, '--output', $gateOut)
        if ($code -ne 0) { throw '54-feature Gate training failed.' }
    }
    if (Test-Path -LiteralPath $ablationOut) { Write-Log "Exists, skipping 42-feature ablation: $ablationOut" }
    else {
        $code = Invoke-Python @($trainer, '--unified-cache', $run, '--output', $ablationOut, '--drop-phase-probe-features')
        if ($code -ne 0) { throw '42-feature ablation training failed.' }
    }
    Write-Log "Training done. Compare $gateOut\report.json with $ablationOut\report.json (variants, calibration.selected, nested_review_frames). No default Gate was changed."
}

switch ($Mode) {
    'Check' {
        Invoke-Preflight
        Write-Log 'Estimate from the 32-frame pilot: about 19,400 requests, about 37 USD (OpenRouter) and 13 CNY (Aliyun), about 8 h at 8 workers.'
        Write-Log 'Plan caps written by the collector: openrouter_usd 60, aliyun_cny 30, glm_requests 6059, deepseek_requests 6059, xai_usd 0.'
        Write-Log "Check passed. Start with: -Mode Run   (run directory: $run)"
    }
    'Run' {
        Invoke-Preflight
        $runArgs = @($collector, '--output', $run, '--all-training', '--reuse-pilot', $pilot, '--allow-paid', '--workers', $Workers)
        if (Test-Path -LiteralPath (Join-Path $run 'receipt.json')) {
            Write-Log 'Collection already complete; skipping to training.'
        } elseif (Test-Path -LiteralPath $run) {
            $runArgs += '--resume'
            Write-Log "Resuming existing run directory $run (finished journals replay for free)."
        } else {
            Write-Log "Starting a fresh run directory $run."
        }
        Write-Log 'Keep this window open and the machine awake. Ctrl+C leaves a resumable state; rerun -Mode Run to continue.'
        $code = Invoke-Python $runArgs
        Write-Log "Collector exit code $code"
        if ($code -ne 0) {
            Write-Log 'Collector stopped. Inspect with -Mode Status. Soft stop: rerun -Mode Run. Hard interruption: -Mode Recover (then -Apply), then -Mode Run.'
            exit $code
        }
        if (Test-Path -LiteralPath (Join-Path $run 'receipt.json')) {
            Write-Log 'Collection sealed. Training offline (no API calls).'
            Invoke-Train
        } else {
            Write-Log 'Collector exited without a receipt; inspect with -Mode Status.'
            exit 1
        }
    }
    'Status' {
        $code = Invoke-Python @($statusTool, '--run', $run)
        exit $code
    }
    'Recover' {
        $recoverArgs = @($recoverTool, '--run', $run)
        if ($Apply) {
            $running = Get-RunningCollectors
            if ($running.Count -gt 0) { throw 'A collector is still running; stop it before -Recover -Apply.' }
            $recoverArgs += '--apply'
        }
        $code = Invoke-Python $recoverArgs
        exit $code
    }
    'Train' {
        Invoke-Train
    }
}
