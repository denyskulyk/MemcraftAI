# The worker (python -m memcraft.worker) as a hidden background process, its log in data\logs\worker.log.
#
#   pwsh -File deploy\worker.ps1 start | stop | status
#
# stop asks the worker to finish its current steps (a stop file it watches), waits up to
# -StopSeconds, then ends the process. It also ends the worker's own llama-server on port 8098.
# It never touches ComfyUI on 8188, llama-swap or any scheduled task; ComfyUI on 8199 keeps
# running (deploy\comfyui.ps1 stop ends it). No scheduled task is registered here.
param(
    [Parameter(Mandatory = $true, Position = 0)][ValidateSet('start', 'stop', 'status')][string]$Action,
    [ValidateRange(5, 3600)][int]$StopSeconds = 120,
    # Extra arguments for the worker, such as '--gemma-idle-minutes 10'.
    [string]$WorkerArgs = ''
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'common.ps1')

$Python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
$LogDir = Join-Path $ProjectRoot 'data\logs'
$LogFile = Join-Path $LogDir 'worker.log'
$StopFile = Join-Path $ProjectRoot 'data\worker.stop'

function Get-WorkerProcess {
    @(Get-CimInstance Win32_Process | Where-Object {
            $_.CommandLine -and $_.CommandLine -match '-m\s+memcraft\.worker\b' -and $_.Name -like 'python*'
        })
}

function Get-GemmaProcess {
    @(Get-CimInstance Win32_Process -Filter "Name='llama-server.exe'" | Where-Object {
            $_.CommandLine -match '--port\s+8098\b'
        })
}

function Start-Worker {
    $running = @(Get-WorkerProcess)
    if ($running.Count -gt 0) {
        Write-Host "The worker already runs (process $($running[0].ProcessId))."
        return
    }
    if (-not (Test-Path -LiteralPath $Python)) { throw "No Python at $Python; set up .venv first." }
    New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
    # The file store never creates its folders (a missing one counts as not mounted), so the
    # first start creates the two that .env names.
    foreach ($line in Get-Content -LiteralPath (Join-Path $ProjectRoot '.env')) {
        if ($line -match '^\s*MEMCRAFT_(SHARED|PRIVATE)_DIR\s*=\s*(.+?)\s*$') {
            New-Item -ItemType Directory -Force -Path $Matches[2].Trim('"', "'") | Out-Null
        }
    }
    Remove-Item -LiteralPath $StopFile -ErrorAction SilentlyContinue
    $arguments = "`"$Python`" -m memcraft.worker --stop-file `"$StopFile`" $WorkerArgs".Trim()
    # Through WMI, so the worker outlives this shell; cmd only redirects the output to the log.
    $commandLine = "cmd.exe /d /c `"$arguments >> `"$LogFile`" 2>&1`""
    $startup = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ ShowWindow = [uint16]0 }
    $result = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
        CommandLine = $commandLine; CurrentDirectory = $ProjectRoot; ProcessStartupInformation = $startup
    }
    if ($result.ReturnValue -ne 0) { throw "Could not start the worker (Win32_Process.Create returned $($result.ReturnValue))." }
    $watch = [Diagnostics.Stopwatch]::StartNew()
    while (@(Get-WorkerProcess).Count -eq 0) {
        if ($watch.Elapsed.TotalSeconds -gt 20) { throw "The worker did not start; see $LogFile." }
        Start-Sleep -Milliseconds 500
    }
    Start-Sleep -Seconds 3
    if (@(Get-WorkerProcess).Count -eq 0) { throw "The worker exited at once; see $LogFile." }
    Write-Host "The worker runs; log: $LogFile"
}

function Stop-Worker {
    $running = @(Get-WorkerProcess)
    if ($running.Count -gt 0) {
        New-Item -ItemType File -Force -Path $StopFile | Out-Null
        $watch = [Diagnostics.Stopwatch]::StartNew()
        while (@(Get-WorkerProcess).Count -gt 0 -and $watch.Elapsed.TotalSeconds -lt $StopSeconds) {
            Start-Sleep -Seconds 1
        }
        foreach ($process in @(Get-WorkerProcess)) {
            Write-Host "The worker did not stop within $StopSeconds s; ending process $($process.ProcessId)."
            Stop-Process -Id $process.ProcessId -Force -ErrorAction SilentlyContinue
        }
        Remove-Item -LiteralPath $StopFile -ErrorAction SilentlyContinue
        Write-Host 'The worker stopped.'
    }
    else {
        Write-Host 'The worker is not running.'
    }
    foreach ($process in @(Get-GemmaProcess)) {
        Stop-Process -Id $process.ProcessId -Force -ErrorAction SilentlyContinue
        Write-Host "Stopped the worker's llama-server (process $($process.ProcessId))."
    }
}

switch ($Action) {
    'start' { Start-Worker }
    'stop' { Stop-Worker }
    'status' {
        $running = @(Get-WorkerProcess)
        $gemma = @(Get-GemmaProcess)
        Write-Host ("Worker: {0}; Gemma llama-server on 8098: {1}." -f $(if ($running.Count) { "running (process $($running[0].ProcessId))" } else { 'not running' }), $(if ($gemma.Count) { 'running' } else { 'not running' }))
        if ($running.Count -eq 0) { exit 1 }
    }
}
