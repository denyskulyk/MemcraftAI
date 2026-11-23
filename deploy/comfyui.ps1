# The app's own headless ComfyUI on 127.0.0.1:8199, next to the ComfyUI on 8188, which it never touches.
# It runs the ComfyUI install of Comfy Desktop with its python and its model folders, but has its own
# input, output, temp and user folders under data\comfy.
#
#   pwsh -File deploy\comfyui.ps1 start | stop | status
#
# stop ends only processes whose command line runs main.py with --port 8199.
param(
    [Parameter(Mandatory = $true, Position = 0)][ValidateSet('start', 'stop', 'status')][string]$Action,
    [ValidateRange(10, 900)][int]$TimeoutSeconds = 240,
    # The ComfyUI install of Comfy Desktop (holds ComfyUI\main.py).
    [string]$ComfyRoot = (Join-Path $env:LOCALAPPDATA 'Comfy-Desktop\ComfyUI-Installs\ComfyUI'),
    # Comfy Desktop's model paths; empty means the only .yaml in its instance-model-paths folder.
    [string]$ModelPaths = ''
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'common.ps1')

$Port = 8199
$Url = "http://127.0.0.1:$Port"
$DataDir = Join-Path $ProjectRoot 'data\comfy'
$LogFile = Join-Path $DataDir 'server.log'

function Get-OwnProcess {
    # Only our instance: main.py and the port 8199 in the same command line.
    @(Get-CimInstance Win32_Process | Where-Object {
            $_.CommandLine -and $_.CommandLine -match '\bmain\.py\b' -and $_.CommandLine -match "--port\s+$Port\b"
        })
}

function Test-Up {
    try {
        $response = Invoke-WebRequest -Uri "$Url/system_stats" -TimeoutSec 3 -UseBasicParsing
        return $response.StatusCode -eq 200
    }
    catch {
        return $false
    }
}

function Resolve-ModelPaths {
    if ($ModelPaths) { return $ModelPaths }
    $folder = Join-Path $env:APPDATA 'Comfy Desktop\instance-model-paths'
    $found = @(Get-ChildItem -LiteralPath $folder -Filter '*.yaml' -File -ErrorAction SilentlyContinue)
    if ($found.Count -ne 1) {
        throw "Expected one model paths file in $folder, found $($found.Count). Pass -ModelPaths."
    }
    return $found[0].FullName
}

function Start-Comfy {
    if (Test-Up) {
        if (@(Get-OwnProcess).Count -eq 0) { throw "Something else answers on port $Port." }
        Write-Host "ComfyUI already runs on $Url."
        return
    }
    $python = Join-Path $ComfyRoot 'ComfyUI\.venv\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $python)) { throw "No ComfyUI python at $python." }
    $yaml = Resolve-ModelPaths
    $dirs = @{}
    foreach ($name in 'input', 'output', 'temp', 'user') {
        $dirs[$name] = Join-Path $DataDir $name
        New-Item -ItemType Directory -Force -Path $dirs[$name] | Out-Null
    }
    $arguments = @(
        "`"$python`"", '-s', 'ComfyUI\main.py',
        '--listen', '127.0.0.1', '--port', "$Port", '--disable-auto-launch',
        '--extra-model-paths-config', "`"$yaml`"",
        '--input-directory', "`"$($dirs.input)`"", '--output-directory', "`"$($dirs.output)`"",
        '--temp-directory', "`"$($dirs.temp)`"", '--user-directory', "`"$($dirs.user)`"",
        '--database-url', 'sqlite:///:memory:'
    ) -join ' '
    # Started through WMI, so the server outlives this shell and any job object it runs in.
    $commandLine = "cmd.exe /d /c `"$arguments >> `"$LogFile`" 2>&1`""
    $startup = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ ShowWindow = [uint16]0 }
    $result = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
        CommandLine = $commandLine; CurrentDirectory = $ComfyRoot; ProcessStartupInformation = $startup
    }
    if ($result.ReturnValue -ne 0) { throw "Could not start ComfyUI (Win32_Process.Create returned $($result.ReturnValue))." }
    $watch = [Diagnostics.Stopwatch]::StartNew()
    while (-not (Test-Up)) {
        if (@(Get-OwnProcess).Count -eq 0) { throw "ComfyUI exited while starting; see $LogFile." }
        if ($watch.Elapsed.TotalSeconds -gt $TimeoutSeconds) { throw "ComfyUI did not answer within $TimeoutSeconds s; see $LogFile." }
        Start-Sleep -Seconds 1
    }
    Write-Host ("ComfyUI is up on {0} after {1:N0} s." -f $Url, $watch.Elapsed.TotalSeconds)
}

function Stop-Comfy {
    $own = @(Get-OwnProcess)
    if ($own.Count -eq 0) {
        Write-Host 'ComfyUI on 8199 is not running.'
        return
    }
    foreach ($process in $own) {
        Stop-Process -Id $process.ProcessId -Force -ErrorAction SilentlyContinue
    }
    $watch = [Diagnostics.Stopwatch]::StartNew()
    while (@(Get-OwnProcess).Count -gt 0 -or (Test-Up)) {
        if ($watch.Elapsed.TotalSeconds -gt 30) { throw 'ComfyUI on 8199 did not stop within 30 s.' }
        Start-Sleep -Milliseconds 500
    }
    Write-Host "Stopped ComfyUI on $Url."
}

switch ($Action) {
    'start' { Start-Comfy }
    'stop' { Stop-Comfy }
    'status' {
        $own = @(Get-OwnProcess)
        $up = Test-Up
        Write-Host ("ComfyUI on {0}: {1}, {2} process(es)." -f $Url, $(if ($up) { 'answers' } else { 'does not answer' }), $own.Count)
        if (-not $up) { exit 1 }
    }
}
