# Stops MemcraftAI: the worker (after its current step), the app's ComfyUI on 8199, the web
# app and Postgres. The Podman machine keeps running, because Open WebUI uses it too.
# Used by memcraft-stop.bat.
Set-StrictMode -Version 2.0
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'common.ps1')

& (Join-Path $PSScriptRoot 'worker.ps1') stop
& (Join-Path $PSScriptRoot 'comfyui.ps1') stop
& $Podman info --format '{{.Host.Security.Rootless}}' *> $null
if ($LASTEXITCODE -eq 0) {
    foreach ($name in 'memcraft-web', $Container) {
        if (Test-PodmanObject 'container' $name) { Invoke-Podman @('stop', $name) | Out-Null }
    }
}
Write-Host 'MemcraftAI is stopped.'
