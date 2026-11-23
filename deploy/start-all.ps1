# Starts MemcraftAI by hand (nothing starts at boot): the Podman machine if needed,
# Postgres, the web app and the worker. The worker starts ComfyUI on 8199 and Gemma
# in llama-swap itself when a job needs them. Used by memcraft-start.bat.
Set-StrictMode -Version 2.0
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'common.ps1')

& $Podman info --format '{{.Host.Security.Rootless}}' *> $null
if ($LASTEXITCODE -ne 0) {
    Write-Host 'Starting the Podman machine...'
    Invoke-Podman @('machine', 'start', $MachineName) | Out-Null
}
& (Join-Path $PSScriptRoot 'postgres.ps1')
& (Join-Path $PSScriptRoot 'web.ps1') -StartOnly
& (Join-Path $PSScriptRoot 'worker.ps1') start
Write-Host ''
Write-Host 'MemcraftAI is running: https://gzlebsokcj.tailc507bc.ts.net:8443 (on this PC: http://127.0.0.1:8000)'
