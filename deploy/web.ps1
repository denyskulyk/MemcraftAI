# Builds and (re)starts the web app container. Postgres must be running (postgres.ps1).
# -Build only builds. The container listens on 127.0.0.1:8000 through host networking.
param([switch]$Build)

Set-StrictMode -Version 2.0
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'common.ps1')

$Image = 'localhost/memcraft-web:latest'
$Name = 'memcraft-web'
$EnvFile = Join-Path $ProjectRoot '.env'
$Shared = Join-Path $ProjectRoot 'data\shared'

Assert-PodmanRunning
if (-not (Test-Path -LiteralPath $EnvFile)) { throw 'Missing .env. Run deploy\postgres.ps1 first.' }
New-Item -ItemType Directory -Force -Path $Shared | Out-Null

Invoke-Podman @('build', '--file', (Join-Path $PSScriptRoot 'Containerfile.web'), '--tag', $Image, $ProjectRoot)
if ($Build) { return }

if (Test-PodmanObject 'container' $Name) { Invoke-Podman @('rm', '--force', $Name) | Out-Null }
# The .env lines for Windows paths do not apply inside; the shared folder is mounted at /data/shared.
Invoke-Podman @(
    'run', '--detach', '--name', $Name, '--restart', 'unless-stopped',
    '--network', 'host',
    '--env-file', $EnvFile,
    '--env', 'MEMCRAFT_SHARED_DIR=/data/shared',
    '--env', 'MEMCRAFT_PRIVATE_DIR=',
    '--volume', "${Shared}:/data/shared",
    '--memory', '1g',
    $Image
) | Out-Null

$deadline = (Get-Date).AddSeconds(60)
while ((Get-Date) -lt $deadline) {
    try {
        $health = Invoke-RestMethod -Uri 'http://127.0.0.1:8000/health' -TimeoutSec 3
        Write-Host "memcraft-web is up: database $($health.database)."
        return
    }
    catch { Start-Sleep -Seconds 2 }
}
throw "memcraft-web did not answer on 127.0.0.1:8000. Check: podman logs $Name"
