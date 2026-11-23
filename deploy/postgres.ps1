# Creates and starts Postgres 18 for MemcraftAI in the rootless Podman machine.
# Safe to run again. Never starts the Podman machine itself.
param([ValidateRange(10, 3600)][int]$TimeoutSeconds = 180)

Set-StrictMode -Version 2.0
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'common.ps1')

$Image = 'docker.io/library/postgres:18'
$Secret = 'memcraft-pg-password'
$Volume = 'memcraft-pgdata'
$EnvFile = Join-Path $ProjectRoot '.env'
$EnvExample = Join-Path $ProjectRoot '.env.example'

function New-EnvFile {
    # A new password cannot match a secret or database that already exists.
    if ((Test-PodmanObject 'secret' $Secret) -or (Test-PodmanObject 'volume' $Volume)) {
        throw ".env is missing, but $Secret or $Volume already exists. Restore .env with the original password."
    }
    $template = [IO.File]::ReadAllText($EnvExample)
    if (-not $template.Contains('memcraft:CHANGE_ME@')) {
        throw '.env.example has no CHANGE_ME password to replace.'
    }
    $bytes = New-Object byte[] 24
    $rng = [Security.Cryptography.RandomNumberGenerator]::Create()
    try { $rng.GetBytes($bytes) } finally { $rng.Dispose() }
    $newPassword = [Convert]::ToBase64String($bytes).Replace('+', '-').Replace('/', '_')
    # WriteAllText writes UTF-8 without a BOM, which the .env reader needs.
    [IO.File]::WriteAllText($EnvFile, $template.Replace('memcraft:CHANGE_ME@', "memcraft:$newPassword@"))
    Write-Host 'Created .env with a random password.'
}

function Read-EnvPassword {
    $lines = @([IO.File]::ReadAllLines($EnvFile) | Where-Object { $_ -match '^\s*MEMCRAFT_DATABASE_URL\s*=' })
    if ($lines.Count -ne 1) { throw '.env must have exactly one MEMCRAFT_DATABASE_URL line.' }
    $pattern = '^\s*MEMCRAFT_DATABASE_URL\s*=\s*["'']?postgresql\+psycopg://memcraft:([^@/\s]+)@127\.0\.0\.1:5432/memcraft["'']?\s*$'
    if ($lines[0] -notmatch $pattern) {
        throw 'MEMCRAFT_DATABASE_URL in .env must be postgresql+psycopg://memcraft:PASSWORD@127.0.0.1:5432/memcraft'
    }
    $found = [Uri]::UnescapeDataString($Matches[1])
    if ($found -match 'CHANGE_ME') { throw 'Replace CHANGE_ME in .env with a real password first.' }
    return $found
}

Assert-PodmanRunning

# 1. Password: .env first, then the Podman secret.
if (-not (Test-Path -LiteralPath $EnvFile)) { New-EnvFile }
$password = Read-EnvPassword
if (Test-PodmanObject 'secret' $Secret) {
    Write-Host "Secret $Secret exists, left as is."
}
else {
    # --env reads the value from this process environment, so it is not on a
    # command line. A pipe would add a CRLF to the password.
    $env:MEMCRAFT_PG_SECRET = $password
    try {
        Invoke-Podman @('secret', 'create', '--env', $Secret, 'MEMCRAFT_PG_SECRET') | Out-Null
    }
    finally {
        Remove-Item Env:MEMCRAFT_PG_SECRET
    }
    Write-Host "Created secret $Secret."
}

# 2. Container. The named volume is created by podman run when it is missing.
if (Test-PodmanObject 'container' $Container) {
    $state = "$(Invoke-Podman @('container', 'inspect', '--format', '{{.State.Status}}', $Container))".Trim()
    if ($state -ne 'running') { Invoke-Podman @('start', $Container) | Out-Null }
    Write-Host "Container $Container exists."
}
else {
    Invoke-Podman @(
        'run', '--detach', '--name', $Container,
        '--restart', 'unless-stopped',
        '--publish', '127.0.0.1:5432:5432',
        # From 18 on the data lives in a versioned folder below this path.
        '--volume', "${Volume}:/var/lib/postgresql",
        '--secret', $Secret,
        '--env', "POSTGRES_USER=$DbUser",
        '--env', "POSTGRES_DB=$DbName",
        '--env', "POSTGRES_PASSWORD_FILE=/run/secrets/$Secret",
        '--memory', '512m', '--shm-size', '128m',
        '--stop-timeout', '30',
        # -h 127.0.0.1: the first-run setup server has no TCP, so it never counts as healthy.
        '--health-cmd', "pg_isready -h 127.0.0.1 -U $DbUser -d $DbName",
        '--health-interval', '30s', '--health-timeout', '5s', '--health-retries', '3',
        '--health-start-period', '60s',
        $Image,
        'postgres', '-c', 'shared_buffers=128MB', '-c', 'max_connections=30',
        '-c', 'work_mem=4MB', '-c', 'maintenance_work_mem=32MB'
    ) | Out-Null
    Write-Host "Created container $Container."
}

# 3. The database must be reachable from this PC only.
$ports = @(Invoke-Podman @('port', $Container))
if ($ports.Count -ne 1 -or "$($ports[0])".Trim() -ne '5432/tcp -> 127.0.0.1:5432') {
    throw "$Container is not published on 127.0.0.1:5432 only. Check: podman port $Container"
}

# 4. Wait. The check is run here too, so this does not depend on health timers.
$deadline = (Get-Date).AddSeconds($TimeoutSeconds)
while ($true) {
    & $Podman healthcheck run $Container | Out-Null
    $health = "$(Invoke-Podman @('container', 'inspect', '--format', '{{.State.Health.Status}}', $Container))".Trim()
    if ($health -eq 'healthy') { break }
    if ((Get-Date) -gt $deadline) {
        throw "$Container is '$health' after $TimeoutSeconds s. Check: podman logs $Container"
    }
    Start-Sleep -Seconds 2
}
Write-Host "$Container is healthy on 127.0.0.1:5432."

# Denys chose no autostart (2026-11-23). podman-restart.service is not an option:
# it would also start searxng, which is switched off on purpose. After a restart run this
# script again; it starts the existing container.
