# Brings the database in .env to the newest schema.
Set-StrictMode -Version 2.0
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'common.ps1')

$python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw "Missing $python" }
if (-not (Test-Path -LiteralPath (Join-Path $ProjectRoot '.env'))) { throw 'Missing .env. Run deploy\postgres.ps1 first.' }

# alembic.ini and .env are found relative to the project root.
Push-Location -LiteralPath $ProjectRoot
try {
    & $python -m alembic upgrade head
    if ($LASTEXITCODE -ne 0) { throw "alembic failed with exit code $LASTEXITCODE." }
}
finally {
    Pop-Location
}
