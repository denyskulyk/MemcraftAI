# Runs pytest against a scratch database whose name ends in _test. Everything in it is dropped.
# Example: pwsh -ExecutionPolicy Bypass -File .\deploy\test.ps1 -Database memcraft_test tests\test_storage.py
param(
    [ValidatePattern('^[a-z][a-z0-9_]{0,50}_test$')][string]$Database = 'memcraft_test',
    [Parameter(ValueFromRemainingArguments = $true)][string[]]$PytestArgs
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'common.ps1')

Assert-PodmanRunning

$found = Invoke-Podman @('exec', $Container, 'psql', '-U', $DbUser, '-d', $DbName, '-Atc',
    "SELECT 1 FROM pg_database WHERE datname = '$Database'")
if ("$found".Trim() -ne '1') {
    Invoke-Podman @('exec', $Container, 'createdb', '-U', $DbUser, $Database)
}

# The password stays in .env and in this process; it is never printed.
$line = @([IO.File]::ReadAllLines((Join-Path $ProjectRoot '.env')) |
    Where-Object { $_ -match '^\s*MEMCRAFT_DATABASE_URL\s*=' })[0]
$url = ($line -replace '^\s*MEMCRAFT_DATABASE_URL\s*=\s*', '').Trim().Trim('"', "'")
$env:MEMCRAFT_TEST_DATABASE_URL = $url -replace '/memcraft$', "/$Database"
Push-Location -LiteralPath $ProjectRoot
try {
    & (Join-Path $ProjectRoot '.venv\Scripts\python.exe') -m pytest -q -W 'ignore::DeprecationWarning' @PytestArgs
    $code = $LASTEXITCODE
}
finally {
    Pop-Location
    Remove-Item Env:MEMCRAFT_TEST_DATABASE_URL
}
exit $code
