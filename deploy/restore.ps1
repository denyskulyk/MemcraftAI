# Restores a dump into a NEW database. It never drops or overwrites one:
# the live name is refused and createdb fails if the database exists.
param(
    [Parameter(Mandatory = $true)][string]$Dump,
    [string]$Database = 'memcraft_restore_test'
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'common.ps1')

if ($Database -cnotmatch '^[a-z][a-z0-9_]{0,62}$') {
    throw 'Database must be lower-case letters, digits and underscores, starting with a letter.'
}
if (@($DbName, 'postgres', 'template0', 'template1') -contains $Database) {
    throw "Refusing to restore into '$Database'."
}
$file = Get-Item -LiteralPath $Dump
if ($file.PSIsContainer) { throw 'Dump must be a file.' }

Assert-PodmanRunning

$temp = "/tmp/restore-$([guid]::NewGuid().ToString('N')).dump"
# podman cp gets a ./ path: C:\... could be read as a container called C.
Push-Location -LiteralPath $file.DirectoryName
try {
    Invoke-Podman @('cp', "./$($file.Name)", "${Container}:$temp")
    Invoke-Podman @('exec', $Container, 'pg_restore', '--list', $temp) | Out-Null
    Invoke-Podman @('exec', $Container, 'createdb', '-U', $DbUser, '-T', 'template0', $Database)
    Invoke-Podman @('exec', $Container, 'pg_restore', '-U', $DbUser, '-d', $Database,
        '--no-owner', '--exit-on-error', '--single-transaction', $temp)
}
finally {
    Pop-Location
    & $Podman exec $Container rm -f $temp
    if ($LASTEXITCODE -ne 0) { Write-Warning "Could not remove $temp from $Container." }
}
Write-Host "Restored $($file.Name) into $Database."
Write-Host "Remove it later with: podman exec $Container dropdb -U $DbUser $Database"
