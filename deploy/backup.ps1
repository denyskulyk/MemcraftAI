# Dumps the live database to backups\memcraft-YYYYMMDD-HHmmss.dump and keeps the newest -Keep files.
# Old backups are only removed after the new one is in place.
param([ValidateRange(1, 1000)][int]$Keep = 14)

Set-StrictMode -Version 2.0
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'common.ps1')

Assert-PodmanRunning

$stamp = (Get-Date).ToString('yyyyMMdd-HHmmss', [Globalization.CultureInfo]::InvariantCulture)
$name = "memcraft-$stamp.dump"
$temp = "/tmp/$name"
$folder = Join-Path $ProjectRoot 'backups'
$partial = Join-Path $folder "$name.partial"
New-Item -ItemType Directory -Force -Path $folder | Out-Null

# podman cp gets a ./ path: C:\... could be read as a container called C.
Push-Location -LiteralPath $folder
try {
    Invoke-Podman @('exec', $Container, 'pg_dump', '-U', $DbUser, '-d', $DbName, '-Fc', '-f', $temp)
    Invoke-Podman @('exec', $Container, 'pg_restore', '--list', $temp) | Out-Null
    $size = [long]"$(Invoke-Podman @('exec', $Container, 'stat', '-c', '%s', $temp))".Trim()
    Invoke-Podman @('cp', "${Container}:$temp", "./$name.partial")
    if ((Get-Item -LiteralPath $partial).Length -ne $size) { throw 'The copied dump has the wrong size.' }
    Move-Item -LiteralPath $partial -Destination (Join-Path $folder $name)
}
finally {
    Pop-Location
    if (Test-Path -LiteralPath $partial) { Remove-Item -LiteralPath $partial }
    & $Podman exec $Container rm -f $temp
    if ($LASTEXITCODE -ne 0) { Write-Warning "Could not remove $temp from $Container." }
}
Write-Host "Saved $(Join-Path $folder $name) ($size bytes)."

$old = @(Get-ChildItem -LiteralPath $folder -File |
    Where-Object { $_.Name -cmatch '^memcraft-[0-9]{8}-[0-9]{6}\.dump$' } |
    Sort-Object Name -Descending |
    Select-Object -Skip $Keep)
foreach ($file in $old) {
    Remove-Item -LiteralPath $file.FullName
    Write-Host "Removed old backup $($file.Name)."
}
