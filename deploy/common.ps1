# Shared by the other deploy scripts. Dot-source it; it does nothing on its own.

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$MachineName = 'podman-machine-default'
$Container = 'memcraft-postgres'
$DbName = 'memcraft'
$DbUser = 'memcraft'

$podmanCommand = Get-Command podman.exe -CommandType Application -ErrorAction SilentlyContinue |
    Select-Object -First 1
if ($podmanCommand) {
    $Podman = $podmanCommand.Source
}
else {
    $Podman = Join-Path $env:LOCALAPPDATA 'Programs\Podman\podman.exe'
}

# Exit codes are checked by hand; "podman ... exists" uses 1 for "no".
$PSNativeCommandUseErrorActionPreference = $false

# stderr is never redirected here: Windows PowerShell 5.1 turns redirected
# stderr into a terminating error when ErrorActionPreference is Stop.
function Invoke-Podman([string[]]$Arguments) {
    & $Podman $Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "podman $($Arguments[0]) $($Arguments[1]) failed with exit code $LASTEXITCODE."
    }
}

# Kind is container, secret or volume.
function Test-PodmanObject([string]$Kind, [string]$Name) {
    & $Podman $Kind exists $Name
    switch ($LASTEXITCODE) {
        0 { return $true }
        1 { return $false }
        default { throw "podman $Kind exists failed with exit code $LASTEXITCODE." }
    }
}

function Assert-PodmanRunning {
    if (Test-Path -LiteralPath $Podman) {
        $rootless = & $Podman info --format '{{.Host.Security.Rootless}}'
        if ($LASTEXITCODE -eq 0) {
            if ("$rootless".Trim() -ne 'true') {
                Write-Host 'The active Podman connection is not rootless. These scripts expect the rootless machine.'
                exit 1
            }
            return
        }
    }
    Write-Host "Podman is not reachable. Start it yourself with: podman machine start $MachineName"
    exit 1
}
