<#
One-time admin setup for MemcraftAI. Run it once, from an elevated PowerShell, as your own user.
Every step is skipped when it is already done, so running it again is safe.

  1. Git and ffmpeg (winget, only when missing)
  2. A WSL memory limit in .wslconfig; everything else in that file stays, and a backup is made
  3. A scheduled task that starts the Podman machine at boot, without stored password

It does not restart WSL, Podman, ComfyUI, llama-swap or the PC. The memory limit and the boot
task take effect at the next restart. Use -WhatIf to only see what it would do (no admin needed).
#>
[CmdletBinding(SupportsShouldProcess)]
param(
    [ValidateRange(3, 32)][int]$WslMemoryGB = 6
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Find-Tool([string]$Name, [string[]]$Extra) {
    # Read PATH from the registry: this window may be older than the installation.
    $paths = @()
    foreach ($scope in 'Machine', 'User') {
        $value = [Environment]::GetEnvironmentVariable('Path', $scope)
        if ($value) { $paths += $value -split ';' }
    }
    $paths += $Extra
    foreach ($dir in $paths) {
        if (-not $dir) { continue }
        $file = Join-Path ([Environment]::ExpandEnvironmentVariables($dir.Trim('"'))) "$Name.exe"
        if (Test-Path -LiteralPath $file) { return $file }
    }
    return $null
}

function Install-IfMissing([string]$Name, [string]$Id, [string[]]$Extra) {
    $found = Find-Tool $Name $Extra
    if ($found) {
        Write-Host "[skip] $Name is already here: $found"
        return
    }
    if (-not $PSCmdlet.ShouldProcess("$Id (winget, for all users)", 'Install')) { return }
    & winget install --id $Id --exact --source winget --scope machine --silent `
        --accept-source-agreements --accept-package-agreements
    if ($LASTEXITCODE -ne 0) { throw "winget could not install $Id (code $LASTEXITCODE)." }
    $found = Find-Tool $Name $Extra
    if (-not $found) { throw "$Id was installed but $Name.exe is not on PATH yet." }
    Write-Host "[done] $Name installed: $found"
}

# Returns the new text, or $null when a memory limit is already set.
function Add-WslMemoryLimit([string]$Text, [int]$GB) {
    $lines = @()
    if ($Text) { $lines = @(($Text -replace "`r`n", "`n").TrimEnd("`n") -split "`n") }
    $section = ''
    $header = -1
    for ($i = 0; $i -lt $lines.Count; $i++) {
        if ($lines[$i] -match '^\s*\[([^\]]+)\]') {
            $section = $Matches[1].Trim().ToLowerInvariant()
            if ($section -eq 'wsl2' -and $header -lt 0) { $header = $i }
        } elseif ($section -eq 'wsl2' -and $lines[$i] -match '^\s*memory\s*=') {
            return $null
        }
    }
    $new = New-Object System.Collections.Generic.List[string]
    foreach ($line in $lines) { $new.Add($line) }
    if ($header -ge 0) {
        $new.Insert($header + 1, "memory=${GB}GB")
    } else {
        $new.Add('[wsl2]')
        $new.Add("memory=${GB}GB")
    }
    return ($new -join "`r`n") + "`r`n"
}

$admin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()
).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $admin -and -not $WhatIfPreference) {
    Write-Host 'Please run this from an elevated PowerShell (Run as administrator).'
    exit 1
}
$podman = Join-Path $env:LOCALAPPDATA 'Programs\Podman\podman.exe'
if (-not (Test-Path -LiteralPath $podman)) {
    throw "Podman was not found at $podman. Run this as the Windows user that owns Podman."
}

Write-Host '1. Git and ffmpeg'
$links = "$env:ProgramFiles\WinGet\Links", "$env:LOCALAPPDATA\Microsoft\WinGet\Links"
Install-IfMissing 'git' 'Git.Git' @(
    "$env:ProgramFiles\Git\cmd", "$env:LOCALAPPDATA\Programs\Git\cmd")
Install-IfMissing 'ffmpeg' 'Gyan.FFmpeg.Essentials' $links

Write-Host '2. WSL memory limit'
$config = Join-Path $env:USERPROFILE '.wslconfig'
$old = ''
if (Test-Path -LiteralPath $config) { $old = [IO.File]::ReadAllText($config) }
$new = Add-WslMemoryLimit $old $WslMemoryGB
if ($null -eq $new) {
    Write-Host "[skip] $config already sets a memory limit."
} elseif ($PSCmdlet.ShouldProcess($config, "Add memory=${WslMemoryGB}GB")) {
    if ($old) {
        $backup = "$config.bak-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
        Copy-Item -LiteralPath $config -Destination $backup
        Write-Host "[done] backup: $backup"
    }
    [IO.File]::WriteAllText($config, $new, (New-Object Text.UTF8Encoding $false))
    Write-Host "[done] memory=${WslMemoryGB}GB, used from the next restart of WSL"
}

Write-Host '3. Podman machine at boot'
$taskName = 'MemcraftAI podman machine at boot'
if (Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue) {
    Write-Host "[skip] task '$taskName' already exists."
} elseif ($PSCmdlet.ShouldProcess($taskName, 'Register scheduled task')) {
    # S4U: runs as you before anyone logs in, stores no password, and is not elevated.
    # Your existing logon task 'LocalAI podman machine' stays as the fallback.
    $user = [Security.Principal.WindowsIdentity]::GetCurrent().Name
    Register-ScheduledTask -TaskName $taskName `
        -Action (New-ScheduledTaskAction -Execute $podman -Argument 'machine start') `
        -Trigger (New-ScheduledTaskTrigger -AtStartup) `
        -Principal (New-ScheduledTaskPrincipal -UserId $user -LogonType S4U -RunLevel Limited) `
        -Settings (New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew `
            -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
            -ExecutionTimeLimit (New-TimeSpan -Minutes 15)) | Out-Null
    Write-Host "[done] task '$taskName' registered."
}

Write-Host 'Finished. Restart the PC when no job is running, then check Open WebUI still opens.'
