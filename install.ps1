<#
.SYNOPSIS
    Install Grad on Windows: a virtual environment, the dependencies, and a
    shortcut that opens the workspace without a console window.

.DESCRIPTION
    This does not produce a single self-contained .exe, and the reason is worth
    stating rather than discovering. `claude-agent-sdk` does not call the API
    directly -- it spawns the `claude` CLI and speaks stream-JSON over its
    stdio. So the agent's brain is a separate native binary holding its own
    subscription auth, and no amount of PyInstaller bundling can absorb it. The
    same is true of JupyterLab, which is a Python environment with kernels and
    is the point of the app rather than an implementation detail of it.

    What is achievable, and what this does, is an install with one visible
    entry point: a Start Menu shortcut that launches `pythonw.exe` (no console),
    holds the single-instance lock, and lives in the notification area until you
    quit it.

.PARAMETER InstallExtras
    Extras to install. Defaults to the set the desktop app needs.

.PARAMETER NoShortcut
    Skip creating the Start Menu and Desktop shortcuts.

.PARAMETER Python
    Python to build the environment with. Defaults to whatever `py -3` or
    `python` resolves to.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\install.ps1
#>

[CmdletBinding()]
param(
    [string] $InstallExtras = "ui,notebook,agent,lab",
    [switch] $NoShortcut,
    [string] $Python = ""
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvDir = Join-Path $Root ".venv"
$AppData = Join-Path $env:LOCALAPPDATA "Grad"

function Write-Step($message) { Write-Host "==> $message" -ForegroundColor Cyan }
function Write-Warn($message) { Write-Host "  ! $message" -ForegroundColor Yellow }
function Write-Ok($message)   { Write-Host "  + $message" -ForegroundColor Green }

# --------------------------------------------------------------------------
# 1. Python
# --------------------------------------------------------------------------
Write-Step "Locating Python 3.11 or newer"

function Resolve-Python {
    param([string] $Explicit)
    if ($Explicit) { return $Explicit }
    # `py -3` first: the launcher knows about every install, where `python` on
    # PATH may be the Microsoft Store stub that only opens the Store page.
    if (Get-Command py -ErrorAction SilentlyContinue) {
        $probe = & py -3 -c "import sys; print(sys.executable)" 2>$null
        if ($LASTEXITCODE -eq 0 -and $probe) { return $probe.Trim() }
    }
    if (Get-Command python -ErrorAction SilentlyContinue) { return "python" }
    return ""
}

$PythonExe = Resolve-Python -Explicit $Python
if (-not $PythonExe) {
    throw "No Python found. Install 3.11+ from https://www.python.org/downloads/ and re-run."
}

$Version = & $PythonExe -c "import sys; print('%d.%d' % sys.version_info[:2])"
$Parts = $Version.Split('.')
if ([int]$Parts[0] -lt 3 -or ([int]$Parts[0] -eq 3 -and [int]$Parts[1] -lt 11)) {
    throw "Python $Version is too old; Grad needs 3.11 or newer."
}
Write-Ok "Python $Version at $PythonExe"

# --------------------------------------------------------------------------
# 2. The environment
# --------------------------------------------------------------------------
Write-Step "Creating the virtual environment"
if (-not (Test-Path $VenvDir)) {
    & $PythonExe -m venv $VenvDir
    Write-Ok "created $VenvDir"
} else {
    Write-Ok "reusing $VenvDir"
}

$VenvPython  = Join-Path $VenvDir "Scripts\python.exe"
$VenvPythonW = Join-Path $VenvDir "Scripts\pythonw.exe"
if (-not (Test-Path $VenvPython)) { throw "The virtual environment has no python.exe: $VenvPython" }

Write-Step "Installing Grad and its dependencies (this takes a few minutes)"
& $VenvPython -m pip install --upgrade pip --quiet
# Editable, because this repository *is* the install: the workspace, the
# prompts and the skills are read from here at runtime.
& $VenvPython -m pip install -e "$($Root)[$($InstallExtras)]"
if ($LASTEXITCODE -ne 0) { throw "pip install failed." }
Write-Ok "installed extras: $InstallExtras"

# --------------------------------------------------------------------------
# 3. The things this installer cannot install
# --------------------------------------------------------------------------
Write-Step "Checking what has to be present but cannot be vendored"

$Claude = Get-Command claude -ErrorAction SilentlyContinue
if ($Claude) {
    Write-Ok "claude CLI at $($Claude.Source)"
} else {
    Write-Warn 'The "claude" CLI was not found on PATH.'
    Write-Warn "The agent spawns it as a subprocess -- without it, the workspace"
    Write-Warn "opens but no turn can run. Install it, then re-run this script:"
    Write-Warn "    npm install -g @anthropic-ai/claude-code"
}

# WebView2 is what pywebview renders into. It ships with Windows 11 and recent
# 10, so this is a check rather than a step -- but a missing runtime shows up as
# a window that opens blank, which is not a self-explaining failure.
$WebView2 = @(
    "HKLM:\SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}",
    "HKLM:\SOFTWARE\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}"
) | Where-Object { Test-Path $_ } | Select-Object -First 1

if ($WebView2) {
    Write-Ok "WebView2 runtime present"
} else {
    Write-Warn "WebView2 runtime not detected. The desktop window needs it:"
    Write-Warn "    https://developer.microsoft.com/microsoft-edge/webview2/"
    Write-Warn "Without it, run the browser fallback: python agent.py --ui (then open the port)."
}

# --------------------------------------------------------------------------
# 4. The shortcut
# --------------------------------------------------------------------------
if (-not $NoShortcut) {
    Write-Step "Creating shortcuts"

    New-Item -ItemType Directory -Force -Path $AppData | Out-Null
    $IconPath = Join-Path $AppData "grad.ico"
    # Drawn by the app itself, so the shortcut and the notification area cannot
    # show two different marks. See ui/desktop.py:write_icon.
    & $VenvPython -c "from ui import desktop; desktop.write_icon(r'$IconPath')" 2>$null
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $IconPath)) {
        Write-Warn "could not render the icon; the shortcut will use Python's"
        $IconPath = $VenvPythonW
    }

    $Shell = New-Object -ComObject WScript.Shell
    $Targets = @(
        (Join-Path ([Environment]::GetFolderPath('Programs')) "Grad.lnk"),
        (Join-Path ([Environment]::GetFolderPath('Desktop'))  "Grad.lnk")
    )
    foreach ($LinkPath in $Targets) {
        $Link = $Shell.CreateShortcut($LinkPath)
        # pythonw.exe, not python.exe: the console-subsystem interpreter would
        # put a black window behind the app for its whole lifetime. This is the
        # same reasoning core/spawn.py applies to every child process.
        $Link.TargetPath       = $VenvPythonW
        $Link.Arguments        = "`"$(Join-Path $Root 'agent.py')`" --ui"
        $Link.WorkingDirectory = $Root
        $Link.IconLocation     = $IconPath
        $Link.Description      = "Grad - a personal research agent for mathematics and machine learning"
        $Link.Save()
        Write-Ok $LinkPath
    }
}

# --------------------------------------------------------------------------
# 5. Where things live
# --------------------------------------------------------------------------
Write-Host ""
Write-Step "Done"
Write-Host "  Workspace (your research, versioned):  $Root"
Write-Host "  App state (layouts, logs, Lab token):  $AppData"
Write-Host ""
Write-Host "  Start it from the Start Menu, or:"
Write-Host "      $VenvPythonW `"$(Join-Path $Root 'agent.py')`" --ui"
Write-Host ""
Write-Host "  It opens on port 8080, or the next free port above it, and stays in"
Write-Host "  the notification area when you close the window. Quit from there."
Write-Host ""
