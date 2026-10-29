# Locate a real Python interpreter without activating a shell environment.
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$candidates = @()
if ($env:AUDIOBOOK_PYTHON) {
    $candidates += [PSCustomObject]@{ Exe = $env:AUDIOBOOK_PYTHON; Prefix = @() }
}
$existing = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (Test-Path -LiteralPath $existing) {
    $candidates += [PSCustomObject]@{ Exe = $existing; Prefix = @() }
}
foreach ($command in @(Get-Command py -CommandType Application -ErrorAction SilentlyContinue)) {
    $candidates += [PSCustomObject]@{ Exe = $command.Source; Prefix = @('-3.12') }
}
foreach ($name in @('python', 'python3.12', 'python3')) {
    foreach ($command in @(Get-Command $name -CommandType Application -ErrorAction SilentlyContinue)) {
        if ($command.Source -notlike '*\Microsoft\WindowsApps\*') {
            $candidates += [PSCustomObject]@{ Exe = $command.Source; Prefix = @() }
        }
    }
}
foreach ($candidate in $candidates) {
    if (-not (Test-Path -LiteralPath $candidate.Exe)) { continue }
    $prefix = @($candidate.Prefix)
    try {
        & $candidate.Exe @prefix -c 'import sys,struct; sys.exit(0 if sys.version_info[:2]==(3,12) and struct.calcsize(chr(80))==8 else 1)' 2>$null
        if ($LASTEXITCODE -ne 0) { continue }
    } catch {
        continue
    }
    # pip/model downloads can legitimately write progress or warnings to stderr.
    # An installer failure must return once, not retry with another interpreter.
    $ErrorActionPreference = 'Continue'
    & $candidate.Exe @prefix (Join-Path $projectRoot 'installer.py') @args
    exit $LASTEXITCODE
}
Write-Host 'Python 3.12 64-bit was not found. Install it from https://www.python.org/downloads/windows/ then run installer.bat again.'
exit 1
