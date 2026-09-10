# Build the React UI and stage it where FastAPI serves it from.
#
# The bundle is committed so the Raspberry Pi needs no Node toolchain: the Pi's
# whole update cycle is `git pull` + `systemctl restart aerosense`. That means a
# UI change is not shipped until this script has run AND the result is
# committed.
#
# Usage:  pwsh scripts/build_ui.ps1

$ErrorActionPreference = "Stop"

$UiRepo = "C:\thesis_ui\badminton"
$Target = Join-Path $PSScriptRoot "..\app\static\ui"

if (-not (Test-Path $UiRepo)) { throw "UI repo not found at $UiRepo" }

Write-Host "==> building $UiRepo"
Push-Location $UiRepo
try {
    npm run build
    if ($LASTEXITCODE -ne 0) { throw "npm run build failed ($LASTEXITCODE)" }
}
finally { Pop-Location }

$Dist = Join-Path $UiRepo "dist"
if (-not (Test-Path (Join-Path $Dist "index.html"))) {
    throw "no index.html in $Dist - the build produced nothing"
}

Write-Host "==> staging into $Target"
if (Test-Path $Target) { Remove-Item -Recurse -Force $Target }
New-Item -ItemType Directory -Force -Path $Target | Out-Null
Copy-Item -Recurse -Force (Join-Path $Dist "*") $Target

$n = (Get-ChildItem -Recurse -File $Target | Measure-Object).Count
Write-Host "==> staged $n files"
Write-Host ""
Write-Host "Now commit the bundle, then on the Pi:"
Write-Host "  git pull && sudo systemctl restart aerosense"
