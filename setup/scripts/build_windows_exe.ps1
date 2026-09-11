# Build the distributable Windows bundle.
#
#   pwsh scripts\build_windows_exe.ps1        (or powershell, from setup\)
#
# Produces dist\AeroSense\AeroSense.exe plus its dependencies, then zips the
# folder for release. Expect roughly 1.5 GB and a slow first build: torch
# alone is ~450 MB and openvino another ~230 MB.
#
# The zip is NOT committed - it is far over GitHub's 100 MB per-file limit and
# ships as a Release asset instead. Only this script and aerosense.spec live
# in git.

$ErrorActionPreference = "Stop"

$Setup = Split-Path -Parent $PSScriptRoot
Push-Location $Setup
try {
    # The bundle embeds app/static/ui, so a stale UI here becomes a stale UI
    # inside the exe with no way to tell from the outside.
    if (-not (Test-Path "app\static\ui\index.html")) {
        throw "app\static\ui is missing - run scripts\build_ui.ps1 first"
    }

    Write-Host "==> building (this takes a while)"
    python -m PyInstaller scripts\aerosense.spec --noconfirm --distpath dist --workpath build
    if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed ($LASTEXITCODE)" }

    $exe = "dist\AeroSense\AeroSense.exe"
    if (-not (Test-Path $exe)) { throw "no exe produced at $exe" }

    $size = [math]::Round(((Get-ChildItem -Recurse -File "dist\AeroSense" |
        Measure-Object -Property Length -Sum).Sum / 1GB), 2)
    Write-Host "==> built dist\AeroSense ($size GB)"

    $zip = "dist\AeroSense-window_v-win64.zip"
    if (Test-Path $zip) { Remove-Item $zip }
    Write-Host "==> compressing (slow)"
    Compress-Archive -Path "dist\AeroSense\*" -DestinationPath $zip -CompressionLevel Optimal

    $zipMb = [math]::Round(((Get-Item $zip).Length / 1MB), 0)
    Write-Host "==> $zip ($zipMb MB)"
    Write-Host ""
    Write-Host "Publish it with:"
    Write-Host "  gh release create window_v $zip --title 'AeroSense window_v' --notes '...'"
}
finally { Pop-Location }
