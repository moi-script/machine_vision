# Create Desktop and Start-menu shortcuts that launch AeroSense.
#
#   powershell -ExecutionPolicy Bypass -File scripts\install-windows-shortcut.ps1
#
# The shortcuts point at start-aerosense.bat, which starts the backend if
# needed and then opens the app window. Re-running this overwrites the
# existing shortcuts, so it is safe after moving the repo.
#
# This is separate from installing the PWA itself: to get the app into
# Chrome's own app list (and its Start-menu entry, taskbar pinning and
# standalone window chrome), open http://localhost:8000 in Chrome and use the
# install button in the address bar. Either route works; the shortcut is the
# one that guarantees the backend is running.

$ErrorActionPreference = "Stop"

$Root   = Split-Path -Parent $PSScriptRoot              # ...\setup
$Target = Join-Path $PSScriptRoot "start-aerosense.bat"
$Icon   = Join-Path $Root "app\static\ui\icon-192.png"

if (-not (Test-Path $Target)) { throw "launcher not found at $Target" }

# .lnk files cannot use a PNG as their icon, so convert the app icon to .ico
# once and keep it beside the launcher.
$IcoPath = Join-Path $PSScriptRoot "aerosense.ico"
if ((Test-Path $Icon) -and -not (Test-Path $IcoPath)) {
    try {
        Add-Type -AssemblyName System.Drawing
        $png = [System.Drawing.Image]::FromFile($Icon)
        $bmp = New-Object System.Drawing.Bitmap $png, 128, 128
        $hIcon = $bmp.GetHicon()
        $ico = [System.Drawing.Icon]::FromHandle($hIcon)
        $fs = [System.IO.File]::Create($IcoPath)
        $ico.Save($fs)
        $fs.Close(); $bmp.Dispose(); $png.Dispose()
        Write-Host "==> wrote $IcoPath"
    }
    catch {
        # A missing icon is cosmetic; never fail the whole install over it.
        Write-Warning "could not build the .ico ($_). Shortcuts will use the default icon."
    }
}

$shell = New-Object -ComObject WScript.Shell
$places = @(
    [System.Environment]::GetFolderPath("Desktop"),
    (Join-Path ([System.Environment]::GetFolderPath("StartMenu")) "Programs")
)

foreach ($dir in $places) {
    if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Force -Path $dir | Out-Null }
    $lnk = $shell.CreateShortcut((Join-Path $dir "AeroSense.lnk"))
    $lnk.TargetPath       = $Target
    $lnk.WorkingDirectory = $Root
    $lnk.Description      = "Start the AeroSense backend and open the app"
    $lnk.WindowStyle      = 7          # start minimised; the app window is what you look at
    if (Test-Path $IcoPath) { $lnk.IconLocation = $IcoPath }
    $lnk.Save()
    Write-Host "==> shortcut created in $dir"
}

Write-Host ""
Write-Host "Done. Launch AeroSense from the Desktop or Start menu."
Write-Host "To also install it as a Chrome app (its own window, taskbar icon):"
Write-Host "  open http://localhost:8000 in Chrome, then use the install button in the address bar."
