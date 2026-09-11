@echo off
REM ---------------------------------------------------------------------------
REM Start AeroSense on Windows and open the app window.
REM
REM The installed PWA is only a window onto http://localhost:8000 - it cannot
REM start the Python backend, so clicking its icon with the server down gives a
REM connection error. This script closes that gap: it starts the server if it
REM is not already answering, waits until the API really responds, and only
REM then opens the app.
REM
REM WHY IT PROBES THE API INSTEAD OF THE PORT
REM "Is something listening on 8000?" is the wrong question. Another service
REM can hold that port - this was found on the dev machine, where an unrelated
REM `uvicorn server:app --reload --port 8000` was bound to 127.0.0.1 while
REM AeroSense bound 0.0.0.0, and Windows allowed both. A port check would have
REM reported "already running" and opened the app window onto the wrong
REM service. Asking /api/health who is there is the only reliable test.
REM
REM Create Desktop and Start-menu shortcuts to this file with:
REM     powershell -ExecutionPolicy Bypass -File scripts\install-windows-shortcut.ps1
REM ---------------------------------------------------------------------------
setlocal

set "ROOT=%~dp0.."
set "URL=http://localhost:8000"
set "HEALTH=http://127.0.0.1:8000/api/health"
set "WAIT_CEILING=60"

REM --- is AeroSense already answering on 8000? -------------------------------
REM AeroSense's /api/health returns a JSON object containing "engine". Any
REM other service answering on that port will not.
call :probe
if "%PROBE%"=="aerosense" (
  echo AeroSense backend already running.
  goto open
)
if "%PROBE%"=="foreign" goto occupied

echo Starting AeroSense backend...
REM /min so the console does not steal focus. Visible rather than hidden: if
REM Python fails to start, the operator needs somewhere to read the traceback.
start "AeroSense backend" /min /d "%ROOT%" cmd /c python run_server.py

echo Waiting for the API...
set /a ELAPSED=0
:waitloop
call :probe
if "%PROBE%"=="aerosense" (
  echo   API ready after %ELAPSED%s.
  goto open
)
if "%PROBE%"=="foreign" goto occupied
if %ELAPSED% GEQ %WAIT_CEILING% (
  echo   API did not answer within %WAIT_CEILING%s - opening anyway.
  echo   If the window shows an error, read the backend console for a traceback.
  goto open
)
timeout /t 1 /nobreak >nul
set /a ELAPSED+=1
goto waitloop

:occupied
echo.
echo   ERROR: port 8000 is held by a different service.
echo   Its /api/health did not look like AeroSense, so this script will not
echo   open a window onto it. Stop whatever else is using port 8000, then
echo   run this again. To see what holds it:
echo.
echo       netstat -ano ^| findstr :8000
echo       tasklist /fi "pid eq <the PID from the last column>"
echo.
pause
exit /b 1

:open
REM --app gives a chrome-less window whether or not the PWA has been installed.
start "" chrome.exe --app=%URL%
if errorlevel 1 (
  echo Chrome not found on PATH - opening in your default browser instead.
  start "" %URL%
)
endlocal
exit /b 0

REM --- probe: sets PROBE to aerosense ^| foreign ^| down ----------------------
:probe
set "PROBE=down"
curl -fsS --connect-timeout 2 --max-time 5 "%HEALTH%" 2>nul | findstr /C:"engine" >nul 2>&1
if not errorlevel 1 (
  set "PROBE=aerosense"
  exit /b 0
)
REM Answered, but not with AeroSense's health payload.
curl -sS -o nul --connect-timeout 2 --max-time 5 "%HEALTH%" >nul 2>&1
if not errorlevel 1 set "PROBE=foreign"
exit /b 0
