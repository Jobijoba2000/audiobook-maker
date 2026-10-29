@echo off
setlocal
chcp 65001 >nul
set "PYTHONUTF8=1"
cd /d "%~dp0"
set "PYTHONDONTWRITEBYTECODE=1"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\bootstrap.ps1" %*
set "RESULT=%ERRORLEVEL%"
if "%AUDIOBOOK_NO_PAUSE%"=="1" exit /b %RESULT%
for %%A in (%*) do if /I "%%~A"=="--no-pause" exit /b %RESULT%
pause
exit /b %RESULT%
