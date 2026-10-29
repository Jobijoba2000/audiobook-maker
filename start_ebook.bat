@echo off
setlocal
chcp 65001 >nul
set "PYTHONUTF8=1"
set "PYTHONDONTWRITEBYTECODE=1"
cd /d "%~dp0"
if exist "%~dp0.cache\ffmpeg\bin\ffmpeg.exe" set "PATH=%~dp0.cache\ffmpeg\bin;%PATH%"
if not exist ".venv\Scripts\python.exe" (
    echo Environnement Python absent. Lancez installer.bat.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -B -m epub_audio %*
set "RESULT=%ERRORLEVEL%"
if not "%RESULT%"=="0" (
    pause
) else (
    if "%~1"=="" pause
)
exit /b %RESULT%
