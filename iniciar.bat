@echo off
chcp 65001 >nul
cd /d "%~dp0"
set "PATH=%LOCALAPPDATA%\Microsoft\WinGet\Links;%PATH%"
if not exist .venv (
  echo Rode primeiro o instalar.bat
  pause
  exit /b 1
)
.venv\Scripts\python -m rallycut
pause
