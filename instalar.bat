@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
set "PATH=%LOCALAPPDATA%\Microsoft\WinGet\Links;%PATH%"
echo.
echo ===== Instalando o RallyCut =====
echo.

rem --- Python 3.12 -------------------------------------------------------
call :achar_python
if not defined PYEXE (
  echo [1/4] Instalando Python 3.12...
  winget install -e --id Python.Python.3.12 --accept-source-agreements --accept-package-agreements
  call :achar_python
)
if not defined PYEXE (
  echo Python instalado, mas nao encontrado nesta janela. Feche e rode o instalar.bat de novo.
  goto erro
)
echo [1/4] Python: %PYEXE%

rem --- FFmpeg ------------------------------------------------------------
where ffmpeg >nul 2>&1
if errorlevel 1 (
  echo [2/4] Instalando FFmpeg...
  winget install -e --id Gyan.FFmpeg --accept-source-agreements --accept-package-agreements
  if errorlevel 1 goto erro
) else (
  echo [2/4] FFmpeg ja instalado.
)

rem --- Ambiente virtual ----------------------------------------------------
if not exist .venv (
  echo [3/4] Criando ambiente Python...
  "%PYEXE%" -m venv .venv
  if errorlevel 1 goto erro
)
.venv\Scripts\python -m pip install --upgrade pip -q

rem --- Dependencias (PyTorch com GPU se houver placa NVIDIA) -----------------
echo [4/4] Instalando bibliotecas (pode levar alguns minutos)...
where nvidia-smi >nul 2>&1
if not errorlevel 1 (
  echo       Placa NVIDIA encontrada: instalando PyTorch com suporte a GPU.
  .venv\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
  if errorlevel 1 goto erro
)
.venv\Scripts\pip install -r requirements.txt
if errorlevel 1 goto erro
.venv\Scripts\python -c "import torch; print('      Processamento em:', 'GPU ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

echo.
echo ===== Pronto! Abra o RallyCut com o arquivo iniciar.bat =====
echo (se o FFmpeg acabou de ser instalado e der erro, feche e abra o iniciar.bat de novo)
pause
exit /b 0

:achar_python
set "PYEXE="
for /f "delims=" %%i in ('py -3.12 -c "import sys;print(sys.executable)" 2^>nul') do set "PYEXE=%%i"
if not defined PYEXE if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" set "PYEXE=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
if not defined PYEXE if exist "%ProgramFiles%\Python312\python.exe" set "PYEXE=%ProgramFiles%\Python312\python.exe"
exit /b 0

:erro
echo.
echo *** Algo deu errado na instalacao. Copie a mensagem acima e me mande. ***
pause
exit /b 1
