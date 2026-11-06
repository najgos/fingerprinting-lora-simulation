@echo off
rem Double-click to start the simulator.
rem The first start creates a virtual environment and installs the packages (a few minutes).
rem The environment lives in a short path under %LOCALAPPDATA%, because Qt has file paths too long
rem for Windows when the project sits in a deep folder (e.g. an unpacked GitHub ZIP in Downloads).
cd /d "%~dp0"
set "VENV=%LOCALAPPDATA%\lora-fingerprinting-sim\venv"

if exist "%VENV%\Scripts\python.exe" goto :check

set "PY="
python -c "import sys; sys.exit(sys.version_info < (3, 10))" >nul 2>nul && set "PY=python"
if not defined PY py -3 -c "import sys; sys.exit(sys.version_info < (3, 10))" >nul 2>nul && set "PY=py -3"
if not defined PY goto :nopython
echo Creating the virtual environment in %VENV% ...
%PY% -m venv "%VENV%" || goto :failed

:check
"%VENV%\Scripts\python.exe" -c "import numpy, matplotlib, PySide6.QtWidgets" >nul 2>nul && goto :start
echo Installing packages, first start only...
"%VENV%\Scripts\python.exe" -m pip install --disable-pip-version-check -r requirements.txt || goto :failed

:start
start "" "%VENV%\Scripts\pythonw.exe" app.py
exit /b 0

:nopython
echo Python 3.10 or newer was not found.
echo Install it from https://www.python.org and tick "Add python.exe to PATH", then run this file again.
pause
exit /b 1

:failed
echo Setting up failed, see the messages above.
pause
exit /b 1
