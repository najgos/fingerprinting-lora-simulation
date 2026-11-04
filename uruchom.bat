@echo off
rem Starts the simulator without a console window (double-click this file).
cd /d "%~dp0"
start "" ".venv\Scripts\pythonw.exe" app.py
