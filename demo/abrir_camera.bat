
@echo off
chcp 65001 >nul
cd /d "%~dp0.."
".venv\Scripts\python.exe" -W ignore demo\ao_vivo.py %*
pause
