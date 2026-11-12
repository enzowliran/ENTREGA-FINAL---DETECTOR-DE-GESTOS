@echo off
chcp 65001 >nul
cd /d "%~dp0.."
rem Parte nova: CNN 3D so do Jester + numeros pelos pontos da mao (MediaPipe). Ver PONTOS_MAO.md.
call "%~dp0python_pontos.bat" -W ignore demo\ao_vivo_pontos.py %*
pause
