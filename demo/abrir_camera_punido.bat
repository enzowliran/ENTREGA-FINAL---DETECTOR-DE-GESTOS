@echo off
chcp 65001 >nul
cd /d "%~dp0.."
call "%~dp0python.bat" -W ignore demo\ao_vivo.py --modelo resultados\jester\cnn_3d_pequeno_punido\melhor.pt %*
pause
