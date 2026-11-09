@echo off
chcp 65001 >nul
cd /d "%~dp0.."
".venv\Scripts\python.exe" -W ignore demo\player.py --pagina hospital.html --porta 8771 --hospital --pausa 0.8 --modelo resultados\jester\cnn_3d_mais_gestos\melhor.pt %*
pause
