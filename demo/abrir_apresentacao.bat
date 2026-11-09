@echo off
chcp 65001 >nul
cd /d "%~dp0.."
".venv\Scripts\python.exe" -W ignore demo\gerar_apresentacao.py
".venv\Scripts\python.exe" -W ignore demo\player.py --pagina apresentacao.html --porta 8767 --apresentacao --tela-cheia --pausa 0.8 --modelo resultados\jester\cnn_3d_mais_gestos\melhor.pt %*
pause
