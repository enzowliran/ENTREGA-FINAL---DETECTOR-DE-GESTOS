@echo off
chcp 65001 >nul
cd /d "%~dp0.."
rem Apresentacao final (demo/apresentacao_final.html), gerada de demo/apresentacao_final_modelo.html com os numeros de resultados/.
call "%~dp0python.bat" -W ignore demo\gerar_apresentacao_final.py
rem Modelo final: CNN 3D pre-treinada (36 classes) + GRU sobre os pontos da mao (MediaPipe). Precisa do .venv_pontos (python_pontos.bat).
call "%~dp0python_pontos.bat" -W ignore demo\player.py --pagina apresentacao_final.html --porta 8767 --apresentacao --tela-cheia --pausa 0.8 --modelo resultados\pontos\modelo_final\cnn.pt --pontos resultados\pontos\modelo_final\gru.pt %*
pause
