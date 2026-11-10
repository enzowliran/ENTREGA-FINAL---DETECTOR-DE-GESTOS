@echo off
chcp 65001 >nul
cd /d "%~dp0.."
if not exist demo\musicas mkdir demo\musicas
call "%~dp0python.bat" -W ignore demo\player.py --pagina painel_carro.html --porta 8770 --painel --modelo resultados\jester\cnn_3d_mais_gestos\melhor.pt %*
pause
