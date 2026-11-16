@echo off
chcp 65001 >nul
cd /d "%~dp0.."
rem Carro em 3D (demo/carro3d.html): visão do motorista, com as duas mãos em 3D dentro do carro e os gestos do modelo.
rem A webcam fica com o servidor (gestos); a página recebe o quadro dele em /video_cru para desenhar as mãos.
if not exist demo\musicas mkdir demo\musicas
call "%~dp0python.bat" -W ignore demo\player.py --pagina carro3d.html --porta 8772 --painel --tela-cheia --modelo resultados\jester\cnn_3d_mais_gestos\melhor.pt %*
pause
