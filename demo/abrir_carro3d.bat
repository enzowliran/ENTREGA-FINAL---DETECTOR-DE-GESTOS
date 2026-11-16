@echo off
chcp 65001 >nul
cd /d "%~dp0.."
rem Carro em 3D (demo/carro3d.html): visao do motorista, com as duas maos em 3D dentro do carro.
rem Modelo final: CNN 3D pre-treinada + GRU sobre os pontos da mao (reconhece 1 a 4 dedos). A webcam fica com o servidor;
rem a pagina recebe o quadro dele em /video_cru para desenhar as maos.
if not exist demo\musicas mkdir demo\musicas
call "%~dp0python_pontos.bat" -W ignore demo\player.py --pagina carro3d.html --porta 8772 --painel --carro3d --tela-cheia --modelo resultados\pontos\modelo_final\cnn.pt --pontos resultados\pontos\modelo_final\gru.pt %*
pause
