@echo off
chcp 65001 >nul
cd /d "%~dp0.."
rem Teste ao vivo na webcam (demo/teste_gestos.html) com o modelo final, o mesmo do carro 3D (mesmos comandos e gatilho).
rem Pede um gesto de cada vez, fala o que reconheceu e com que confiança e grava a sessão em resultados\webcam_ao_vivo\.
rem Feche outros programas que usem a webcam antes.
call "%~dp0python_pontos.bat" -W ignore demo\player.py --pagina teste_gestos.html --porta 8776 --painel --carro3d --modelo resultados\pontos\modelo_final\cnn.pt --pontos resultados\pontos\modelo_final\gru.pt %*
pause
