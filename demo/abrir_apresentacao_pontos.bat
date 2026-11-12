@echo off
chcp 65001 >nul
cd /d "%~dp0.."
rem Parte nova: a mesma apresentacao, com os numeros pelos pontos da mao. Ver PONTOS_MAO.md.
rem Nao roda o gerar_apresentacao.py: usa o demo\apresentacao.html como ele esta.
call "%~dp0python_pontos.bat" -W ignore demo\player_pontos.py --pagina apresentacao.html --porta 8767 --apresentacao --tela-cheia --pausa 0.8 %*
pause
