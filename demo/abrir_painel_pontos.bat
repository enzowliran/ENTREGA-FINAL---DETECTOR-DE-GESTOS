@echo off
chcp 65001 >nul
cd /d "%~dp0.."
rem Parte nova: o mesmo painel do carro, com os numeros pelos pontos da mao. Ver PONTOS_MAO.md.
if not exist demo\musicas mkdir demo\musicas
call "%~dp0python_pontos.bat" -W ignore demo\player_pontos.py --pagina painel_carro.html --porta 8770 --painel %*
pause
