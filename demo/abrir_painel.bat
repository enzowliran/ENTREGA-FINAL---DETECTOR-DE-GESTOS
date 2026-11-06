@echo off
chcp 65001 >nul
cd /d "%~dp0.."
if not exist demo\musicas mkdir demo\musicas
".venv\Scripts\python.exe" -W ignore demo\player.py --pagina painel_carro.html --porta 8770 --painel %*
pause
