@echo off
chcp 65001 >nul
cd /d "%~dp0.."
call "%~dp0python.bat" -W ignore demo\player.py --pagina tenis.html --porta 8769 --limiar 0.6 --seguidas 2 --pausa 0.6 --janelas 1.0 1.5 %*
pause
