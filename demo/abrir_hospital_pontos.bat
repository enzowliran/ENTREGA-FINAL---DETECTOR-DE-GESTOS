@echo off
chcp 65001 >nul
cd /d "%~dp0.."
rem Parte nova: o mesmo hospital, com os numeros pelos pontos da mao. Ver PONTOS_MAO.md.
call "%~dp0python_pontos.bat" -W ignore demo\player_pontos.py --pagina hospital.html --porta 8771 --hospital --pausa 0.8 %*
pause
