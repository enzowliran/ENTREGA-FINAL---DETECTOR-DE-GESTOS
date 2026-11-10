@echo off
chcp 65001 >nul
cd /d "%~dp0.."
set /p PESSOA=Seu nome (sem espacos):
call "%~dp0python.bat" -W ignore demo\gravar_gestos.py --pessoa %PESSOA% %*
pause
