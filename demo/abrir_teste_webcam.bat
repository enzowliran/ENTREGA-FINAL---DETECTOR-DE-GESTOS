@echo off
chcp 65001 >nul
cd /d "%~dp0.."
echo Testador de gestos na webcam
echo.
set /p PESSOA=Seu nome (sem espacos):
echo.
echo Conjuntos: 1 = gestos das demos (padrao)   2 = so os dedos   3 = todos
set /p OPCAO=Escolha 1, 2 ou 3:
set CONJUNTO=demos
if "%OPCAO%"=="2" set CONJUNTO=dedos
if "%OPCAO%"=="3" set CONJUNTO=todos
call "%~dp0python.bat" -W ignore demo\testar_webcam.py --pessoa %PESSOA% --conjunto %CONJUNTO% %*
echo.
echo Juntando todas as sessoes ja feitas:
call "%~dp0python.bat" -W ignore demo\testar_webcam.py --resumo
pause
