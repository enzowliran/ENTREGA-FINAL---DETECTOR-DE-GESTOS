@echo off
chcp 65001 >nul
cd /d "%~dp0.."
echo Testador de gestos na webcam
echo.
echo Atualizando o projeto com o GitHub...
git pull --ff-only -q 2>nul || echo (nao deu para atualizar agora; seguindo com a versao local)
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
set /p ENVIAR=Enviar seus resultados para o GitHub do grupo? (s/n):
if /i not "%ENVIAR%"=="s" goto resumo
git add resultados/webcam
git commit -q -m "Teste na webcam: %PESSOA%" && git pull --rebase -q && git push -q && echo Resultados enviados. || echo Nao consegui enviar. Rode depois: git pull --rebase e git push
:resumo
echo.
echo Juntando as sessoes de todos os integrantes:
call "%~dp0python.bat" -W ignore demo\testar_webcam.py --resumo
pause
