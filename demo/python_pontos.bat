@echo off
rem Roda o Python do ambiente separado .venv_pontos, que tem o MediaPipe.
rem O .venv original continua intocado: o MediaPipe traz o opencv-contrib, que nao pode conviver com o opencv-python dele.
rem Se o .venv_pontos ainda nao existe, ele e criado aqui (precisa do uv: https://docs.astral.sh/uv/).
rem Mesma regra do python.bat: com o Python base do uv, ele e usado direto com os pacotes do .venv_pontos no PYTHONPATH,
rem porque o Controle Inteligente de Aplicativos do Windows pode bloquear o .venv_pontos\Scripts\python.exe.
set "RAIZ=%~dp0.."
if not exist "%RAIZ%\.venv_pontos\Lib\site-packages\mediapipe" (
    echo Criando o ambiente .venv_pontos com o MediaPipe. Leva uns 2 minutos, so na primeira vez...
    set "UV=uv"
    if exist "%USERPROFILE%\.local\bin\uv.exe" set "UV=%USERPROFILE%\.local\bin\uv.exe"
    call "%%UV%%" venv "%RAIZ%\.venv_pontos" --python 3.12 --allow-existing || goto erro
    call "%%UV%%" pip install --python "%RAIZ%\.venv_pontos" -r "%RAIZ%\requirements_pontos.txt" --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match || goto erro
)
set "BASE=%APPDATA%\uv\python\cpython-3.12-windows-x86_64-none\python.exe"
if exist "%BASE%" (
    set "PYTHONPATH=%RAIZ%\.venv_pontos\Lib\site-packages;%PYTHONPATH%"
    rem Sem torch no .venv_pontos, usa o torch do .venv (com GPU), depois dos pacotes do MediaPipe.
    if not exist "%RAIZ%\.venv_pontos\Lib\site-packages\torch" if exist "%RAIZ%\.venv\Lib\site-packages\torch" set "PYTHONPATH=%RAIZ%\.venv_pontos\Lib\site-packages;%PYTHONPATH%;%RAIZ%\.venv\Lib\site-packages"
    "%BASE%" %*
) else (
    "%RAIZ%\.venv_pontos\Scripts\python.exe" %*
)
exit /b %errorlevel%
:erro
echo Nao consegui criar o .venv_pontos. Instale o uv (https://docs.astral.sh/uv/) e tente de novo.
exit /b 1
