@echo off
rem Roda o Python do projeto.
rem O Controle Inteligente de Aplicativos do Windows pode bloquear o .venv\Scripts\python.exe (um lancador sem assinatura).
rem Por isso, quando existe o Python base do uv, ele e usado direto com os pacotes da .venv no PYTHONPATH.
set "RAIZ=%~dp0.."
set "BASE=%APPDATA%\uv\python\cpython-3.12-windows-x86_64-none\python.exe"
if exist "%BASE%" (
    set "PYTHONPATH=%RAIZ%\.venv\Lib\site-packages;%PYTHONPATH%"
    "%BASE%" %*
) else (
    "%RAIZ%\.venv\Scripts\python.exe" %*
)
