@echo off
chcp 65001 >nul
cd /d "%~dp0.."
rem Mão 3D ao vivo (MediaPipe + three.js) fora da apresentação. Ver demo/hand3d.html.
rem Serve a pasta do projeto só neste computador (127.0.0.1): a página e o modelo modelos_externos\hand_landmarker.task.
rem Feche outros programas que usem a webcam (a apresentação, o teste na webcam) antes de ligar a câmera.
if not exist "modelos_externos\hand_landmarker.task" (
    echo Falta o modelo modelos_externos\hand_landmarker.task. Rode uma vez demo\abrir_camera_pontos.bat, que baixa o modelo.
    pause
    exit /b 1
)
rem Abre o navegador 2 s depois, quando o servidor abaixo já estiver no ar.
start "" /min powershell -NoProfile -Command "Start-Sleep 2; Start-Process 'http://127.0.0.1:8770/demo/hand3d.html'"
echo Mão 3D em http://127.0.0.1:8770/demo/hand3d.html  (feche esta janela para encerrar)
call "%~dp0python.bat" -m http.server 8770 --bind 127.0.0.1
pause
