@echo off
chcp 65001 >nul
cd /d "%~dp0.."
rem MÃ£o 3D ao vivo (MediaPipe + three.js) fora da apresentaÃ§Ã£o. Ver demo/hand3d.html.
rem Serve a pasta do projeto sÃ³ neste computador (127.0.0.1): a pÃ¡gina e o modelo modelos_externos\hand_landmarker.task.
rem Feche outros programas que usem a webcam (a apresentaÃ§Ã£o, o teste na webcam) antes de ligar a cÃ¢mera.
if not exist "modelos_externos\hand_landmarker.task" (
    echo Falta o modelo modelos_externos\hand_landmarker.task. Rode uma vez demo\abrir_camera_pontos.bat, que baixa o modelo.
    pause
    exit /b 1
)
rem Abre o navegador 2 s depois, quando o servidor abaixo jÃ¡ estiver no ar.
start "" /min powershell -NoProfile -Command "Start-Sleep 2; Start-Process 'http://127.0.0.1:8774/demo/hand3d.html'"
echo MÃ£o 3D em http://127.0.0.1:8774/demo/hand3d.html  (feche esta janela para encerrar)
call "%~dp0python.bat" -m http.server 8774 --bind 127.0.0.1
pause
