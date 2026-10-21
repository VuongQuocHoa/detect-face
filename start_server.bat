@echo off
setlocal enabledelayedexpansion
title NEURAL//VERIFY - AI Recognition Server (1-Click)
color 0b

echo ===============================================================================
echo     _   _  _____ _   _ ____      _    _     //  __     _______ ____  ___ _____ __   __
echo    ^| \ ^| ^|^| ____^| ^| ^| ^|  _ \    / \  ^| ^|    //  \ \   / / ____^|  _ \^|_ _^|  ___^|\ \ / /
echo    ^|  \^| ^|^|  _^| ^| ^| ^| ^| ^|_) ^|  / _ \ ^| ^|   //    \ \ / /^|  _^| ^| ^|_) ^|^| ^|^| ^|_    \ V / 
echo    ^| ^|\  ^|^| ^|___^| ^|_^| ^|  _ ^<  / ___ \^| ^|___//      \ V / ^| ^|___^|  _ ^< ^| ^|^|  _^|    ^| ^|  
echo    ^|_^| \_^|^|_____^|\___/^|_^| \_\/_/   \_\_____//        \_/  ^|_____^|_^| \_\___^|_^|      ^|_^|  
echo ===============================================================================
echo  [SYSTEM] Detecting Local Area Network (LAN) IP for Mobile APK Connection...
echo.

:: Get active IPv4 address using Python helper
for /f "tokens=*" %%a in ('.\.venv\Scripts\python.exe tools\get_ip.py') do set LAN_IP=%%a

if "%LAN_IP%"=="" set LAN_IP=127.0.0.1

echo  =============================================================================
echo    [ONLINE] SERVER ACCESS POINTS:
echo    - PC Browser:        http://127.0.0.1:8000
echo    - Mobile APK / LAN:  http://%LAN_IP%:8000
echo    - WebSocket Stream:  ws://%LAN_IP%:8000/ws/recognize
echo  =============================================================================
echo.
echo  [INSTRUCTION FOR MOBILE APK USERS]:
echo    1. Connect your Android phone to the SAME Wi-Fi network as this PC.
echo    2. Open FaceID CyberHUD on your Android phone.
echo    3. Enter Server IP: %LAN_IP%:8000
echo  =============================================================================
echo.

if not exist ".\.venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found at .\.venv\Scripts\python.exe!
    pause
    exit /b 1
)

echo  [STARTING] Launching Intel OpenVINO Hardware-Accelerated Uvicorn Server (Auto-Reload Enabled)...
echo.
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
pause
