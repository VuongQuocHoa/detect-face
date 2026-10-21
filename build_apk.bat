@echo off
setlocal enabledelayedexpansion
title FaceID CyberHUD - Automated APK Builder
color 0b

echo ===============================================================================
echo     _    ____  _  __   ____  _   _ ___ _     ____  _____ ____  
echo    / \  ^|  _ \^| ^|/ /  ^| __ )^| ^| ^| ^|_ _^| ^|   ^|  _ \^| ____^|  _ \ 
echo   / _ \ ^| ^|_) ^| ' /   ^|  _ \^| ^| ^| ^|^| ^|^| ^|   ^| ^| ^| ^|  _^| ^| ^|_) ^|
echo  / ___ \^|  __/^| . \   ^| ^|_) ^| ^|_^| ^|^| ^|^| ^|___^| ^|_^| ^| ^|___^|  _ ^< 
echo /_/   \_\_^|   ^|_^|\_\  ^|____/ \___/^|___^|_____^|____/^|_____^|_^| \_\
echo ===============================================================================
echo.

set JAVA_BIN=C:\Program Files\JetBrains\PyCharm 2026.2.1\jbr\bin
if not exist "%JAVA_BIN%\java.exe" (
    echo [ERROR] Java Runtime not found at %JAVA_BIN%!
    pause
    exit /b 1
)

echo [1/2] Syncing assets and building APK with package_and_sign.py...
.\.venv\Scripts\python.exe tools\package_and_sign.py
if errorlevel 1 (
    echo [ERROR] Packaging failed!
    pause
    exit /b 1
)

echo.
echo ===============================================================================
echo   SUCCESS! Android APK built and verified successfully:
echo   [APK FILE] dist\FaceID_CyberHUD.apk
echo ===============================================================================
echo.
pause
