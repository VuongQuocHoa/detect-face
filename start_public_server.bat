@echo off
setlocal enabledelayedexpansion
title NEURAL//VERIFY - AI Recognition Public Server (Cloudflare Tunnel 1-Click)
color 0b

echo ===============================================================================
echo     _   _  _____ _   _ ____      _    _     //  __     _______ ____  ___ _____ __   __
echo    ^| \ ^| ^|^| ____^| ^| ^| ^|  _ \    / \  ^| ^|    //  \ \   / / ____^|  _ \^|_ _^|  ___^|\ \ / /
echo    ^|  \^| ^|^|  _^| ^| ^| ^| ^| ^|_) ^|  / _ \ ^| ^|   //    \ \ / /^|  _^| ^| ^|_) ^|^| ^|^| ^|_    \ V / 
echo    ^| ^|\  ^|^| ^|___^| ^|_^| ^|  _ ^<  / ___ \^| ^|___//      \ V / ^| ^|___^|  _ ^< ^| ^|^|  _^|    ^| ^|  
echo    ^|_^| \_^|^|_____^|\___/^|_^| \_\/_/   \_\_____//        \_/  ^|_____^|_^| \_\___^|_^|      ^|_^|  
echo ===============================================================================
echo  [SYSTEM] Starting AI Face Recognition Server + Public Cloudflare Tunnel...
echo.

if not exist ".\.venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found at .\.venv\Scripts\python.exe!
    pause
    exit /b 1
)

if not exist ".\tools\cloudflared.exe" (
    echo [ERROR] cloudflared.exe not found in .\tools directory!
    pause
    exit /b 1
)

.\.venv\Scripts\python.exe tools\run_tunnel.py
pause
