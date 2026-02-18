@echo off
setlocal EnableDelayedExpansion
title Antigravity AI Server Launcher
color 0A

:: ============================================================
:: ONE-CLICK START: Run without arguments to start all servers
:: ============================================================
if "%1"=="" goto START_ALL_DIRECT

:MENU
cls
echo ============================================================
echo             ANTIGRAVITY AI SERVER LAUNCHER
echo ============================================================
echo.
echo   Available Servers:
echo.
echo   [1] Qwen3 TTS Server         (Port 8000) - Text-to-Speech
echo   [2] ACE-Step Music Server    (Port 8001) - Music Generation
echo   [3] Qwen3 ASR Server         (Port 8002) - Speech-to-Text
echo   [4] Vision Service           (Port 8003) - GLM-Image Generation
echo.
echo   [A] Start ALL Servers (with auto-restart)
echo   [U] Unified Server (All in one window)
echo   [R] Restart ALL Servers
echo   [K] Kill ALL Servers
echo   [Q] Quit
echo.
echo ============================================================
set /p choice="Select an option: "

if /i "%choice%"=="1" goto TTS
if /i "%choice%"=="2" goto ACESTEP
if /i "%choice%"=="3" goto ASR
if /i "%choice%"=="4" goto VISION
if /i "%choice%"=="A" goto START_ALL
if /i "%choice%"=="U" goto UNIFIED
if /i "%choice%"=="R" goto RESTART_ALL
if /i "%choice%"=="K" goto KILL_ALL
if /i "%choice%"=="Q" goto EXIT
echo Invalid choice. Press any key to try again...
pause >nul
goto MENU

:UNIFIED
call :KILL_ALL
echo.
echo Starting Unified Server (All-in-One)...
echo Press Ctrl+C in the new window to stop all servers.
echo.
start "Unified AI Server" cmd /k "cd /d %~dp0 && python unified_server.py"
goto MENU

:KILL_ALL
echo.
echo Stopping any existing servers...
:: Kill Python processes on ports 8000, 8001, 8002, 8003
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8000.*LISTENING"') do (
    echo Killing process on port 8000 (PID: %%a)
    taskkill /F /PID %%a >nul 2>&1
)
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8001.*LISTENING"') do (
    echo Killing process on port 8001 (PID: %%a)
    taskkill /F /PID %%a >nul 2>&1
)
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8002.*LISTENING"') do (
    echo Killing process on port 8002 (PID: %%a)
    taskkill /F /PID %%a >nul 2>&1
)
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8003.*LISTENING"') do (
    echo Killing process on port 8003 (PID: %%a)
    taskkill /F /PID %%a >nul 2>&1
)
echo All servers stopped.
timeout /t 2 >nul
goto :EOF

:TTS
call :KILL_PORT 8000
echo.
echo Starting Qwen3 TTS Server on port 8000...
start "Qwen3 TTS Server" cmd /k "cd /d %~dp0Qwen3-TTS && set HF_HOME=D:\hf_models&& python server.py"
echo TTS Server starting in new window!
pause
goto MENU

:ACESTEP
call :KILL_PORT 8001
echo.
echo Starting ACE-Step Music Server on port 8001...
start "ACE-Step Music Server" cmd /k "cd /d %~dp0ACE-Step-1.5 && uv run acestep-api --host 0.0.0.0 --port 8001"
echo ACE-Step Server starting in new window!
pause
goto MENU

:ASR
call :KILL_PORT 8002
echo.
echo Starting Qwen3 ASR Server on port 8002...
start "Qwen3 ASR Server" cmd /k "cd /d %~dp0Qwen3-ASR && set HF_HOME=D:\hf_models && python server.py"
echo ASR Server starting in new window!
pause
goto MENU

:VISION
call :KILL_PORT 8003
echo.
echo Starting Vision Service on port 8003...
start "Vision Service" cmd /k "cd /d %~dp0Vision-Service && set HF_HOME=D:\hf_models && python vision_server.py"
echo Vision Service starting in new window!
pause
goto MENU

:RESTART_ALL
call :KILL_ALL
goto START_ALL

:START_ALL_DIRECT
:: One-click mode: Kill existing and start all
echo ============================================================
echo        ANTIGRAVITY AI - Starting All Servers
echo ============================================================
echo.

:START_ALL
echo Checking for existing servers...
:: Kill any existing servers first
call :KILL_ALL
timeout /t 2 >nul

echo.
echo [1/4] Starting Qwen3 TTS Server (port 8000)...
start "Qwen3 TTS Server" cmd /k "cd /d %~dp0Qwen3-TTS && set HF_HOME=D:\hf_models&& python server.py"
timeout /t 3 >nul

echo [2/4] Starting ACE-Step Music Server (port 8001)...
start "ACE-Step Music Server" cmd /k "cd /d %~dp0ACE-Step-1.5 && uv run acestep-api --host 0.0.0.0 --port 8001"
timeout /t 3 >nul

echo [3/4] Starting Qwen3 ASR Server (port 8002)...
start "Qwen3 ASR Server" cmd /k "cd /d %~dp0Qwen3-ASR && set HF_HOME=D:\hf_models && python server.py"
timeout /t 3 >nul

echo [4/4] Starting Vision Service (port 8003)...
start "Vision Service" cmd /k "cd /d %~dp0Vision-Service && set HF_HOME=D:\hf_models && python vision_server.py"

echo.
echo ============================================================
echo   All servers are starting in separate windows!
echo.
echo   Endpoints:
echo     TTS:       http://localhost:8000
echo     ACE-Step:  http://localhost:8001
echo     ASR:       http://localhost:8002
echo     Vision:    http://localhost:8003
echo.
echo   Wait ~60 seconds for models to load, then test with:
echo     python test_all_servers.py
echo ============================================================

if "%1"=="" (
    echo.
    echo Press any key to open menu, or close this window...
    pause >nul
    goto MENU
)
pause
goto MENU

:KILL_PORT
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":%1.*LISTENING"') do (
    echo Stopping existing server on port %1 (PID: %%a)
    taskkill /F /PID %%a >nul 2>&1
)
goto :EOF

:EXIT
echo Goodbye!
exit /b 0
