@echo off
echo ============================================
echo    ACE-Step 1.5 Music Generation API Server
echo ============================================
echo.
echo Starting API server on http://localhost:8001
echo Models will be downloaded automatically on first run (~3-4GB)
echo.
echo API Documentation: docs/en/API.md
echo.

cd /d "%~dp0"

REM Start the API server using uv
uv run acestep-api --port 8001

pause
