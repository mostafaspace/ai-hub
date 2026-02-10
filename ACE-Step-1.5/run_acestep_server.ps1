Write-Host "============================================" -ForegroundColor Cyan
Write-Host "   ACE-Step 1.5 Music Generation API Server" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Starting API server on http://localhost:8001" -ForegroundColor Green
Write-Host "Models will be downloaded automatically on first run (~3-4GB)" -ForegroundColor Yellow
Write-Host ""
Write-Host "API Documentation: docs/en/API.md" -ForegroundColor Gray
Write-Host ""

# Change to script directory
Set-Location $PSScriptRoot

# Start the API server using uv
uv run acestep-api --port 8001

Read-Host "Press Enter to exit"
