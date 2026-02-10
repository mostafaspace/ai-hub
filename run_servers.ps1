# Antigravity AI Server Launcher
# PowerShell version with menu-driven server selection

$Host.UI.RawUI.WindowTitle = "Antigravity AI Server Launcher"

# ============================================================
# SERVER CONFIGURATION - Add new servers here!
# ============================================================
$servers = @(
    @{
        Name = "Qwen3 TTS Server"
        Port = 8000
        Description = "Text-to-Speech"
        Path = Join-Path $PSScriptRoot "Qwen3-TTS"
        Command = "set HF_HOME=D:\hf_models&& pip install -r requirements.txt -q && python server.py"
        UseUv = $false
    },
    @{
        Name = "ACE-Step Music Server"
        Port = 8001
        Description = "Music Generation"
        Path = Join-Path $PSScriptRoot "ACE-Step-1.5"
        Command = "acestep-api --host 0.0.0.0 --port 8001"
        UseUv = $true
    }
    # Add more servers here following the same pattern:
    # @{
    #     Name = "New Server Name"
    #     Port = 8002
    #     Description = "What it does"
    #     Path = Join-Path $PSScriptRoot "folder-name"
    #     Command = "command to run"
    #     UseUv = $true  # or $false for pip/python
    # }
)

function Show-Menu {
    Clear-Host
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host "            ANTIGRAVITY AI SERVER LAUNCHER" -ForegroundColor Cyan
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  Available Servers:" -ForegroundColor White
    Write-Host ""
    
    for ($i = 0; $i -lt $servers.Count; $i++) {
        $s = $servers[$i]
        $num = $i + 1
        Write-Host "  [$num] $($s.Name)" -ForegroundColor Green -NoNewline
        Write-Host " (Port $($s.Port))" -ForegroundColor DarkGray -NoNewline
        Write-Host " - $($s.Description)" -ForegroundColor Yellow
    }
    
    Write-Host ""
    Write-Host "  [A] Start ALL Servers" -ForegroundColor Magenta
    Write-Host "  [S] Check Server Status" -ForegroundColor Blue
    Write-Host "  [Q] Quit" -ForegroundColor Red
    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Cyan
}

function Start-Server {
    param($server)
    
    Write-Host ""
    Write-Host "Starting $($server.Name) on port $($server.Port)..." -ForegroundColor Green
    
    $cmd = $server.Command
    if ($server.UseUv) {
        $cmd = "uv run $cmd"
    }
    
    Start-Process cmd -ArgumentList "/k", "cd /d `"$($server.Path)`" && $cmd" -WindowStyle Normal
    Write-Host "$($server.Name) starting in new window!" -ForegroundColor Cyan
}

function Check-ServerStatus {
    Write-Host ""
    Write-Host "Checking server status..." -ForegroundColor Yellow
    Write-Host ""
    
    foreach ($s in $servers) {
        try {
            $response = Invoke-WebRequest -Uri "http://192.168.1.26:$($s.Port)/health" -TimeoutSec 2 -UseBasicParsing -ErrorAction Stop
            Write-Host "  [ONLINE]  $($s.Name) (Port $($s.Port))" -ForegroundColor Green
        }
        catch {
            Write-Host "  [OFFLINE] $($s.Name) (Port $($s.Port))" -ForegroundColor Red
        }
    }
    Write-Host ""
}

# Main loop
while ($true) {
    Show-Menu
    $choice = Read-Host "Select an option"
    
    if ($choice -eq 'Q' -or $choice -eq 'q') {
        Write-Host "Goodbye!" -ForegroundColor Yellow
        break
    }
    elseif ($choice -eq 'A' -or $choice -eq 'a') {
        Write-Host ""
        Write-Host "Starting ALL servers..." -ForegroundColor Magenta
        for ($i = 0; $i -lt $servers.Count; $i++) {
            Write-Host "[$($i+1)/$($servers.Count)] " -NoNewline
            Start-Server $servers[$i]
            Start-Sleep -Seconds 2
        }
        Write-Host ""
        Write-Host "============================================================" -ForegroundColor Cyan
        Write-Host "  All servers are starting in separate windows!" -ForegroundColor Green
        Write-Host ""
        Write-Host "  Endpoints (Device IP: 192.168.1.26):" -ForegroundColor White
        foreach ($s in $servers) {
            Write-Host "    $($s.Name): http://192.168.1.26:$($s.Port)" -ForegroundColor Yellow
        }
        Write-Host "============================================================" -ForegroundColor Cyan
        Read-Host "Press Enter to continue"
    }
    elseif ($choice -eq 'S' -or $choice -eq 's') {
        Check-ServerStatus
        Read-Host "Press Enter to continue"
    }
    elseif ($choice -match '^\d+$') {
        $index = [int]$choice - 1
        if ($index -ge 0 -and $index -lt $servers.Count) {
            Start-Server $servers[$index]
            Read-Host "Press Enter to continue"
        }
        else {
            Write-Host "Invalid selection!" -ForegroundColor Red
            Start-Sleep -Seconds 1
        }
    }
    else {
        Write-Host "Invalid option!" -ForegroundColor Red
        Start-Sleep -Seconds 1
    }
}
