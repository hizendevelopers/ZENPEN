$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $venvPython)) {
    Write-Host "Creating virtual environment..."
    python -m venv .venv
}

Write-Host "Installing requirements..."
& $venvPython -m pip install -r "backend\requirements.txt"

Write-Host "Starting server on http://127.0.0.1:8000"
& $venvPython -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
