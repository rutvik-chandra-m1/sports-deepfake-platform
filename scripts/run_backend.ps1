# Sets up a virtualenv (if missing) and runs the FastAPI backend with reload.
# Windows PowerShell equivalent of run_backend.sh.
$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$BackendDir = Join-Path $ScriptDir "..\backend"

Set-Location $BackendDir

if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment..."
    python -m venv .venv
}

& ".venv\Scripts\Activate.ps1"
python -m pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet

if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "Created backend/.env from .env.example"
}

Write-Host "Starting backend at http://localhost:8000 (docs at /docs)..."
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
