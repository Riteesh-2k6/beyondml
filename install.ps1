$ErrorActionPreference = "Stop"

Write-Host "==================================" -ForegroundColor Cyan
Write-Host "     Installing BeyondML...       " -ForegroundColor Cyan
Write-Host "==================================" -ForegroundColor Cyan

# Check for Git
if (!(Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Error "Git is required to install BeyondML. Please install Git first."
    exit 1
}

# Check for Python
if (!(Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Error "Python is required to install BeyondML. Please install Python first."
    exit 1
}

$InstallDir = "$env:USERPROFILE\BeyondML"

# Clone or pull
if (!(Test-Path $InstallDir)) {
    Write-Host "Cloning repository into $InstallDir..."
    git clone https://github.com/Riteesh-2k6/beyondml.git $InstallDir
} else {
    Write-Host "Directory $InstallDir already exists, pulling latest changes..."
    Push-Location $InstallDir
    git pull
    Pop-Location
}

Push-Location $InstallDir

# Create venv
if (!(Test-Path ".venv")) {
    Write-Host "Creating virtual environment..."
    python -m venv .venv
}

Write-Host "Activating virtual environment and installing dependencies..."
& ".venv\Scripts\python.exe" -m pip install --upgrade pip
& ".venv\Scripts\pip.exe" install -e .

Write-Host "`n✅ BeyondML installed successfully!" -ForegroundColor Green
Write-Host "To run the application, just copy and paste these lines:" -ForegroundColor Yellow
Write-Host "cd ~/BeyondML"
Write-Host ".\"
Write-Host "beyondml run"

Pop-Location
