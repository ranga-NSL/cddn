[propertyType] $PropertyName# CDN Environment Setup Script
# Creates and activates virtual environment for Configuration-Driven Dependency Network

Write-Host "Setting up CDN virtual environment..." -ForegroundColor Green

# Create virtual environment using Python 3.13
Write-Host "Creating virtual environment..." -ForegroundColor Yellow
D:\python313\python.exe -m venv cdn_env

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& ".\cdn_env\Scripts\Activate.ps1"

# Upgrade pip
Write-Host "Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# Install requirements
Write-Host "Installing dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt

Write-Host "`nEnvironment setup complete!" -ForegroundColor Green
Write-Host "To activate: .\cdn_env\Scripts\Activate.ps1" -ForegroundColor Cyan
Write-Host "To run: python CDN_graph.py" -ForegroundColor Cyan






