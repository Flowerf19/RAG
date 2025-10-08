#!/usr/bin/env powershell
<#
.SYNOPSIS
    Test runner script for RAG project - Loader module only
    
.DESCRIPTION
    Script để chạy tests cho PDFLoader class theo chuẩn OOP.
    Chỉ focus vào loaders module, không test các module khác.
    
.PARAMETER Coverage
    Chạy với coverage report
    
.PARAMETER Verbose
    Chạy với verbose output
    
.PARAMETER Quick
    Chạy quick tests only (skip slow tests)
    
.EXAMPLE
    .\run_tests.ps1
    .\run_tests.ps1 -Coverage
    .\run_tests.ps1 -Verbose -Coverage
#>

param(
    [switch]$Coverage,
    [switch]$Verbose,
    [switch]$Quick
)

# Colors for output
$ErrorColor = "Red"
$SuccessColor = "Green"
$InfoColor = "Cyan"
$WarningColor = "Yellow"

Write-Host "=== RAG Project Test Runner - Loader Module Only ===" -ForegroundColor $InfoColor
Write-Host ""

# Check if we're in the right directory
$expectedPath = "C:\Users\ENGUYEHWC\Downloads\RAG\RAG"
$currentPath = Get-Location

if ($currentPath.Path -ne $expectedPath) {
    Write-Host "⚠️  Changing to project directory: $expectedPath" -ForegroundColor $WarningColor
    Set-Location $expectedPath
}

# Check if virtual environment is activated
if ($env:VIRTUAL_ENV -eq $null) {
    Write-Host "🔄 Activating virtual environment..." -ForegroundColor $WarningColor
    & .\\.venv\\Scripts\\Activate.ps1
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Failed to activate virtual environment!" -ForegroundColor $ErrorColor
        Write-Host "Make sure .venv exists. Run: python -m venv .venv" -ForegroundColor $ErrorColor
        exit 1
    }
}

Write-Host "✅ Virtual environment activated" -ForegroundColor $SuccessColor

# Check if pytest is installed
Write-Host "🔍 Checking dependencies..." -ForegroundColor $InfoColor
$pytestCheck = python -m pytest --version 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "📦 Installing pytest dependencies..." -ForegroundColor $WarningColor
    pip install pytest pytest-cov pytest-mock
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Failed to install pytest!" -ForegroundColor $ErrorColor
        exit 1
    }
}

Write-Host "✅ Dependencies ready" -ForegroundColor $SuccessColor
Write-Host ""

# Build pytest command
$pytestCmd = @("python", "-m", "pytest")

# Add test path - only test loaders
$pytestCmd += "tests/test_loader.py"

# Add verbosity
if ($Verbose) {
    $pytestCmd += "-v"
} else {
    $pytestCmd += "-q"
}

# Add coverage
if ($Coverage) {
    $pytestCmd += "--cov=loaders"
    $pytestCmd += "--cov-report=term-missing"
    $pytestCmd += "--cov-report=html:htmlcov"
}

# Quick tests (skip slow markers)
if ($Quick) {
    $pytestCmd += '-m', '"not slow"'
}

# Add colors and output formatting
$pytestCmd += "--color=yes"
$pytestCmd += "--tb=short"

Write-Host "🧪 Running tests for PDFLoader..." -ForegroundColor $InfoColor
Write-Host "Command: $($pytestCmd -join ' ')" -ForegroundColor $InfoColor
Write-Host ""

# Run tests
$startTime = Get-Date
& $pytestCmd[0] $pytestCmd[1..($pytestCmd.Length-1)]
$exitCode = $LASTEXITCODE
$endTime = Get-Date
$duration = $endTime - $startTime

Write-Host ""
Write-Host "⏱️  Test duration: $($duration.TotalSeconds.ToString('F2')) seconds" -ForegroundColor $InfoColor

if ($exitCode -eq 0) {
    Write-Host "🎉 All tests passed!" -ForegroundColor $SuccessColor
    
    if ($Coverage) {
        Write-Host "📊 Coverage report generated in htmlcov/index.html" -ForegroundColor $InfoColor
    }
} else {
    Write-Host "❌ Some tests failed!" -ForegroundColor $ErrorColor
    Write-Host "Check the output above for details." -ForegroundColor $ErrorColor
}

Write-Host ""
Write-Host "=== Test Complete ===" -ForegroundColor $InfoColor

exit $exitCode