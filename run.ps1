<#
.SYNOPSIS
    Master runner script for the Ayurvedic Prakriti Classifier project.

.DESCRIPTION
    Contains all commands to run, test, and serve the project components.

.EXAMPLE
    .\run.ps1 -Command app
    .\run.ps1 -Command test
    .\run.ps1 -Command docs
    .\run.ps1 -Command pipeline
    .\run.ps1 -Command train
    .\run.ps1 -Command all
#>

[CmdletBinding()]
param(
    [Parameter(Position=0)]
    [ValidateSet("app", "test", "docs", "pipeline", "train", "all", "menu")]
    [string]$Command = "all"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot

$PythonExe = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $PythonExe)) {
    $PythonExe = "python"
}

Write-Host "==============================================================================" -ForegroundColor Green
Write-Host "        AYURVEDIC PRAKRITI CLASSIFIER - POWERSHELL RUNNER" -ForegroundColor Green
Write-Host "==============================================================================" -ForegroundColor Green
Write-Host " Project Directory : $ProjectRoot"
Write-Host " Python Runtime    : $PythonExe"
Write-Host " Executing Command : $Command"
Write-Host "=============================================================================="

& $PythonExe (Join-Path $ProjectRoot "run_project.py") $Command
