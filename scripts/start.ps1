[CmdletBinding()]
param([ValidateRange(1024, 65535)][int]$Port = 8091)

$ErrorActionPreference = "Stop"
$projectDirectory = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$pythonExecutable = Join-Path $projectDirectory ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $pythonExecutable)) {
    throw "Create .venv and install the project dependencies first; see README.md."
}
$serverArguments = @("-X", "utf8", "-m", "uvicorn", "medical_ai_selective_context.main:app",
    "--app-dir", (Join-Path $projectDirectory "src"), "--host", "127.0.0.1",
    "--port", "$Port", "--no-access-log")
$settingsFile = Join-Path $projectDirectory ".env"
if (Test-Path -LiteralPath $settingsFile) {
    $serverArguments += @("--env-file", $settingsFile)
}
Push-Location -LiteralPath $projectDirectory
try { & $pythonExecutable @serverArguments }
finally { Pop-Location }
