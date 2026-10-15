[CmdletBinding()]
param(
    [ValidateRange(1024, 65535)][int]$Port = 8092,
    [string]$BindHost = "127.0.0.1",
    [string]$CcoDirectory = ""
)

$ErrorActionPreference = "Stop"
$projectDirectory = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$settingsFile = Join-Path $projectDirectory ".env"

if ($CcoDirectory) {
    $resolvedCcoDirectory = [System.IO.Path]::GetFullPath($CcoDirectory)
    if (-not (Test-Path -LiteralPath $resolvedCcoDirectory -PathType Container)) {
        throw "CCO klasörü bulunamadı: $resolvedCcoDirectory"
    }
    $env:DEMO_CCO_DIR = $resolvedCcoDirectory
}

if ($BindHost -notin @("127.0.0.1", "localhost", "::1") -and -not $env:DEMO_ALLOWED_NETWORKS) {
    throw "Ofis ağına bind etmeden önce DEMO_ALLOWED_NETWORKS değerini açık CIDR listesiyle ayarlayın."
}

$venvPython = Join-Path $projectDirectory ".venv\Scripts\python.exe"
$pythonExecutable = if (Test-Path -LiteralPath $venvPython) {
    $venvPython
} else {
    (Get-Command python -ErrorAction Stop).Source
}

& $pythonExecutable -c "import fastapi,httpx,uvicorn,pydantic" 2>$null
if ($LASTEXITCODE -ne 0) {
    throw 'Python bağımlılıkları eksik. Önce: python -m pip install -e ".[dev]"'
}

$ccoSource = if ($env:DEMO_CCO_DIR) { $env:DEMO_CCO_DIR } else { Join-Path $projectDirectory "data\cco" }
$ccoCount = @(Get-ChildItem -LiteralPath $ccoSource -Filter "*.json" -File -ErrorAction SilentlyContinue).Count

Write-Host ""
Write-Host "MedicalAI Digital Twin hazır: http://${BindHost}:$Port" -ForegroundColor Green
Write-Host "CCO kaydı: $ccoCount · Kaynak: $ccoSource" -ForegroundColor DarkGray
Write-Host "Durdurmak için Ctrl+C kullanın." -ForegroundColor DarkGray
Write-Host ""

Push-Location -LiteralPath $projectDirectory
try {
    $serverArguments = @(
        "-X", "utf8", "-m", "uvicorn", "demo.patient_chat.app:app",
        "--host", $BindHost, "--port", $Port, "--no-access-log"
    )
    if (Test-Path -LiteralPath $settingsFile) {
        $serverArguments += @("--env-file", $settingsFile)
        Write-Host ".env ayarları yüklendi." -ForegroundColor DarkGray
    }
    & $pythonExecutable @serverArguments
} finally {
    Pop-Location
}
