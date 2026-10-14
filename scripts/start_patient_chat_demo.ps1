[CmdletBinding()]
param(
    [ValidateRange(1024, 65535)][int]$Port = 8092,
    [ValidateRange(1024, 65535)][int]$TunnelPort = 18000
)

$ErrorActionPreference = "Stop"
$projectDirectory = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$privateCcoDirectory = Join-Path $projectDirectory "data\evaluation\patient_centric_private\3_hasta_orijinal_ve_fact\orijinal_cco"
$requiredFiles = @(
    "response_1791739336553.json",
    "38c788d204d3-v2.json",
    "461429d00de8-v2.json"
)
foreach ($fileName in $requiredFiles) {
    $path = Join-Path $privateCcoDirectory $fileName
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Demo hasta kaydı bulunamadı: $path"
    }
}

$venvPython = Join-Path $projectDirectory ".venv\Scripts\python.exe"
$pythonExecutable = if (Test-Path -LiteralPath $venvPython) {
    $venvPython
} else {
    (Get-Command python -ErrorAction Stop).Source
}

& $pythonExecutable -c "import fastapi,httpx,uvicorn,pydantic" 2>$null
if ($LASTEXITCODE -ne 0) {
    throw "Python bağımlılıkları eksik. Proje bağımlılıklarını kurup yeniden deneyin."
}

$modelUrl = "http://127.0.0.1:$TunnelPort"
$tunnelProcess = $null
try {
    $modelReady = $false
    try {
        $null = Invoke-RestMethod -Uri "$modelUrl/v1/models" -TimeoutSec 2
        $modelReady = $true
    } catch {
        $forwardSpec = "127.0.0.1:${TunnelPort}:127.0.0.1:8000"
        $sshArguments = @(
            "-N", "-o", "BatchMode=yes", "-o", "ExitOnForwardFailure=yes",
            "-o", "ServerAliveInterval=30", "-L", $forwardSpec,
            "arge-adm@185.48.24.103"
        )
        $tunnelProcess = Start-Process -FilePath "ssh" -ArgumentList $sshArguments `
            -WindowStyle Hidden -PassThru
        for ($attempt = 0; $attempt -lt 30; $attempt++) {
            if ($tunnelProcess.HasExited) {
                throw "SSH tüneli açılamadı. SSH anahtarını ve GPU sunucusunu kontrol edin."
            }
            try {
                $null = Invoke-RestMethod -Uri "$modelUrl/v1/models" -TimeoutSec 2
                $modelReady = $true
                break
            } catch {
                Start-Sleep -Milliseconds 500
            }
        }
    }
    if (-not $modelReady) {
        throw "Gemma servisine SSH tüneli üzerinden ulaşılamadı."
    }

    $env:DEMO_VLLM_BASE_URL = $modelUrl
    $env:DEMO_MODEL = "gemma4-26b-a4b"
    Write-Host ""
    Write-Host "MedicalAI demo hazır: http://127.0.0.1:$Port" -ForegroundColor Green
    Write-Host "Durdurmak için Ctrl+C kullanın." -ForegroundColor DarkGray
    Write-Host ""
    Push-Location -LiteralPath $projectDirectory
    try {
        & $pythonExecutable -X utf8 -m uvicorn demo.patient_chat.app:app `
            --host 127.0.0.1 --port $Port --no-access-log
    } finally {
        Pop-Location
    }
} finally {
    if ($null -ne $tunnelProcess -and -not $tunnelProcess.HasExited) {
        Stop-Process -Id $tunnelProcess.Id
    }
}
