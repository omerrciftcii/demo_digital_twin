[CmdletBinding()]
param(
    [ValidateRange(1024, 65535)][int]$Port = 8092,
    [ValidateRange(1024, 65535)][int]$TunnelPort = 18000,
    [string]$CcoDirectory = "",
    [string]$GpuHost = "arge-adm@185.48.24.103",
    [ValidateRange(1, 65535)][int]$GpuServicePort = 8000
)

$ErrorActionPreference = "Stop"

$modelUrl = "http://127.0.0.1:$TunnelPort"
try {
    $null = Invoke-RestMethod -Uri "$modelUrl/v1/models" -TimeoutSec 2
} catch {
    $tunnelCommand = "ssh -N -L ${TunnelPort}:127.0.0.1:${GpuServicePort} $GpuHost"
    throw "GPU Gemma tüneli açık değil. Ayrı bir PowerShell penceresinde çalıştırın ve SSH parolasını yalnız o isteme yazın: $tunnelCommand"
}

$env:DEMO_VLLM_BASE_URL = $modelUrl
$launcher = Join-Path $PSScriptRoot "start_digital_twin_demo.ps1"
& $launcher -Port $Port -CcoDirectory $CcoDirectory
exit $LASTEXITCODE
