[CmdletBinding()]
param(
    [ValidateRange(1024, 65535)][int]$Port = 8093,
    [ValidateRange(1024, 65535)][int]$TunnelPort = 18000,
    [ValidatePattern('^\d{1,3}(\.\d{1,3}){3}/\d{1,2}$')][string]$OfficeCidr = "10.34.14.0/23",
    [string]$AccessToken = ""
)

$ErrorActionPreference = "Stop"
$projectDirectory = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))

function Test-IpInCidr {
    param([string]$IpAddress, [string]$Cidr)

    $parts = $Cidr.Split("/")
    $networkBytes = [System.Net.IPAddress]::Parse($parts[0]).GetAddressBytes()
    $addressBytes = [System.Net.IPAddress]::Parse($IpAddress).GetAddressBytes()
    $prefixLength = [int]$parts[1]
    if ($networkBytes.Length -ne 4 -or $addressBytes.Length -ne 4) {
        return $false
    }
    for ($index = 0; $index -lt 4; $index++) {
        $bits = [Math]::Max(0, [Math]::Min(8, $prefixLength - ($index * 8)))
        $mask = if ($bits -eq 0) { 0 } else { (0xFF -shl (8 - $bits)) -band 0xFF }
        if (($networkBytes[$index] -band $mask) -ne ($addressBytes[$index] -band $mask)) {
            return $false
        }
    }
    return $true
}

$network = Get-NetIPConfiguration |
    Where-Object { $_.NetAdapter.Status -eq "Up" -and $_.IPv4Address } |
    ForEach-Object {
        foreach ($address in $_.IPv4Address) {
            [pscustomobject]@{
                Interface = $_.InterfaceAlias
                Address = $address.IPAddress
            }
        }
    } |
    Where-Object { Test-IpInCidr -IpAddress $_.Address -Cidr $OfficeCidr } |
    Select-Object -First 1

if ($null -eq $network) {
    throw "Bu bilgisayar $OfficeCidr ofis ağına bağlı değil. Mobil API başlatılmadı."
}

$modelUrl = "http://127.0.0.1:$TunnelPort"
try {
    $null = Invoke-RestMethod -Uri "$modelUrl/v1/models" -TimeoutSec 3
} catch {
    throw "GPU SSH tüneli bulunamadı. Önce şu komutu ayrı bir pencerede çalıştırın: ssh -N -L ${TunnelPort}:127.0.0.1:8000 arge-adm@185.48.24.103"
}

if (-not $AccessToken) {
    $AccessToken = [Convert]::ToHexString([Security.Cryptography.RandomNumberGenerator]::GetBytes(12)).ToLowerInvariant()
}

$pythonExecutable = (Get-Command python -ErrorAction Stop).Source
& $pythonExecutable -c "import fastapi,httpx,uvicorn,pydantic" 2>$null
if ($LASTEXITCODE -ne 0) {
    throw "Python bağımlılıkları eksik. Proje bağımlılıklarını kurup yeniden deneyin."
}

$env:DEMO_VLLM_BASE_URL = $modelUrl
$env:DEMO_MODEL = "gemma4-26b-a4b"
$env:MOBILE_DEMO_TOKEN = $AccessToken
$env:MOBILE_ALLOWED_CIDRS = "$OfficeCidr,127.0.0.0/8"
$env:MOBILE_ALLOWED_HOSTS = "$($network.Address),127.0.0.1,localhost"

Write-Host ""
Write-Host "MedicalAI mobil API hazır" -ForegroundColor Green
Write-Host "Sunucu: http://$($network.Address):$Port" -ForegroundColor Cyan
Write-Host "Erişim kodu: $AccessToken" -ForegroundColor Yellow
Write-Host "İzinli ağ: $OfficeCidr" -ForegroundColor DarkGray
Write-Host "Durdurmak için Ctrl+C kullanın." -ForegroundColor DarkGray
Write-Host ""

Push-Location -LiteralPath $projectDirectory
try {
    & $pythonExecutable -X utf8 -m uvicorn demo.mobile_chat_api.app:app `
        --host 0.0.0.0 --port $Port --no-access-log
} finally {
    Pop-Location
}
