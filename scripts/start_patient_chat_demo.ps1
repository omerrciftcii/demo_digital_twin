[CmdletBinding()]
param(
    [ValidateRange(1024, 65535)][int]$Port = 8092,
    [ValidateRange(1024, 65535)][int]$TunnelPort = 18000,
    [string]$CcoDirectory = ""
)

$ErrorActionPreference = "Stop"

if ($PSBoundParameters.ContainsKey("TunnelPort")) {
    Write-Warning "TunnelPort artık kullanılmıyor; OpenAI dışındaki demo modelleri OpenRouter üzerinden çalışır."
}

$launcher = Join-Path $PSScriptRoot "start_digital_twin_demo.ps1"
& $launcher -Port $Port -CcoDirectory $CcoDirectory
exit $LASTEXITCODE
