[CmdletBinding()]
param(
    [ValidateRange(1024, 65535)][int]$Port = 8093,
    [ValidatePattern('^\d{1,3}(\.\d{1,3}){3}/\d{1,2}$')][string]$OfficeCidr = "10.34.14.0/23"
)

$ErrorActionPreference = "Stop"
$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = [Security.Principal.WindowsPrincipal]::new($identity)
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw "Bu betiği Yönetici olarak açılmış PowerShell içinde çalıştırın."
}

$ruleName = "MedicalAI-Mobile-Demo-$Port"
$rule = Get-NetFirewallRule -Name $ruleName -ErrorAction SilentlyContinue
if ($null -eq $rule) {
    New-NetFirewallRule `
        -Name $ruleName `
        -DisplayName "MedicalAI Mobile Demo ($Port)" `
        -Description "Yalnızca ofis ağı $OfficeCidr için MedicalAI Android demo API erişimi." `
        -Direction Inbound `
        -Action Allow `
        -Protocol TCP `
        -LocalPort $Port `
        -RemoteAddress $OfficeCidr `
        -Profile Any | Out-Null
} else {
    Set-NetFirewallRule -Name $ruleName -Enabled True -Action Allow -Profile Any | Out-Null
    $rule | Get-NetFirewallAddressFilter | Set-NetFirewallAddressFilter -RemoteAddress $OfficeCidr | Out-Null
    $rule | Get-NetFirewallPortFilter | Set-NetFirewallPortFilter -Protocol TCP -LocalPort $Port | Out-Null
}

Write-Host "Güvenlik duvarı hazır: TCP $Port yalnızca $OfficeCidr" -ForegroundColor Green
