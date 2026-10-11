[CmdletBinding()]
param(
    [string]$BaseUrl = "https://wag.kendineiyibak.app",

    [Parameter(Mandatory = $true)]
    [string]$SubjectIdsPath,

    [ValidateSet("user-chat", "physician-view", "periodic-review")]
    [string]$Purpose = "user-chat",

    [string]$OutputDirectory = (Join-Path $PSScriptRoot "..\data\cco"),

    [string]$AsOf,

    [string]$AccessToken = $env:KIB_ACCESS_TOKEN
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($AccessToken)) {
    $secureToken = Read-Host "Prod access token" -AsSecureString
    $AccessToken = [System.Net.NetworkCredential]::new("", $secureToken).Password
}
if ([string]::IsNullOrWhiteSpace($AccessToken)) {
    throw "An access token is required."
}

$resolvedSubjectIdsPath = (Resolve-Path -LiteralPath $SubjectIdsPath).Path
$resolvedOutputDirectory = [System.IO.Path]::GetFullPath($OutputDirectory)
New-Item -ItemType Directory -Force -Path $resolvedOutputDirectory | Out-Null

$subjectIds = Get-Content -LiteralPath $resolvedSubjectIdsPath |
    ForEach-Object { $_.Trim() } |
    Where-Object { $_ -and -not $_.StartsWith("#") } |
    Select-Object -Unique

if (-not $subjectIds) {
    throw "No subject IDs were found in $resolvedSubjectIdsPath."
}

$validSubjectIds = foreach ($subjectId in $subjectIds) {
    $parsed = [Guid]::Empty
    if (-not [Guid]::TryParse($subjectId, [ref]$parsed)) {
        throw "Invalid subject ID: $subjectId"
    }
    $parsed.ToString()
}

$endpoint = "$($BaseUrl.TrimEnd('/'))/api/clinical-insight-service/cco"
$headers = @{ Authorization = "Bearer $AccessToken" }
$succeeded = 0
$failed = @()

foreach ($subjectId in $validSubjectIds) {
    $body = @{
        subjectId = $subjectId
        purpose = $Purpose
        correlationId = "selective-v2-capture-$([Guid]::NewGuid().ToString('N'))"
    }
    if (-not [string]::IsNullOrWhiteSpace($AsOf)) {
        $body.asOf = $AsOf
    }

    try {
        $response = Invoke-RestMethod `
            -Method Post `
            -Uri $endpoint `
            -Headers $headers `
            -ContentType "application/json" `
            -Body ($body | ConvertTo-Json -Compress)
        # Each capture is immutable; never replace an earlier patient's snapshot.
        $captureName = "$subjectId-$([DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfffZ'))-$([Guid]::NewGuid().ToString('N')).json"
        $target = Join-Path $resolvedOutputDirectory $captureName
        $response |
            ConvertTo-Json -Depth 100 |
            Set-Content -LiteralPath $target -Encoding utf8
        $succeeded++
        Write-Host "Saved $subjectId"
    }
    catch {
        $failed += [pscustomobject]@{
            subjectId = $subjectId
            error = $_.Exception.Message
        }
        Write-Warning "Failed $subjectId"
    }
}

[pscustomobject]@{
    endpoint = $endpoint
    requested = $validSubjectIds.Count
    succeeded = $succeeded
    failed = $failed
    outputDirectory = $resolvedOutputDirectory
}

