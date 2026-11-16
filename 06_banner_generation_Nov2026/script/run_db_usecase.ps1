param(
    [string]$UseCase = "",
    [switch]$All,
    [string]$ApiBase = "http://127.0.0.1:8000",
    [ValidateSet("default", "quality", "eco", "custom", "all")]
    [string]$Mode = "quality",
    [int]$TargetVersions = 3,
    [int]$MaxAttempts = 4,
    [int]$RetryDelaySeconds = 5
)

$ErrorActionPreference = "Stop"

$useCases = @(
    "AUTO_TOPUP",
    "AUTOBILL",
    "BH_AN_NINH_MANG",
    "BILLPAY",
    "DAT_TAXI",
    "DICH_VU_THE",
    "ESAVING",
    "FLEX_FUND",
    "FLEXI",
    "GUI_FD_1_TY",
    "MAX_SAVING",
    "QUET_QR",
    "TACH_LENH_CHUYEN_TIEN",
    "THAU_CHI_TIN_CHAP",
    "TOPUP",
    "VE_MAY_BAY",
    "VE_XEM_PHIM",
    "VIETLOTT",
    "VOUCHER_LAI_SUAT"
)

function Select-UseCase {
    Write-Host ""
    Write-Host "===== CHON USE CASE DB =====" -ForegroundColor Cyan
    for ($index = 0; $index -lt $useCases.Count; $index++) {
        Write-Host ("{0,2}. {1}" -f ($index + 1), $useCases[$index])
    }

    $choice = Read-Host "Nhap so 1-$($useCases.Count)"
    $number = 0
    if (-not [int]::TryParse($choice, [ref]$number) -or $number -lt 1 -or $number -gt $useCases.Count) {
        throw "Lua chon '$choice' khong hop le."
    }
    return $useCases[$number - 1]
}

function Invoke-GenerateBatch {
    param(
        [string]$RequestName,
        [string]$Segment,
        [int]$MissingCount,
        [int]$Attempt
    )

    $body = @{
        request_name = $RequestName
        mode = $Mode
        custom_models = @()
        num_version_override = $MissingCount
    } | ConvertTo-Json -Depth 4

    Write-Host "Request: $RequestName (attempt $Attempt, xin $MissingCount anh)"

    try {
        $response = Invoke-RestMethod `
            -Uri "$ApiBase/generate" `
            -Method Post `
            -ContentType "application/json; charset=utf-8" `
            -Body $body
    }
    catch {
        $errorText = $_.Exception.Message
        for ($i = 1; $i -le $MissingCount; $i++) {
            Write-Host ("[{0}] anh {1}: LOI - API request failed: {2}" -f $Segment, $i, $errorText) -ForegroundColor Red
        }
        return 0
    }

    if (-not $response.ok) {
        $errorText = if ($response.error) { $response.error } else { "Unknown API error" }
        for ($i = 1; $i -le $MissingCount; $i++) {
            Write-Host ("[{0}] anh {1}: LOI - {2}" -f $Segment, $i, $errorText) -ForegroundColor Red
        }
        return 0
    }

    $succeeded = 0
    $results = @($response.results)

    foreach ($version in 1..$MissingCount) {
        $versionResults = @($results | Where-Object { [int]$_.version -eq $version })
        if ($versionResults.Count -eq 0) {
            Write-Host ("[{0}] anh {1}: LOI - API khong tra ket qua" -f $Segment, $version) -ForegroundColor Red
            continue
        }

        $errors = @($versionResults | Where-Object { -not $_.ok })
        if ($errors.Count -gt 0) {
            $details = ($errors | ForEach-Object {
                $model = if ($_.image_model) { $_.image_model } else { "unknown-model" }
                $message = if ($_.error) { $_.error } else { "Unknown generation error" }
                "${model}: $message"
            }) -join " | "
            Write-Host ("[{0}] anh {1}: LOI - {2}" -f $Segment, $version, $details) -ForegroundColor Red
        }
        else {
            $paths = ($versionResults | ForEach-Object { $_.path }) -join " | "
            Write-Host ("[{0}] anh {1}: DONE - {2}" -f $Segment, $version, $paths) -ForegroundColor Green
            $succeeded++
        }
    }

    return $succeeded
}

function Invoke-Segment {
    param(
        [string]$SelectedUseCase,
        [string]$Segment
    )

    $requestName = "DB_${SelectedUseCase}_1024x440_${Segment}"

    Write-Host ""
    Write-Host "===== $SelectedUseCase | $Segment =====" -ForegroundColor Cyan

    $done = 0
    $attempt = 1

    while ($done -lt $TargetVersions -and $attempt -le $MaxAttempts) {
        $missing = $TargetVersions - $done
        if ($attempt -gt 1) {
            Write-Host ("[{0}] con thieu {1} anh, cho {2}s roi thu lai (attempt {3}/{4})..." -f $Segment, $missing, $RetryDelaySeconds, $attempt, $MaxAttempts) -ForegroundColor Yellow
            Start-Sleep -Seconds $RetryDelaySeconds
        }

        $succeeded = Invoke-GenerateBatch -RequestName $requestName -Segment $Segment -MissingCount $missing -Attempt $attempt
        $done += $succeeded
        $attempt++
    }

    $failed = $TargetVersions - $done
    if ($failed -gt 0) {
        Write-Host ("[{0}] DUNG sau {1} attempt(s), con thieu {2}/{3} anh." -f $Segment, ($attempt - 1), $failed, $TargetVersions) -ForegroundColor Red
    }

    return [pscustomobject]@{ Done = $done; Failed = $failed }
}

if ($All) {
    $selectedUseCases = $useCases
}
else {
    if ([string]::IsNullOrWhiteSpace($UseCase)) {
        $UseCase = Select-UseCase
    }
    $UseCase = $UseCase.Trim().ToUpperInvariant()
    if ($UseCase -notin $useCases) {
        throw "Use case '$UseCase' khong ton tai. Chay script khong co -UseCase de xem danh sach."
    }
    $selectedUseCases = @($UseCase)
}

$totalDone = 0
$totalFailed = 0
$segments = @("MASS", "MASS_AFF", "AFF")

foreach ($selected in $selectedUseCases) {
    $useCaseDone = 0
    $useCaseFailed = 0
    Write-Host ""
    Write-Host "########################################" -ForegroundColor Yellow
    Write-Host "BAT DAU USE CASE: $selected" -ForegroundColor Yellow
    Write-Host "########################################" -ForegroundColor Yellow

    foreach ($segment in $segments) {
        $summary = Invoke-Segment -SelectedUseCase $selected -Segment $segment
        $useCaseDone += $summary.Done
        $useCaseFailed += $summary.Failed
        $totalDone += $summary.Done
        $totalFailed += $summary.Failed
    }

    Write-Host ""
    Write-Host "KET THUC $selected - DONE: $useCaseDone | LOI: $useCaseFailed" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "===== TONG KET =====" -ForegroundColor Cyan
Write-Host "DONE: $totalDone"
Write-Host "LOI : $totalFailed"

if ($totalFailed -gt 0) {
    exit 1
}
