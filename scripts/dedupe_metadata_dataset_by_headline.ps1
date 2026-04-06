$inputPath = "D:\NUS\Year 3\Sem 2\CS4248\CS4248-project\Sarcasm_Headlines_Dataset_With_Metadata_plus_extra_onion.json"
$outputPath = "D:\NUS\Year 3\Sem 2\CS4248\CS4248-project\Sarcasm_Headlines_Dataset_With_Metadata_plus_extra_onion_deduped.json"

function Normalize-Headline {
    param([string]$Headline)
    if ([string]::IsNullOrWhiteSpace($Headline)) { return $null }
    return (($Headline -replace "\s+", " ").Trim()).ToLowerInvariant()
}

$seen = @{}
$keptLines = New-Object System.Collections.Generic.List[string]
$totalRows = 0
$keptRows = 0
$removedRows = 0
$blankHeadlineRows = 0

Get-Content $inputPath | ForEach-Object {
    if ([string]::IsNullOrWhiteSpace($_)) { return }

    $totalRows++
    $line = $_
    $obj = $line | ConvertFrom-Json
    $headlineKey = Normalize-Headline ([string]$obj.headline)

    if ($null -eq $headlineKey) {
        $blankHeadlineRows++
        $keptLines.Add($line)
        $keptRows++
        return
    }

    if ($seen.ContainsKey($headlineKey)) {
        $removedRows++
        return
    }

    $seen[$headlineKey] = $true
    $keptLines.Add($line)
    $keptRows++
}

$keptLines | Set-Content $outputPath -Encoding UTF8

Write-Output "INPUT_ROWS=$totalRows"
Write-Output "KEPT_ROWS=$keptRows"
Write-Output "REMOVED_DUPLICATE_HEADLINE_ROWS=$removedRows"
Write-Output "BLANK_HEADLINE_ROWS_KEPT=$blankHeadlineRows"
Write-Output "OUTPUT=$outputPath"
