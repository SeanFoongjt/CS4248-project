$datasetPath = "D:\NUS\Year 3\Sem 2\CS4248\CS4248-project\Sarcasm_Headlines_Dataset_With_Metadata.json"
$cleanOnionPath = "D:\NUS\Year 3\Sem 2\CS4248\CS4248-project\extra_onion_article_metadata_clean_textfixed.jsonl"
$outputPath = "D:\NUS\Year 3\Sem 2\CS4248\CS4248-project\Sarcasm_Headlines_Dataset_With_Metadata_plus_extra_onion.json"

$idPattern = "-(\d{9,12})(?:/amp)?(?:[/?#].*)?$"

function Get-OnionId {
    param([string]$Link)
    if ([string]::IsNullOrWhiteSpace($Link)) { return $null }
    if ($Link -match $idPattern) { return $matches[1] }
    return $null
}

$existingOnionIds = @{}
$totalDatasetRows = 0
$existingOnionRows = 0

Get-Content $datasetPath | ForEach-Object {
    if ([string]::IsNullOrWhiteSpace($_)) { return }
    $totalDatasetRows++
    $obj = $_ | ConvertFrom-Json
    if ([string]$obj.source -ne "onion") { return }
    $existingOnionRows++
    $onionId = Get-OnionId ([string]$obj.article_link)
    if ($null -ne $onionId) {
        $existingOnionIds[$onionId] = $true
    }
}

$mergedLines = New-Object System.Collections.Generic.List[string]
Get-Content $datasetPath | ForEach-Object {
    if (-not [string]::IsNullOrWhiteSpace($_)) {
        $mergedLines.Add($_)
    }
}

$cleanRowsRead = 0
$addedRows = 0
$skippedExistingId = 0
$skippedMissingId = 0

Get-Content $cleanOnionPath | ForEach-Object {
    if ([string]::IsNullOrWhiteSpace($_)) { return }

    $cleanRowsRead++
    $obj = $_ | ConvertFrom-Json
    $onionId = Get-OnionId ([string]$obj.article_link)

    if ($null -eq $onionId) {
        $skippedMissingId++
        return
    }

    if ($existingOnionIds.ContainsKey($onionId)) {
        $skippedExistingId++
        return
    }

    $obj | Add-Member -NotePropertyName is_sarcastic -NotePropertyValue 1 -Force
    $obj.PSObject.Properties.Remove("fetch_error")
    $obj.PSObject.Properties.Remove("parse_error")

    $mergedLines.Add(($obj | ConvertTo-Json -Compress -Depth 10))
    $existingOnionIds[$onionId] = $true
    $addedRows++
}

$mergedLines | Set-Content $outputPath -Encoding UTF8

Write-Output "DATASET_ROWS_BEFORE=$totalDatasetRows"
Write-Output "EXISTING_ONION_ROWS=$existingOnionRows"
Write-Output "CLEAN_ROWS_READ=$cleanRowsRead"
Write-Output "ADDED_ROWS=$addedRows"
Write-Output "SKIPPED_EXISTING_ID=$skippedExistingId"
Write-Output "SKIPPED_MISSING_ID=$skippedMissingId"
Write-Output "DATASET_ROWS_AFTER=$($mergedLines.Count)"
Write-Output "OUTPUT=$outputPath"
