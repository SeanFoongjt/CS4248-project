$datasetPath = "D:\NUS\Year 3\Sem 2\CS4248\CS4248-project\Sarcasm_Headlines_Dataset_With_Metadata_plus_extra_onion_deduped.json"
$huffPath = "D:\NUS\Year 3\Sem 2\CS4248\CS4248-project\news-data-scraping\extra_huff_article_metadata.jsonl"
$cleanHuffPath = "D:\NUS\Year 3\Sem 2\CS4248\CS4248-project\extra_huff_article_metadata_clean.jsonl"
$outputPath = "D:\NUS\Year 3\Sem 2\CS4248\CS4248-project\Sarcasm_Headlines_Dataset_With_Metadata_plus_extra_onion_huff_deduped.json"

$hexUsPattern = "_us_([0-9a-fA-F]{24})(?:[/?#_.-]|$)"
$hexFallbackPattern = "(?:^|[/?#_.=-])([0-9a-fA-F]{24})(?:[/?#_.-]|$)"
$numericPattern = "(?:^|[-_/])([0-9]{6,})(?:[/?#_.-]|$)"

function Get-HuffId {
    param([string]$Link)
    if ([string]::IsNullOrWhiteSpace($Link)) { return $null }
    if ($Link -match $hexUsPattern) { return $matches[1].ToLowerInvariant() }
    if ($Link -match $hexFallbackPattern) { return $matches[1].ToLowerInvariant() }
    if ($Link -match $numericPattern) { return "num:$($matches[1])" }
    return $null
}

function Normalize-Headline {
    param([string]$Headline)
    if ([string]::IsNullOrWhiteSpace($Headline)) { return $null }
    return (($Headline -replace "\s+", " ").Trim()).ToLowerInvariant()
}

$existingHuffIds = @{}
$existingHeadlineKeys = @{}
$mergedLines = New-Object System.Collections.Generic.List[string]
$datasetRowsBefore = 0

Get-Content $datasetPath | ForEach-Object {
    if ([string]::IsNullOrWhiteSpace($_)) { return }
    $datasetRowsBefore++
    $line = $_
    $obj = $line | ConvertFrom-Json
    $mergedLines.Add($line)

    $headlineKey = Normalize-Headline ([string]$obj.headline)
    if ($null -ne $headlineKey) {
        $existingHeadlineKeys[$headlineKey] = $true
    }

    if ([string]$obj.source -eq "huff") {
        $huffId = Get-HuffId ([string]$obj.article_link)
        if ($null -ne $huffId) {
            $existingHuffIds[$huffId] = $true
        }
    }
}

$cleanRowsRead = 0
$cleanRowsKept = 0
$skippedFetchError = 0
$skippedParseError = 0
$skippedExistingId = 0
$skippedDuplicateHeadline = 0
$skippedMissingId = 0

$cleanLines = New-Object System.Collections.Generic.List[string]

Get-Content $huffPath | ForEach-Object {
    if ([string]::IsNullOrWhiteSpace($_)) { return }
    $cleanRowsRead++
    $obj = $_ | ConvertFrom-Json

    $hasFetch = -not [string]::IsNullOrWhiteSpace([string]$obj.fetch_error)
    $hasParse = -not [string]::IsNullOrWhiteSpace([string]$obj.parse_error)
    if ($hasFetch) { $skippedFetchError++; return }
    if ($hasParse) { $skippedParseError++; return }

    $huffId = Get-HuffId ([string]$obj.article_link)
    if ($null -eq $huffId) { $skippedMissingId++; return }
    if ($existingHuffIds.ContainsKey($huffId)) { $skippedExistingId++; return }

    $headlineKey = Normalize-Headline ([string]$obj.headline)
    if ($null -ne $headlineKey -and $existingHeadlineKeys.ContainsKey($headlineKey)) {
        $skippedDuplicateHeadline++
        return
    }

    $obj | Add-Member -NotePropertyName is_sarcastic -NotePropertyValue 0 -Force
    $obj.PSObject.Properties.Remove("fetch_error")
    $obj.PSObject.Properties.Remove("parse_error")

    $json = $obj | ConvertTo-Json -Compress -Depth 10
    $cleanLines.Add($json)
    $mergedLines.Add($json)

    $existingHuffIds[$huffId] = $true
    if ($null -ne $headlineKey) {
        $existingHeadlineKeys[$headlineKey] = $true
    }
    $cleanRowsKept++
}

$cleanLines | Set-Content $cleanHuffPath -Encoding UTF8
$mergedLines | Set-Content $outputPath -Encoding UTF8

Write-Output "DATASET_ROWS_BEFORE=$datasetRowsBefore"
Write-Output "HUFF_ROWS_READ=$cleanRowsRead"
Write-Output "HUFF_ROWS_KEPT=$cleanRowsKept"
Write-Output "SKIPPED_FETCH_ERROR=$skippedFetchError"
Write-Output "SKIPPED_PARSE_ERROR=$skippedParseError"
Write-Output "SKIPPED_EXISTING_ID=$skippedExistingId"
Write-Output "SKIPPED_DUPLICATE_HEADLINE=$skippedDuplicateHeadline"
Write-Output "SKIPPED_MISSING_ID=$skippedMissingId"
Write-Output "CLEAN_HUFF_OUTPUT=$cleanHuffPath"
Write-Output "MERGED_OUTPUT=$outputPath"
Write-Output "DATASET_ROWS_AFTER=$($mergedLines.Count)"
