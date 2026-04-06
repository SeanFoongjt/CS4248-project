$inputPath = "D:\NUS\Year 3\Sem 2\CS4248\CS4248-project\Sarcasm_Headlines_Dataset_With_Metadata.json"
$outputPath = "D:\NUS\Year 3\Sem 2\CS4248\CS4248-project\Sarcasm_Headlines_Dataset_With_Metadata_no_encoding_issue_rows.json"

$patterns = @(
    "Ã",
    "Â",
    "â€™",
    "â€œ",
    "â€",
    "�",
    "&#39;",
    "&amp;",
    "&quot;",
    "&lt;",
    "&gt;"
)

$keptLines = New-Object System.Collections.Generic.List[string]
$totalRows = 0
$removedRows = 0
$keptRows = 0
$removedOnion = 0
$removedHuff = 0
$keptOnion = 0
$keptHuff = 0

Get-Content $inputPath | ForEach-Object {
    if ([string]::IsNullOrWhiteSpace($_)) { return }

    $totalRows++
    $line = [string]$_
    $obj = $line | ConvertFrom-Json

    $hasIssue = $false
    foreach ($pattern in $patterns) {
        if ($line.Contains($pattern)) {
            $hasIssue = $true
            break
        }
    }

    if ($hasIssue) {
        $removedRows++
        if ([string]$obj.source -eq "onion") { $removedOnion++ }
        elseif ([string]$obj.source -eq "huff") { $removedHuff++ }
        return
    }

    $keptLines.Add($line)
    $keptRows++
    if ([string]$obj.source -eq "onion") { $keptOnion++ }
    elseif ([string]$obj.source -eq "huff") { $keptHuff++ }
}

$keptLines | Set-Content $outputPath -Encoding UTF8

Write-Output "INPUT_ROWS=$totalRows"
Write-Output "REMOVED_ROWS=$removedRows"
Write-Output "KEPT_ROWS=$keptRows"
Write-Output "REMOVED_ONION_ROWS=$removedOnion"
Write-Output "REMOVED_HUFF_ROWS=$removedHuff"
Write-Output "KEPT_ONION_ROWS=$keptOnion"
Write-Output "KEPT_HUFF_ROWS=$keptHuff"
Write-Output "OUTPUT=$outputPath"
