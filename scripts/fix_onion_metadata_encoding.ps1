$in = "D:\NUS\Year 3\Sem 2\CS4248\CS4248-project\extra_onion_article_metadata_clean.jsonl"
$out = "D:\NUS\Year 3\Sem 2\CS4248\CS4248-project\extra_onion_article_metadata_clean_textfixed.jsonl"

Add-Type -AssemblyName System.Web

$fields = @("headline", "description", "author_description", "author_name", "article_section")

$replacements = @(
    @{ Bad = ([string][char]0x00E2) + ([char]0x20AC) + ([char]0x201D); Good = [string][char]0x2014 }, # â€” -> —
    @{ Bad = ([string][char]0x00E2) + ([char]0x20AC) + ([char]0x201C); Good = [string][char]0x2013 }, # â€“ -> –
    @{ Bad = ([string][char]0x00E2) + ([char]0x20AC) + ([char]0x2122); Good = [string][char]0x2019 }, # â€™ -> ’
    @{ Bad = ([string][char]0x00E2) + ([char]0x20AC) + ([char]0x02DC); Good = [string][char]0x2018 }, # â€˜ -> ‘
    @{ Bad = ([string][char]0x00E2) + ([char]0x20AC) + ([char]0x0153); Good = [string][char]0x201C }, # â€œ -> “
    @{ Bad = ([string][char]0x00E2) + ([char]0x20AC) + ([char]0x009D); Good = [string][char]0x201D }, # â€� -> ”
    @{ Bad = ([string][char]0x00E2) + ([char]0x20AC) + ([char]0x00A6); Good = [string][char]0x2026 }, # â€¦ -> …
    @{ Bad = ([string][char]0x00C2) + " "; Good = " " },                                               # Â  -> space
    @{ Bad = [string][char]0x00C2; Good = "" }                                                         # Â -> empty
)

$totalRows = 0
$changedRows = 0

Get-Content $in | ForEach-Object {
    if ([string]::IsNullOrWhiteSpace($_)) { return }

    $obj = $_ | ConvertFrom-Json
    $rowChanged = $false

    foreach ($field in $fields) {
        $value = $obj.$field
        if ($null -eq $value -or $value -isnot [string]) { continue }

        $fixed = [System.Web.HttpUtility]::HtmlDecode($value)
        foreach ($replacement in $replacements) {
            $fixed = $fixed.Replace($replacement.Bad, $replacement.Good)
        }

        if ($fixed -ne $value) {
            $obj.$field = $fixed
            $rowChanged = $true
        }
    }

    $totalRows++
    if ($rowChanged) { $changedRows++ }

    $obj | ConvertTo-Json -Compress -Depth 10
} | Set-Content $out -Encoding UTF8

Write-Output "TOTAL_ROWS=$totalRows"
Write-Output "CHANGED_ROWS=$changedRows"
Write-Output "OUTPUT=$out"
