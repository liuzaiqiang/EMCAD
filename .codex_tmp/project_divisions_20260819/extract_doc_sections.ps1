param(
    [Parameter(Mandatory = $true)]
    [string]$DocumentPath,
    [int]$ContextBefore = 1,
    [int]$ContextAfter = 12,
    [int]$StartCleanIndex = 0,
    [int]$EndCleanIndex = 0,
    [switch]$SkipTables
)

$keywordBase64 = @(
    '6K++6aKY5ZCN56ew', '5oC75L2T55uu5qCH', '56CU56m255uu5qCH',
    '5Li76KaB56CU56m25YaF5a65', '56CU56m25YaF5a65', '5ouf6Kej5Yaz',
    '5YWz6ZSu5oqA5pyv', '56CU56m25pa55rOV', '5oqA5pyv6Lev57q/',
    '5Yib5paw54K5', '6ICD5qC45oyH5qCH', '6aKE5pyf5oiQ5p6c',
    '6L+b5bqm5a6J5o6S', '5bm05bqm6K6h5YiS', '5Lu75Yqh5YiG5bel'
)
$keywords = $keywordBase64 | ForEach-Object {
    [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($_))
}

$word = $null
$document = $null

try {
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $word.DisplayAlerts = 0
    $document = $word.Documents.Open($DocumentPath, $false, $true)

    $paragraphs = New-Object System.Collections.Generic.List[string]
    foreach ($paragraph in $document.Paragraphs) {
        $text = $paragraph.Range.Text -replace '[\r\a\v\f]', ''
        $text = $text -replace '\s+', ' '
        $text = $text.Trim()
        if ($text.Length -gt 0) {
            $paragraphs.Add($text)
        }
    }

    $selected = New-Object 'System.Collections.Generic.SortedSet[int]'
    if (($StartCleanIndex -gt 0) -and ($EndCleanIndex -ge $StartCleanIndex)) {
        $start = [Math]::Max(0, $StartCleanIndex - 1)
        $end = [Math]::Min($paragraphs.Count - 1, $EndCleanIndex - 1)
        for ($index = $start; $index -le $end; $index++) {
            [void]$selected.Add($index)
        }
    }
    else {
        for ($index = 0; $index -lt $paragraphs.Count; $index++) {
            $line = $paragraphs[$index]
            $matched = $false
            foreach ($keyword in $keywords) {
                if ($line.Contains($keyword)) {
                    $matched = $true
                    break
                }
            }
            if ($matched) {
                $start = [Math]::Max(0, $index - $ContextBefore)
                $end = [Math]::Min($paragraphs.Count - 1, $index + $ContextAfter)
                for ($contextIndex = $start; $contextIndex -le $end; $contextIndex++) {
                    [void]$selected.Add($contextIndex)
                }
            }
        }
    }

    Write-Output ('DOCUMENT: ' + $DocumentPath)
    Write-Output ('PARAGRAPHS: ' + $paragraphs.Count)
    Write-Output '--- MATCHED PARAGRAPH CONTEXT ---'
    $previous = -2
    foreach ($index in $selected) {
        if ($index -gt $previous + 1) {
            Write-Output '...'
        }
        Write-Output ('[{0:D4}] {1}' -f ($index + 1), $paragraphs[$index])
        $previous = $index
    }

    if (-not $SkipTables) {
        Write-Output '--- TABLE SUMMARIES ---'
        for ($tableIndex = 1; $tableIndex -le $document.Tables.Count; $tableIndex++) {
            $table = $document.Tables.Item($tableIndex)
            $rows = New-Object System.Collections.Generic.List[string]
            try {
                for ($rowIndex = 1; $rowIndex -le $table.Rows.Count; $rowIndex++) {
                    $cells = New-Object System.Collections.Generic.List[string]
                    foreach ($cell in $table.Rows.Item($rowIndex).Cells) {
                        $cellText = $cell.Range.Text -replace '[\r\a\v\f]', ''
                        $cellText = $cellText -replace '\s+', ' '
                        $cells.Add($cellText.Trim())
                    }
                    $rows.Add(($cells -join ' | '))
                }
            }
            catch {
                continue
            }
            $tableText = $rows -join ' '
            $includeTable = $false
            foreach ($keyword in $keywords) {
                if ($tableText.Contains($keyword)) {
                    $includeTable = $true
                    break
                }
            }
            if ($includeTable) {
                Write-Output ('TABLE {0} ({1} rows)' -f $tableIndex, $table.Rows.Count)
                foreach ($row in $rows) {
                    Write-Output $row
                }
            }
        }
    }
}
finally {
    if ($document -ne $null) {
        $document.Close($false)
        [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($document)
    }
    if ($word -ne $null) {
        $word.Quit()
        [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($word)
    }
    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()
}
