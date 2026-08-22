$ErrorActionPreference = 'Stop'
$fileName = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('6K++6aKY5YiG5bel5riF5Y2VLnhsc3g='))
$xlsx = Join-Path (Get-Location) ('outputs\01a00a71-7343-74e3-b3c1-b9774898c53b\' + $fileName)
$png = Join-Path (Get-Location) '.codex_tmp\project_divisions_20260819\division_preview.png'
$excel = $null
$book = $null
$chartObject = $null
try {
    $excel = New-Object -ComObject Excel.Application
    $excel.Visible = $false
    $excel.DisplayAlerts = $false
    $book = $excel.Workbooks.Open($xlsx, $false, $true)
    $sheet = $book.Worksheets.Item(1)
    $range = $sheet.Range('A1:C31')
    $range.CopyPicture(1, 2)
    $chartObject = $sheet.ChartObjects().Add(0, 0, $range.Width + 8, $range.Height + 8)
    $chartObject.Chart.Paste() | Out-Null
    $chartObject.Chart.Export($png, 'PNG') | Out-Null
    $book.Close($false)
    $excel.Quit()
    Write-Output ('PNG=' + $png)
}
finally {
    if ($book -ne $null) { try { $book.Close($false) } catch {} }
    if ($excel -ne $null) { try { $excel.Quit() } catch {} }
}
