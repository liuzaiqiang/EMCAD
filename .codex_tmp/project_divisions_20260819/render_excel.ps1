$ErrorActionPreference = 'Stop'
$fileName = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('6K++6aKY5YiG5bel5riF5Y2VLnhsc3g='))
$xlsx = Join-Path (Get-Location) ('outputs\01a00a71-7343-74e3-b3c1-b9774898c53b\' + $fileName)
$pdf = Join-Path (Get-Location) '.codex_tmp\project_divisions_20260819\division_preview.pdf'
$excel = $null
$book = $null
try {
    $excel = New-Object -ComObject Excel.Application
    $excel.Visible = $false
    $excel.DisplayAlerts = $false
    $book = $excel.Workbooks.Open($xlsx, $false, $true)
    $sheet = $book.Worksheets.Item(1)
    $sheet.PageSetup.Orientation = 2
    $sheet.PageSetup.FitToPagesWide = 1
    $sheet.PageSetup.FitToPagesTall = 1
    $sheet.PageSetup.Zoom = $false
    $sheet.PageSetup.LeftMargin = $excel.InchesToPoints(0.25)
    $sheet.PageSetup.RightMargin = $excel.InchesToPoints(0.25)
    $sheet.PageSetup.TopMargin = $excel.InchesToPoints(0.35)
    $sheet.PageSetup.BottomMargin = $excel.InchesToPoints(0.35)
    $book.ExportAsFixedFormat(0, $pdf)
    $book.Close($false)
    $excel.Quit()
    Write-Output ('PDF=' + $pdf)
}
finally {
    if ($book -ne $null) { try { $book.Close($false) } catch {} }
    if ($excel -ne $null) { try { $excel.Quit() } catch {} }
}
