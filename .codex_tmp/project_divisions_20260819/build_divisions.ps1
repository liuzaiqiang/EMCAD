$ErrorActionPreference = 'Stop'

function Decode-Items([string]$encoded) {
    return @($encoded.Split(',') | ForEach-Object {
        [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($_))
    })
}

$task1 = Decode-Items '5Zub6K+t6K+t5paZ6YeH6ZuG,5paH5pys6K+t5paZ5riF5rSX,6K+t6Z+z5pWw5o2u5pW055CG,5Zu+5YOP5pWw5o2u5b2S5qGj,5bmz6KGM6K+t5paZ5p6E5bu6,6K+t6Z+z6L2s5YaZ5qCH5rOo,5Zu+5paH5a+56b2Q5qCH5rOo,5qCH5rOo6LSo6YeP5aSN5qC4,5pWw5o2u5a6J5YWo5a6h5p+l,6K6t57uD6ZuG5YiS5YiG,5YWx5Lqr6KGo5b6B5a2m5Lmg,5L2O6LWE5rqQ6L+B56e7,5bCR5qC35pys6YCC6YWN,5byx55uR552j6K6t57uD,5pWw5o2u5aKe5by6562W55Wl,6K+t56eN6K+G5Yir5bu65qih,6K+t6Z+z6K+G5Yir5bu65qih,5aSa5paH56eNT0NS,5aSa6K+t57+76K+R5bu65qih,6K+t6Z+z5ZCI5oiQ5bu65qih,6Leo6K+t5qOA57Si5bu65qih,6K+t6Z+z5paH5pys5a+56b2Q,5Zu+5paH6K+t5LmJ5a+56b2Q,5Y6f5a2Q6IO95Yqb5bCB6KOF,57uf5LiA5o6l5Y+j6K6+6K6h,6IO95Yqb5rOo5YaM566h55CG,5bqV5bqn5pyN5Yqh5byA5Y+R,6L+Q6KGM55uR5rWL5byA5Y+R,6Leo6K++6aKY6IGU6LCD,6aqM5pS25p2Q5paZ57yW5Yi2'
$task2 = Decode-Items '56uv5L6n57qm5p2f5YiG5p6Q,6L276YeP57uT5p6E6K+G6K6h,5bGC57qn5Ymq5p6d,5rOo5oSP5aS06KOB5Ymq,RkZO6YCa6YGT5Y6L57yp,5pWZ5biI5qih5Z6L5p6E5bu6,6L6T5Ye655+l6K+G6JK46aaP,54m55b6B55+l6K+G6JK46aaP,5YWz57O755+l6K+G6JK46aaP,6JK46aaP57K+5bqm5oGi5aSN,5L2O5q+U54m56YeP5YyW,6YeP5YyW6K+v5beu5qCh5YeG,56Gs5Lu25oSf55+l5pCc57Si,566X5a2Q6YCC6YWN5LyY5YyW,56uv5L6n6YOo572y6aqM6K+B,5Zub6K+t5pWw5o2u6YeH6ZuG,5qih57OK5oyH5Luk5qCH5rOo,5LiJ5qih5oCB57yW56CB,6Leo5qih5oCB5a+56b2Q,6K+t5LmJ6J6N5ZCI5bu65qih,5oSP5Zu+6IGU5ZCI6K+G5Yir,5qe95L2N6IGU5ZCI6K+G5Yir,5qih57OK54q25oCB6K+G5Yir,5Y6G5Y+y5Lqk5LqS5qOA57Si,5Yqo5oCB54q25oCB5bu65qih,5Li75Yqo5r6E5riF562W55Wl,5LiK5LiL5paH6KGl5YWo,5YCZ6YCJ56Gu6K6k5py65Yi2,55So5oi357qg5YGP5a2m5Lmg,5LiJ6L2u5Lqk5LqS5rWL6K+V'
$task3 = Decode-Items '5Zub6K+t6IO95Yqb5o6l5YWl,5pm66IO95L2T5bCB6KOF,6IO95Yqb5YWD5pWw5o2u,5rOo5YaM5Y+R546w5Y2P6K6u,5raI5oGv6YCa5L+h5qih5Z2X,54q25oCB5ZCM5q2l5qih5Z2X,5YWx5Lqr6K6w5b+G5qih5Z2X,5p2D6ZmQ5o6n5Yi25qih5Z2X,57uf5LiA5pyN5Yqh5aWR57qm,REFH5Lu75Yqh5bu65qih,5L6d6LWW5YWz57O76Kej5p6Q,5Yqo5oCB6Lev55Sx562W55Wl,5Liy6KGM6LCD5bqm5a6e546w,5bm26KGM6LCD5bqm5a6e546w,57uT5p6c6aqM6K+B5py65Yi2,57uT5p6c5Y+N6aaI6Zet546v,5a6h6K6h5pel5b+X6K6w5b2V,6LaF5pe25aSE55CG5py65Yi2,5aSx6LSl6YeN6K+V5py65Yi2,5Yay56qB5Zue5rua5py65Yi2,5pu/5Luj6Lev5b6E6KeE5YiS,5YWz6ZSu566X5a2Q5YmW5p6Q,UlZW5ZCR6YeP5LyY5YyW,5bqU55So6YCC6YWN5Zmo5byA5Y+R,MjDnp43mnI3liqHmjqXlhaU=,UklTQy1W5bmz5Y+w56e75qSN,5bel5YW36ZO+5qGG5p626YCC6YWN,56uv5L6n5a6e5py66YOo572y,5LiJ57G75Zy65pmv56S66IyD,55So5oi35rWL6K+V5py65Yi2'

if (($task1.Count -ne 30) -or ($task2.Count -ne 30) -or ($task3.Count -ne 30)) {
    throw 'Each task must contain exactly 30 divisions.'
}

foreach ($task in @($task1, $task2, $task3)) {
    if (($task | Where-Object { $_.Length -gt 10 }).Count -gt 0) {
        throw 'A division exceeds 10 characters.'
    }
    if (($task | Select-Object -Unique).Count -ne 30) {
        throw 'Duplicate division found in a task.'
    }
}

$headers = Decode-Items '6K++6aKY5LiA5YiG5bel,6K++6aKY5LqM5YiG5bel,6K++6aKY5LiJ5YiG5bel'
$outputDir = Join-Path (Get-Location) 'outputs\01a00a71-7343-74e3-b3c1-b9774898c53b'
$outputPath = Join-Path $outputDir ([Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('6K++6aKY5YiG5bel5riF5Y2VLnhsc3g=')))
New-Item -ItemType Directory -Path $outputDir -Force | Out-Null

$excel = $null
$book = $null
$sheet = $null
try {
    $excel = New-Object -ComObject Excel.Application
    $excel.Visible = $false
    $excel.DisplayAlerts = $false
    $book = $excel.Workbooks.Add()
    $sheet = $book.Worksheets.Item(1)
    $sheet.Name = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('6K++6aKY5YiG5bel'))
    $sheet.Activate() | Out-Null
    $sheet.Application.ActiveWindow.DisplayGridlines = $false

    $headerValues = New-Object 'object[,]' 1, 3
    for ($column = 0; $column -lt 3; $column++) {
        $headerValues[0, $column] = $headers[$column]
    }
    $sheet.Range('A1:C1').Value2 = $headerValues
    $values = New-Object 'object[,]' 30, 3
    for ($row = 0; $row -lt 30; $row++) {
        $values[$row, 0] = $task1[$row]
        $values[$row, 1] = $task2[$row]
        $values[$row, 2] = $task3[$row]
    }
    $sheet.Range('A2:C31').Value2 = $values

    $header = $sheet.Range('A1:C1')
    $header.Font.Name = 'Microsoft YaHei'
    $header.Font.Size = 11
    $header.Font.Bold = $true
    $header.Font.Color = 0xFFFFFF
    $header.Interior.Color = 0x1F4E78
    $header.HorizontalAlignment = -4108
    $header.VerticalAlignment = -4108
    $header.RowHeight = 30

    $body = $sheet.Range('A2:C31')
    $body.Font.Name = 'Microsoft YaHei'
    $body.Font.Size = 10.5
    $body.HorizontalAlignment = -4108
    $body.VerticalAlignment = -4108
    $body.RowHeight = 24
    $body.Borders.LineStyle = 1
    $body.Borders.Color = 0xD9E2F3
    $body.Borders.Weight = 2
    $sheet.Range('A2:A31').Interior.Color = 0xEAF3F8
    $sheet.Range('B2:B31').Interior.Color = 0xF3F7FB
    $sheet.Range('C2:C31').Interior.Color = 0xEAF3F8
    $sheet.Range('A1:C31').Borders.LineStyle = 1
    $sheet.Range('A1:C31').Borders.Color = 0xB4C7E7
    $sheet.Range('A1:C31').AutoFilter()
    $sheet.Columns.Item(1).ColumnWidth = 18
    $sheet.Columns.Item(2).ColumnWidth = 18
    $sheet.Columns.Item(3).ColumnWidth = 18
    $sheet.Range('A1:C31').WrapText = $false
    $sheet.Application.ActiveWindow.SplitRow = 1
    $sheet.Application.ActiveWindow.FreezePanes = $true
    $book.SaveAs($outputPath, 51)
    $book.Close($true)
    $excel.Quit()
    Write-Output ('OUTPUT=' + $outputPath)
    Write-Output ('COUNTS=' + ($task1.Count) + ',' + ($task2.Count) + ',' + ($task3.Count))
    Write-Output ('MAXLEN=' + (($task1 + $task2 + $task3 | ForEach-Object { $_.Length } | Measure-Object -Maximum).Maximum))
}
finally {
    if ($book -ne $null) { try { $book.Close($false) } catch {} }
    if ($excel -ne $null) { try { $excel.Quit() } catch {} }
    foreach ($com in @($sheet, $book, $excel)) {
        if ($com -ne $null) { try { [void][Runtime.InteropServices.Marshal]::ReleaseComObject($com) } catch {} }
    }
    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()
}
