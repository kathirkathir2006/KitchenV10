param([string]$Kernel = "kathiresannatarajan/kiq-v5-real-photo-v1", [int]$MaxMinutes = 300)
$env:PYTHONIOENCODING = "utf-8"
$kaggle = "C:\Users\kathi\AppData\Roaming\Python\Python313\Scripts\kaggle.exe"
$deadline = (Get-Date).AddMinutes($MaxMinutes)
while ((Get-Date) -lt $deadline) {
    $s = (& $kaggle kernels status $Kernel 2>&1 | Out-String).Trim()
    Write-Output "[$(Get-Date -Format s)] $s"
    if ($s -match "COMPLETE|ERROR|CANCEL") { break }
    Start-Sleep -Seconds 120
}
