param(
  [int[]]$Ports = @(5173, 8010)
)

$ErrorActionPreference = "Stop"

foreach ($port in $Ports) {
  $connections = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
  foreach ($processId in ($connections | Select-Object -ExpandProperty OwningProcess -Unique)) {
    $process = Get-CimInstance Win32_Process -Filter "ProcessId=$processId" -ErrorAction SilentlyContinue
    if ($process.CommandLine -match "uvicorn|app\.main:app|vite|node_modules") {
      Stop-Process -Id $processId -Force
      Write-Host "Stopped PID=$processId on port $port"
    } else {
      Write-Warning "Skipped unrelated PID=$processId on port $port"
    }
  }
}
