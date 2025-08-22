<# 
  start.ps1
  Usage:
    .\start.ps1                 # normal start
    .\start.ps1 -Rebuild        # rebuild images
    .\start.ps1 -Fresh          # nuke volumes (fresh DB), then start
    .\start.ps1 -NoOpen         # don't open the browser
#>

param(
  [switch]$Rebuild,
  [switch]$Fresh,
  [switch]$NoOpen
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Parse-DotEnv($path) {
  $h = @{}
  if (Test-Path $path) {
    Get-Content $path | ForEach-Object {
      $line = $_.Trim()
      if ($line -and -not $line.StartsWith("#") -and $line.Contains("=")) {
        $k, $v = $line -split "=", 2
        $h[$k.Trim()] = $v.Trim()
      }
    }
  }
  return $h
}

function Wait-For-Health($url, $label, $retries = 60, $sleepSec = 2) {
  Write-Host "Waiting for $label at $url ..."
  for ($i=1; $i -le $retries; $i++) {
    try {
      $res = Invoke-WebRequest -UseBasicParsing -TimeoutSec 3 $url
      if ($res.StatusCode -ge 200 -and $res.StatusCode -lt 300) {
        Write-Host "✅ $label is ready."
        return
      }
    } catch { }
    Start-Sleep -Seconds $sleepSec
  }
  throw "❌ Timeout waiting for $label ($url)"
}

function Wait-For-ContainerHealthy($name, $retries = 60, $sleepSec = 2) {
  Write-Host "Waiting for container '$name' to be healthy..."
  for ($i=1; $i -le $retries; $i++) {
    $status = docker inspect -f "{{.State.Health.Status}}" $name 2>$null
    if ($status -eq "healthy") {
      Write-Host "✅ $name healthy."
      return
    }
    Start-Sleep -Seconds $sleepSec
  }
  throw "❌ Timeout waiting for $name to be healthy"
}

# --- script starts ---
cd "$PSScriptRoot"

# Load .env.docker for DB creds used by the DB container
$envFile = Join-Path $PSScriptRoot ".env.docker"
$dotenv  = Parse-DotEnv $envFile
$PGUSER  = $dotenv["PGUSER"]; if (-not $PGUSER) { $PGUSER = "postgres" }
$PGDB    = $dotenv["PGDATABASE"]; if (-not $PGDB) { $PGDB = "windsor" }

if ($Fresh) {
  Write-Host "↓↓ Fresh start requested: stopping stack and removing volumes ↓↓"
  docker compose down -v
}

Write-Host "↑↑ Starting stack ↑↑"
if ($Rebuild) {
  docker compose up -d --build
} else {
  docker compose up -d
}

Write-Host "`nServices:"
docker compose ps

# Wait for DB health
Wait-For-ContainerHealthy -name "windsor_db"

# Wait for API /health (api is published at host 8001)
Wait-For-Health -url "http://127.0.0.1:8001/health" -label "API"

# Quick DB sanity (counts)
try {
  Write-Host "`nDB sanity check (documents / chunks):"
  docker exec -i windsor_db psql -U $PGUSER -d $PGDB -c "SELECT count(*) AS documents FROM documents; SELECT count(*) AS chunks FROM document_chunks;" | Out-Host
} catch {
  Write-Warning "Could not run psql sanity check: $($_.Exception.Message)"
}

Write-Host "`nAPI -> http://127.0.0.1:8001/docs"
Write-Host "App -> http://127.0.0.1:8501`n"

if (-not $NoOpen) {
  try { Start-Process "http://127.0.0.1:8501" } catch {}
}

Write-Host "Tips:"
Write-Host "  docker compose ps                                # show status"
Write-Host "  docker logs -f windsor_api                       # tail API"
Write-Host "  docker logs -f windsor_indexer                   # tail indexer"
Write-Host "  docker exec -it windsor_api curl -s http://api:8000/health   # in-network API check"
Write-Host "  docker compose down -v                           # stop & wipe volumes (fresh DB)"
