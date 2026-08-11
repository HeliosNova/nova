# Offsite encrypted backup — restic → Backblaze B2 (prepared 2026-07-08).
#
# STATUS: READY BUT NOT ARMED — needs owner-provided B2 credentials once.
# The local tiers (daily verified VACUUM INTO snapshots in .\backups\) run
# automatically; this script adds the offsite tier that survives disk loss
# and the Docker-vhdx wipe class (docker/for-win#14461, June 2026 PC reset).
#
# One-time setup (owner):
#   1. winget install restic.restic
#   2. Create a B2 bucket (e.g. nova-backup) + app key at backblaze.com
#   3. Set machine-scoped env vars (PowerShell as admin):
#        [Environment]::SetEnvironmentVariable('RESTIC_REPOSITORY','b2:nova-backup:/', 'Machine')
#        [Environment]::SetEnvironmentVariable('B2_ACCOUNT_ID','<keyID>', 'Machine')
#        [Environment]::SetEnvironmentVariable('B2_ACCOUNT_KEY','<appKey>', 'Machine')
#        [Environment]::SetEnvironmentVariable('RESTIC_PASSWORD','<NEW strong passphrase — store in password manager + print>', 'Machine')
#   4. restic init
#   5. Register the nightly task (from this directory):
#        schtasks /Create /TN "Nova Offsite Backup" /TR "powershell -NoProfile -ExecutionPolicy Bypass -File '$PSScriptRoot\backup_offsite.ps1'" /SC DAILY /ST 04:30
#   6. Run once manually and then do a restore drill:
#        restic restore latest --target C:\temp\restore-drill --include /backups
#
# What gets backed up (client-side encrypted; B2 sees only ciphertext):
#   - .\backups\           daily verified SQLite snapshots + models manifest + config overrides
#   - eval baselines       (part of /data exports if copied into backups\)
# What does NOT: model weights (30GB, re-pullable via models_manifest.txt).

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot   # nova_ checkout root
$backupDir = Join-Path $repoRoot "backups"

if (-not $env:RESTIC_REPOSITORY) {
    Write-Error "RESTIC_REPOSITORY not set — complete the one-time setup in this script's header."
    exit 1
}
if (-not (Test-Path $backupDir)) {
    Write-Error "Backup dir not found: $backupDir"
    exit 1
}

restic backup $backupDir --tag nova-nightly --use-fs-snapshot
if ($LASTEXITCODE -ne 0) { Write-Error "restic backup failed"; exit 1 }

# Retention: 14 daily, 8 weekly, 12 monthly
restic forget --tag nova-nightly --keep-daily 14 --keep-weekly 8 --keep-monthly 12 --prune
restic check --read-data-subset=5%
Write-Output "Offsite backup complete: $(Get-Date -Format s)"
