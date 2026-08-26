# Second-drive offsite copy — F:\...\backups  ->  E:\nova-offsite  (armed 2026-07-09).
#
# WHY: the daily verified VACUUM INTO snapshots (backup.py, run by the System
# Maintenance monitor) land in .\backups\ on F: — the SAME physical disk as the
# live DB. A disk failure loses both. This mirrors them to a SEPARATE physical
# drive (E:), so an F: disk failure is survivable. Fully local/sovereign — the
# data never leaves the machine (owner choice 2026-07-09; declined cloud/OneDrive
# to preserve local-only sovereignty). NOTE: this does NOT survive a full machine
# / docker-vhdx wipe (the class that hit this PC 2026-06-14). For that, arm the
# encrypted restic tier (scripts/backup_offsite.ps1) when ready.
#
# Register the nightly task (run once, as the owner, from any shell):
#   schtasks /Create /TN "Nova Local Offsite" /SC DAILY /ST 05:00 /RL LIMITED /F `
#     /TR "powershell -NoProfile -ExecutionPolicy Bypass -File 'F:\Helios Project\nova_\scripts\backup_local_offsite.ps1'"
# 05:00 is after the container's daily maintenance backup (~02:51 observed).

$ErrorActionPreference = "Stop"
$src  = "F:\Helios Project\nova_\backups"
$dst  = "E:\nova-offsite"
$keep = 14   # retain the newest N .db snapshots on the offsite drive

if (-not (Test-Path $src)) { Write-Error "Source backup dir missing: $src"; exit 1 }
New-Item -ItemType Directory -Force $dst | Out-Null

# Mirror the DB snapshots + the small manifests (models list, config overrides).
# Copy .db by whole-file (they're immutable per-day snapshots). Manifests are
# compared by LastWriteTime + Length (2026-08-25: a same-size content change —
# e.g. a flipped boolean in config_overrides.json — never refreshed under the
# old size-only compare; E:'s copy sat stale from 8/16).
$copied = 0
Get-ChildItem $src -File | Where-Object { $_.Extension -in '.db','.txt','.json' } | ForEach-Object {
    $target = Join-Path $dst $_.Name
    $t = if (Test-Path $target) { Get-Item $target } else { $null }
    if (-not $t -or $t.Length -ne $_.Length -or $t.LastWriteTimeUtc -lt $_.LastWriteTimeUtc) {
        Copy-Item $_.FullName $target -Force
        $copied++
    }
}

# Secrets/config that exist ONLY on F: (2026-08-25 audit: .env — every token
# and API key — had a single copy on the same disk as the live system; F: loss
# = unrecoverable credentials). Same machine, same trust domain as the source;
# the encrypted off-machine tier remains restic (backup_offsite.ps1).
$repo = "F:\Helios Project\nova_"
foreach ($rel in @(".env", "searxng\settings.yml")) {
    $srcFile = Join-Path $repo $rel
    if (Test-Path $srcFile) {
        $flat = "repo-" + ($rel -replace '[\\/]', '_')
        $target = Join-Path $dst $flat
        $s = Get-Item $srcFile
        $t = if (Test-Path $target) { Get-Item $target } else { $null }
        if (-not $t -or $t.Length -ne $s.Length -or $t.LastWriteTimeUtc -lt $s.LastWriteTimeUtc) {
            Copy-Item $srcFile $target -Force
            $copied++
        }
    }
}

# Prune: keep only the newest $keep dated snapshots on the offsite drive.
$snaps = Get-ChildItem $dst -Filter 'nova-*.db' | Sort-Object LastWriteTime -Descending
if ($snaps.Count -gt $keep) {
    $snaps | Select-Object -Skip $keep | Remove-Item -Force
}

$latest = Get-ChildItem $dst -Filter 'nova-*.db' | Sort-Object LastWriteTime -Descending | Select-Object -First 1
Write-Output ("Local offsite sync {0}: copied {1} file(s); {2} snapshot(s) on {3}; latest={4}" -f `
    (Get-Date -Format s), $copied, (Get-ChildItem $dst -Filter 'nova-*.db').Count, $dst, $latest.Name)
