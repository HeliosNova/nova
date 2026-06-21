# Phase-0 cron verification (Windows host).
#
# Companion to scripts/verify_phase_0.py - that script checks the in-container
# state. This one checks the Windows Scheduled Task that triggers the weekly
# fine-tune pipeline, which lives on the host rather than in the container.
#
# USAGE
#   pwsh -File scripts\verify_phase_0_cron.ps1
# or
#   powershell -ExecutionPolicy Bypass -File scripts\verify_phase_0_cron.ps1

$ErrorActionPreference = 'Stop'
$TaskName = 'Nova Weekly Fine-Tune'
$NovaDir = 'C:\Users\sysadmin\Desktop\Helios Project\nova_'
$ShScript = Join-Path $NovaDir 'scripts\finetune_weekly.sh'
$PyScript = Join-Path $NovaDir 'scripts\finetune.py'
$BashExe = 'C:\Program Files\Git\bin\bash.exe'

function Write-Result($name, $ok, $detail) {
    $tag = if ($ok) { 'PASS' } else { 'FAIL' }
    Write-Host ("  [{0}]  {1,-24}  {2}" -f $tag, $name, $detail)
    return $ok
}

Write-Host ('=' * 72)
Write-Host "Phase-0 Cron Verification - $(Get-Date -Format 'yyyy-MM-ddTHH:mm:ssZ')"
Write-Host ('=' * 72)

$allOk = $true

# 1. Scheduled task registered
$task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
$ok = $null -ne $task
$detail = if ($ok) { "registered, state=$($task.State)" } else { 'not registered - run setup_weekly_finetune.ps1' }
$allOk = (Write-Result 'task_registered' $ok $detail) -and $allOk

# 2. Task is enabled
if ($task) {
    $ok = $task.Settings.Enabled -eq $true -and $task.State -ne 'Disabled'
    $detail = "Enabled=$($task.Settings.Enabled) State=$($task.State)"
    $allOk = (Write-Result 'task_enabled' $ok $detail) -and $allOk
}

# 3. Trigger is weekly Sunday 23:00
if ($task) {
    $trg = $task.Triggers | Select-Object -First 1
    $ok = $false
    $detail = 'no trigger'
    if ($trg) {
        $days = $trg.DaysOfWeek
        $startTime = [datetime]$trg.StartBoundary
        $hhmm = $startTime.ToString('HH:mm')
        $ok = ($days -eq 'Sunday' -or $days -like '*Sunday*') -and ($hhmm -eq '23:00')
        $detail = "days=$days time=$hhmm"
    }
    $allOk = (Write-Result 'trigger_sunday_2300' $ok $detail) -and $allOk
}

# 4. Action points at the bash script that exists
if ($task) {
    $act = $task.Actions | Select-Object -First 1
    $ok = $false
    $detail = 'no action'
    if ($act) {
        $arg = $act.Arguments
        $pointsAtScript = $arg -like "*$ShScript*" -or $arg -like '*finetune_weekly.sh*'
        $execIsBash = $act.Execute -eq $BashExe -or $act.Execute -like '*bash.exe'
        $ok = $pointsAtScript -and $execIsBash
        $detail = "execute=$($act.Execute) arg=$arg"
    }
    $allOk = (Write-Result 'action_bash' $ok $detail) -and $allOk
}

# 5. finetune_weekly.sh exists
$ok = Test-Path $ShScript
$detail = if ($ok) { $ShScript } else { "missing: $ShScript" }
$allOk = (Write-Result 'shell_script' $ok $detail) -and $allOk

# 6. finetune.py exists
$ok = Test-Path $PyScript
$detail = if ($ok) { $PyScript } else { "missing: $PyScript" }
$allOk = (Write-Result 'python_script' $ok $detail) -and $allOk

# 7. git-bash exists
$ok = Test-Path $BashExe
$detail = if ($ok) { $BashExe } else { "missing: $BashExe - install Git for Windows" }
$allOk = (Write-Result 'bash_exe' $ok $detail) -and $allOk

# 8. Last run info (informational only, not a pass/fail gate)
if ($task) {
    $info = Get-ScheduledTaskInfo -TaskName $TaskName
    $detail = "LastRun=$($info.LastRunTime) LastResult=$($info.LastTaskResult) NextRun=$($info.NextRunTime)"
    [void](Write-Result 'last_run_info' $true $detail)
}

Write-Host ('=' * 72)
$overall = if ($allOk) { 'PASS' } else { 'FAIL' }
Write-Host "OVERALL: $overall"
if (-not $allOk) { exit 1 }
