$TaskName = 'Nova Weekly Fine-Tune'
$NovaDir = 'C:\Users\sysadmin\Desktop\Helios Project\nova_'
$BashExe = 'C:\Program Files\Git\bin\bash.exe'
$ScriptPath = $NovaDir + '\scripts\finetune_weekly.sh'

Write-Host 'Setting up Nova Weekly Fine-Tune...'

Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue

$Action = New-ScheduledTaskAction -Execute $BashExe -Argument ('--login -c ' + "'" + $ScriptPath + "'") -WorkingDirectory $NovaDir

$Trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Sunday -At '11:00PM'

$Settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Hours 6) -RestartCount 1 -RestartInterval (New-TimeSpan -Minutes 30)

$Principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType S4U -RunLevel Highest

Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger -Settings $Settings -Principal $Principal -Description 'Weekly DPO fine-tuning for Nova AI - 11 PM Sunday Pacific'

Write-Host 'Scheduled task created successfully.'
Write-Host 'Schedule: Every Sunday at 11:00 PM'
