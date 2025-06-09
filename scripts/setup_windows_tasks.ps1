# PowerShell script to set up Windows Task Scheduler tasks for GPT Trading System
# Run as Administrator

$projectPath = "D:\gpt_trader_v1"
$pythonPath = "$projectPath\venv\Scripts\python.exe"

Write-Host "Setting up Windows Task Scheduler tasks for GPT Trading System..." -ForegroundColor Green

# Function to create a scheduled task
function Create-TradingTask {
    param(
        [string]$TaskName,
        [string]$Description,
        [string]$ScriptPath,
        [string]$Arguments,
        [string]$TriggerType,
        [hashtable]$TriggerParams
    )
    
    # Create action
    $action = New-ScheduledTaskAction -Execute $pythonPath -Argument "$ScriptPath $Arguments" -WorkingDirectory $projectPath
    
    # Create trigger based on type
    switch ($TriggerType) {
        "Daily" {
            $trigger = New-ScheduledTaskTrigger -Daily -At $TriggerParams.Time
        }
        "Weekly" {
            $trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek $TriggerParams.DayOfWeek -At $TriggerParams.Time
        }
        "Monthly" {
            $trigger = New-ScheduledTaskTrigger -Daily -At $TriggerParams.Time
            # Note: We'll check in the script if it's the right day of month
        }
        "Hourly" {
            $trigger = New-ScheduledTaskTrigger -Once -At (Get-Date) -RepetitionInterval (New-TimeSpan -Hours $TriggerParams.Interval) -RepetitionDuration ([TimeSpan]::MaxValue)
        }
        "AtStartup" {
            $trigger = New-ScheduledTaskTrigger -AtStartup
        }
    }
    
    # Set principal (run whether user is logged on or not)
    $principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Password -RunLevel Highest
    
    # Set settings
    $settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable -RestartInterval (New-TimeSpan -Minutes 10) -RestartCount 3
    
    # Register the task
    $taskPath = "\GPT Trading System\"
    
    try {
        $existingTask = Get-ScheduledTask -TaskName $TaskName -TaskPath $taskPath -ErrorAction SilentlyContinue
        if ($existingTask) {
            Write-Host "Task '$TaskName' already exists. Updating..." -ForegroundColor Yellow
            Unregister-ScheduledTask -TaskName $TaskName -TaskPath $taskPath -Confirm:$false
        }
        
        Register-ScheduledTask -TaskName $TaskName -TaskPath $taskPath -Description $Description -Action $action -Trigger $trigger -Settings $settings -Principal $principal
        Write-Host "✓ Task '$TaskName' created successfully" -ForegroundColor Green
    }
    catch {
        Write-Host "✗ Failed to create task '$TaskName': $_" -ForegroundColor Red
    }
}

# 1. Main Trading System (runs 24/5 when market is open)
Create-TradingTask -TaskName "GPT Trading System - Main" `
    -Description "Main trading system that runs continuously during market hours" `
    -ScriptPath "$projectPath\trading_loop.py" `
    -Arguments "" `
    -TriggerType "AtStartup" `
    -TriggerParams @{}

# 2. ML Scheduler (runs continuously to monitor and update models)
Create-TradingTask -TaskName "GPT Trading - ML Scheduler" `
    -Description "ML model monitoring and automatic updates" `
    -ScriptPath "$projectPath\scripts\ml_scheduler.py" `
    -Arguments "daemon" `
    -TriggerType "AtStartup" `
    -TriggerParams @{}

# 3. Database Backup (runs daily at 2 AM)
Create-TradingTask -TaskName "GPT Trading - Database Backup" `
    -Description "Daily database backup" `
    -ScriptPath "$projectPath\scripts\automation\database_backup.py" `
    -Arguments "--type daily" `
    -TriggerType "Daily" `
    -TriggerParams @{Time = "02:00"}

# 4. Performance Analytics (runs daily at 11 PM)
Create-TradingTask -TaskName "GPT Trading - Performance Report" `
    -Description "Daily performance analysis and reporting" `
    -ScriptPath "$projectPath\scripts\performance_analytics.py" `
    -Arguments "" `
    -TriggerType "Daily" `
    -TriggerParams @{Time = "23:00"}

# 5. Health Check (optional - runs every hour)
Create-TradingTask -TaskName "GPT Trading - Health Check" `
    -Description "Hourly system health check and monitoring" `
    -ScriptPath "$projectPath\scripts\automation\health_check.py" `
    -Arguments "" `
    -TriggerType "Hourly" `
    -TriggerParams @{Interval = 1}

Write-Host "`nAll tasks created successfully!" -ForegroundColor Green
Write-Host "You can view and manage these tasks in Task Scheduler under 'GPT Trading System' folder" -ForegroundColor Yellow

# Create a batch file to start essential services
$batchContent = @"
@echo off
echo Starting GPT Trading System Services...

REM Activate virtual environment (if exists)
if exist "$projectPath\venv\Scripts\activate.bat" (
    call "$projectPath\venv\Scripts\activate.bat"
)

REM Start main trading system
start "GPT Trading System" /min $pythonPath "$projectPath\trading_loop.py"

REM Wait a bit before starting ML scheduler
timeout /t 10

REM Start ML scheduler
start "ML Scheduler" /min $pythonPath "$projectPath\scripts\ml_scheduler.py" daemon

echo All essential services started!
echo.
echo Optional: Run these in separate terminals as needed:
echo - ML Dashboard: streamlit run scripts\ml_improvement_dashboard.py
echo - Performance Check: python scripts\performance_analytics.py
echo.
pause
"@

$batchPath = "$projectPath\scripts\start_essential_services.bat"
Set-Content -Path $batchPath -Value $batchContent
Write-Host "`nCreated batch file for manual start: $batchPath" -ForegroundColor Cyan

# Create a simple monitoring batch file
$monitorContent = @"
@echo off
echo Starting ML Improvement Dashboard...

REM Activate virtual environment (if exists)
if exist "$projectPath\venv\Scripts\activate.bat" (
    call "$projectPath\venv\Scripts\activate.bat"
)

REM Check if streamlit is installed
$pythonPath -m pip show streamlit >nul 2>&1
if errorlevel 1 (
    echo Installing streamlit...
    $pythonPath -m pip install streamlit
)

REM Start dashboard
streamlit run "$projectPath\scripts\ml_improvement_dashboard.py"
"@

$monitorPath = "$projectPath\scripts\start_ml_dashboard.bat"
Set-Content -Path $monitorPath -Value $monitorContent
Write-Host "Created ML dashboard launcher: $monitorPath" -ForegroundColor Cyan