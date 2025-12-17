@echo off
REM Send Optimization Results to Quest - Windows Batch Script
REM This script runs the send_results_to_quest.py script

echo ========================================
echo ProXeek - Send Results to Quest
echo ========================================
echo.

REM Change to the script directory
cd /d "%~dp0"

REM Run the Python script
python send_results_to_quest.py

REM Pause to show results
echo.
pause

