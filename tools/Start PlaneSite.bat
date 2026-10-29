@echo off
title PlaneSite
cd /d "%~dp0"
"%~dp0python\python.exe" "%~dp0app\main.py" %*
if errorlevel 1 pause
