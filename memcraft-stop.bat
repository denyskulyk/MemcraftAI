@echo off
rem Stops the MemcraftAI worker, its ComfyUI, the web app and Postgres.
pwsh -NoProfile -ExecutionPolicy Bypass -File "%~dp0deploy\stop-all.ps1"
pause
