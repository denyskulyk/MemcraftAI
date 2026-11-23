@echo off
rem Starts MemcraftAI by hand: Podman, Postgres, the web app and the worker.
pwsh -NoProfile -ExecutionPolicy Bypass -File "%~dp0deploy\start-all.ps1"
pause
