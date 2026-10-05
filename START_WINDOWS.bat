@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe py -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
start "" http://127.0.0.1:8010
.venv\Scripts\python.exe run.py
pause
