@echo off

python -c "import importlib.util, sys; spec = importlib.util.find_spec('requests'); print('Installing requests package...') if spec is None else None; sys.exit(0 if spec else 1)"

if %ERRORLEVEL% NEQ 0 (
    python -m ensurepip --upgrade
    python -m pip install requests
)

python scheduling.py
pause