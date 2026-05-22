@echo off
chcp 65001 >nul
cd /d "%~dp0"

set "PYTHON="
where py >nul 2>&1 && (
  py -3.12 -c "import sys" >nul 2>&1 && set "PYTHON=py -3.12"
  if not defined PYTHON py -3.11 -c "import sys" >nul 2>&1 && set "PYTHON=py -3.11"
)
if not defined PYTHON where python >nul 2>&1 && set "PYTHON=python"

if not defined PYTHON (
  echo Нужен Python 3.11 или 3.12. Установите с https://www.python.org/downloads/
  echo При установке отметьте "Add python.exe to PATH".
  pause
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo Создание виртуальной среды (.venv)...
  %PYTHON% -m venv .venv
  if errorlevel 1 (
    echo Не удалось создать .venv
    pause
    exit /b 1
  )
)

call .venv\Scripts\activate.bat
python -m pip install --upgrade pip -q
pip install -r requirements.txt -q
if errorlevel 1 (
  echo Ошибка установки зависимостей. Проверьте интернет и requirements.txt
  pause
  exit /b 1
)

echo.
echo Запуск: http://localhost:8501
echo Остановка: Ctrl+C
echo.
streamlit run streamlit_app.py
pause
