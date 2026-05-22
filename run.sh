#!/usr/bin/env bash
# Локальный запуск платформы (macOS / Linux)
set -euo pipefail
cd "$(dirname "$0")"

PYTHON=""
for cmd in python3.12 python3.11 python3; do
  if command -v "$cmd" >/dev/null 2>&1; then
    ver=$("$cmd" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    major=${ver%%.*}
    minor=${ver#*.}
    if [ "$major" -eq 3 ] && [ "$minor" -ge 11 ] && [ "$minor" -le 12 ]; then
      PYTHON=$cmd
      break
    fi
  fi
done

if [ -z "$PYTHON" ]; then
  echo "Нужен Python 3.11 или 3.12. Установите с https://www.python.org/downloads/"
  exit 1
fi

if [ ! -d .venv ]; then
  echo "Создание виртуальной среды (.venv)..."
  "$PYTHON" -m venv .venv
fi

# shellcheck source=/dev/null
source .venv/bin/activate
python -m pip install --upgrade pip -q
pip install -r requirements.txt -q

echo ""
echo "Запуск: http://localhost:8501"
echo "Остановка: Ctrl+C"
echo ""
exec streamlit run streamlit_app.py
