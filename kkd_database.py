"""
Загрузка лабораторной таблицы ККД в SQLite (data/kkd_lab.sqlite).
Исходный файл по умолчанию: data/ККД_БД.xlsx
"""

from __future__ import annotations

import os
from pathlib import Path
import sqlite3

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_EXCEL = DATA_DIR / "ККД_БД.xlsx"
DEFAULT_SQLITE = DATA_DIR / "kkd_lab.sqlite"
TABLE_NAME = "kkd"


def sqlite_path() -> Path:
    return DEFAULT_SQLITE


def excel_path() -> Path:
    return DEFAULT_EXCEL


def build_kkd_sqlite(
    excel_file: str | os.PathLike | None = None,
    sqlite_file: str | os.PathLike | None = None,
    sheet_name: int | str = 0,
) -> Path:
    """
    Читает Excel и перезаписывает таблицу kkd в SQLite.
    Возвращает путь к файлу БД.
    """
    xls = Path(excel_file) if excel_file else DEFAULT_EXCEL
    db = Path(sqlite_file) if sqlite_file else DEFAULT_SQLITE
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not xls.is_file():
        raise FileNotFoundError(f"Не найден файл Excel: {xls}")
    df = pd.read_excel(xls, sheet_name=sheet_name)
    df.columns = (
        df.columns.astype(str).str.strip().str.replace("\n", "", regex=False).str.replace('"', "", regex=False)
    )
    conn = sqlite3.connect(db)
    df.to_sql(TABLE_NAME, conn, if_exists="replace", index=False)
    conn.close()
    return db


def load_kkd_dataframe(sqlite_file: str | os.PathLike | None = None) -> pd.DataFrame:
    db = Path(sqlite_file) if sqlite_file else DEFAULT_SQLITE
    if not db.is_file():
        raise FileNotFoundError(f"База не найдена: {db}. Сначала выполните сборку из Excel.")
    conn = sqlite3.connect(db)
    try:
        return pd.read_sql(f"SELECT * FROM {TABLE_NAME}", conn)
    finally:
        conn.close()


if __name__ == "__main__":
    out = build_kkd_sqlite()
    print(out)
