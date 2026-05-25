from __future__ import annotations

import hashlib
import io
import os
import platform
import subprocess
import sys
from pathlib import Path
import time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from brooks_corey import (
    PowerBounds,
    auto_exp_bounds_swl_poro,
    auto_power_bounds,
    compute_soil_from_params,
    envelope_max_violation,
    evaluate_brooks_score,
    optimize_brooks_corey_for_region,
    prepare_brooks_training_data,
)
from kkd_database import build_kkd_sqlite, excel_path, load_kkd_dataframe, sqlite_path
from lab_analysis import (
    auto_ab_bounds_from_cloud,
    classify_j_matrix_stairs_columns,
    DEFAULT_MATRIX_J_KEYWORDS,
    DEFAULT_MATRIX_WATER_KEYWORDS,
    filter_lab_df,
    fit_power_j_swn,
    guess_area_column,
    guess_horizon_column,
    guess_j_column,
    is_likely_j_matrix_stairs_format,
    parse_matrix_block_keywords,
    pick_second_swn_column,
    transform_j_stairs_wide_to_long,
)
from optimize import (
    DEFAULT_J_CAP_AT_LOW_SWN,
    DEFAULT_LOW_SWN_THRESHOLD,
    _timing_table_with_total,
    apply_low_swn_j_cap,
    j_power_from_swn,
    run_pipeline,
)

st.set_page_config(page_title="J-функция Леверетта", layout="wide")


def _inject_streamlit_ru_ui_styles() -> None:
    """Русские подписи встроенных кнопок Streamlit (Browse files, Drag and drop, …)."""
    st.markdown(
        """
        <style>
        html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
            overflow-anchor: none;
        }
        /* Загрузка файла: Browse files → Выбрать файл */
        div[data-testid="stFileUploader"] button {
            font-size: 0 !important;
            line-height: 0;
        }
        div[data-testid="stFileUploader"] button::after {
            content: "Выбрать файл";
            font-size: 0.875rem;
            line-height: normal;
        }
        div[data-testid="stFileUploader"] button span,
        div[data-testid="stFileUploader"] button div {
            display: none;
        }
        /* Drag and drop file here */
        div[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] div small {
            font-size: 0 !important;
            line-height: 0;
        }
        div[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] div small::before {
            font-size: 0.8rem;
            line-height: 1.4;
            color: rgba(49, 51, 63, 0.6);
        }
        div[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] div small:nth-of-type(1)::before {
            content: "Перетащите файл сюда";
        }
        div[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] div small:nth-of-type(2)::before {
            content: "или нажмите «Выбрать файл»";
        }
        /* Скачивание: подпись на кнопке уже задаётся в коде; скрываем служебный англ. хвост, если есть */
        [data-testid="stDownloadButton"] button {
            font-size: inherit;
        }
        /* Multiselect: Clear all → Очистить */
        div[data-baseweb="select"] button[aria-label="Clear all"],
        div[data-baseweb="select"] button[title="Clear all"] {
            font-size: 0 !important;
            line-height: 0;
        }
        div[data-baseweb="select"] button[aria-label="Clear all"] span,
        div[data-baseweb="select"] button[title="Clear all"] span,
        div[data-baseweb="select"] button[aria-label="Clear all"] svg,
        div[data-baseweb="select"] button[title="Clear all"] svg {
            display: none !important;
        }
        div[data-baseweb="select"] button[aria-label="Clear all"]::after,
        div[data-baseweb="select"] button[title="Clear all"]::after {
            content: "Очистить";
            font-size: 0.75rem;
            line-height: normal;
        }
        /* Таблицы: иконки панели dataframe */
        [data-testid="stDataFrame"] button[title] svg,
        [data-testid="stDataFrame"] button[aria-label] svg {
            display: none !important;
        }
        [data-testid="stDataFrame"] button[title],
        [data-testid="stDataFrame"] button[aria-label] {
            font-size: 0 !important;
            min-width: 2rem;
        }
        [data-testid="stDataFrame"] button[title="Download as CSV"]::after,
        [data-testid="stDataFrame"] button[aria-label="Download as CSV"]::after {
            content: "CSV";
            font-size: 0.7rem;
            line-height: normal;
        }
        [data-testid="stDataFrame"] button[title="Search"]::after,
        [data-testid="stDataFrame"] button[aria-label="Search"]::after {
            content: "Поиск";
            font-size: 0.7rem;
            line-height: normal;
        }
        [data-testid="stDataFrame"] button[title="Fullscreen"]::after,
        [data-testid="stDataFrame"] button[aria-label="Fullscreen"]::after {
            content: "Экран";
            font-size: 0.7rem;
            line-height: normal;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


_inject_streamlit_ru_ui_styles()

DATA_DIR = Path(__file__).resolve().parent / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
SNAPSHOTS_DIR = DATA_DIR / "snapshots"
PROJECT_ROOT = Path(__file__).resolve().parent

SNAPSHOT_REQUIRED_COLUMNS = (
    "WELL_NAME",
    "_AXIS",
    "Кнг_hist",
    "Кнг_model",
    "METHOD",
    "SNAPSHOT_ID",
    "SNAPSHOT_LABEL",
)


def _decode_upload_text(file_bytes: bytes) -> str:
    for enc in ("utf-8-sig", "utf-8", "cp1251", "latin-1"):
        try:
            return file_bytes.decode(enc)
        except UnicodeDecodeError:
            continue
    return file_bytes.decode("utf-8", errors="replace")


def _detect_csv_separator(text: str) -> str:
    """Запятая или точка с запятой по первой непустой строке (типичный экспорт Excel RU)."""
    header = ""
    for line in text.splitlines():
        if line.strip():
            header = line.strip()
            break
    if not header:
        return ","
    n_semi = header.count(";")
    n_comma = header.count(",")
    if n_semi > n_comma:
        return ";"
    if n_semi == n_comma and n_semi > 0:
        return ";" if len(header.split(";")) >= len(header.split(",")) else ","
    return ","


def _read_csv_text(text: str) -> pd.DataFrame:
    sep = _detect_csv_separator(text)
    kwargs: dict = {"sep": sep, "low_memory": False}
    if sep == ";":
        kwargs["decimal"] = ","
    return pd.read_csv(io.StringIO(text), **kwargs)


def _read_csv_path(path: Path) -> pd.DataFrame:
    return _read_csv_text(_decode_upload_text(path.read_bytes()))


@st.cache_data(show_spinner=False)
def _read_table_from_bytes(file_bytes: bytes, filename: str) -> pd.DataFrame:
    name = filename.lower()
    if name.endswith(".csv"):
        return _read_csv_text(_decode_upload_text(file_bytes))
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(io.BytesIO(file_bytes))
    if name.endswith(".txt"):
        content = file_bytes.decode("utf-8", errors="ignore")
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        if not lines:
            raise ValueError("Пустой txt файл.")
        columns = lines[0][1:].replace('"', "").split()
        data = [line.split() for line in lines[1:]]
        df = pd.DataFrame(data=data, columns=columns)
        if not df.empty:
            first_col = df.columns[0]
            df[first_col] = df[first_col].astype(str).str.replace('"', "", regex=False)
        return df
    raise ValueError("Поддерживаются только файлы .csv, .xlsx/.xls и .txt")


def _read_table(uploaded_file) -> pd.DataFrame:
    return _read_table_from_bytes(uploaded_file.getvalue(), uploaded_file.name)


def _persist_uploaded_file(uploaded_file, prefix: str) -> str:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    target = UPLOAD_DIR / f"{prefix}_{uploaded_file.name}"
    if (not target.exists()) or target.stat().st_size != uploaded_file.size:
        target.write_bytes(uploaded_file.getbuffer())
    return str(target)


@st.cache_data(show_spinner=False)
def _read_table_from_path(path: str) -> pd.DataFrame:
    p = Path(path)
    name = p.name.lower()
    if name.endswith(".csv"):
        return _read_csv_path(p)
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(p)
    if name.endswith(".txt"):
        with p.open("r", encoding="utf-8", errors="ignore") as fh:
            first_line = fh.readline()
        columns = first_line[1:].replace('"', "").split()
        return pd.read_csv(p, sep=r"\s+", skiprows=1, names=columns, engine="c", low_memory=False)
    raise ValueError("Поддерживаются только файлы .csv, .xlsx/.xls и .txt")


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.astype(str).str.strip().str.replace('"', "", regex=False).str.replace("\ufeff", "", regex=False)
    )
    return df


# Не участвует в фильтрации и не обязателен в выгрузке скважин.
_OPTIONAL_WELL_COLS = frozenset({"FWL_GDM", "Perf_GDM"})
DEFAULT_FWL_MIN_CONTINUOUS = 3.0


def _fwl_valid_series(df: pd.DataFrame, col: str = "FWL_GDM") -> pd.Series:
    s = pd.to_numeric(df[col], errors="coerce")
    return s[s.notna() & (s > -900)]


def _apply_fwl_filter(df: pd.DataFrame, settings: dict[str, Any]) -> pd.DataFrame:
    if not settings.get("has_fwl") or "FWL_GDM" not in df.columns:
        return df
    fwl = pd.to_numeric(df["FWL_GDM"], errors="coerce")
    mode = settings.get("mode")
    if mode == "continuous":
        mn = float(settings.get("min_continuous", DEFAULT_FWL_MIN_CONTINUOUS))
        return df.loc[(fwl != 0) & (fwl >= mn)]
    if mode == "discrete":
        excluded = settings.get("exclude_discrete") or ()
        if not excluded:
            return df
        drop = np.zeros(len(df), dtype=bool)
        for v in excluded:
            drop |= np.isclose(fwl.to_numpy(dtype=float), float(v), rtol=0, atol=1e-5)
        return df.loc[~drop]
    return df


def _fwl_filter_settings_ui(mapped: pd.DataFrame, *, key_prefix: str = "shared") -> dict[str, Any]:
    """
    Настройки фильтра FWL_GDM: непрерывные (FWL≠0 и FWL≥порог) или дискретные (исключить выбранные коды).
    """
    out: dict[str, Any] = {
        "has_fwl": False,
        "mode": None,
        "min_continuous": DEFAULT_FWL_MIN_CONTINUOUS,
        "exclude_discrete": (),
    }
    if "FWL_GDM" not in mapped.columns:
        return out

    out["has_fwl"] = True
    valid = _fwl_valid_series(mapped)
    st.subheader("Фильтр FWL_GDM")
    mode = st.radio(
        "Как представлены данные FWL_GDM в файле",
        options=["continuous", "discrete"],
        format_func=lambda x: (
            "Непрерывные (высота до ВНК)" if x == "continuous" else "Дискретные (коды / категории)"
        ),
        index=0,
        key=f"{key_prefix}_fwl_mode",
        horizontal=True,
    )
    out["mode"] = mode

    if mode == "continuous":
        min_val = st.number_input(
            "Оставить строки с FWL ≥",
            min_value=0.0,
            value=DEFAULT_FWL_MIN_CONTINUOUS,
            step=0.1,
            key=f"{key_prefix}_fwl_min",
            help="Строки с FWL = 0 всегда отбрасываются.",
        )
        out["min_continuous"] = float(min_val)
        n_ok = int(((valid != 0) & (valid >= min_val)).sum()) if len(valid) else 0
        st.caption(
            f"Фильтр: FWL ≠ 0 и FWL ≥ {min_val:g}. "
            f"Останется строк: {n_ok} из {len(mapped)}."
        )
    else:
        if valid.empty:
            st.warning("В колонке FWL_GDM нет валидных значений для выбора кодов.")
        else:
            uniq = sorted(valid.unique().tolist(), key=float)
            excluded = st.multiselect(
                "Не использовать строки при значениях FWL",
                options=uniq,
                format_func=lambda x: f"{float(x):g}",
                key=f"{key_prefix}_fwl_exclude",
                help="Отмеченные значения будут исключены из расчёта.",
            )
            out["exclude_discrete"] = tuple(float(x) for x in excluded)
            fwl_all = pd.to_numeric(mapped["FWL_GDM"], errors="coerce")
            drop_mask = np.zeros(len(mapped), dtype=bool)
            for v in excluded:
                drop_mask |= np.isclose(fwl_all.to_numpy(dtype=float), float(v), rtol=0, atol=1e-5)
            n_drop = int(drop_mask.sum())
            st.caption(
                f"Исключается значений FWL: {len(excluded)}. "
                f"Строк к отсечению (оценка): {n_drop} из {len(mapped)}."
            )
    return out


def _fwl_settings_from_session(key_prefix: str = "shared") -> dict[str, Any]:
    """Параметры FWL из session_state (без повторного вывода виджетов)."""
    mode = st.session_state.get(f"{key_prefix}_fwl_mode", "continuous")
    excluded_raw = st.session_state.get(f"{key_prefix}_fwl_exclude", [])
    excluded = tuple(float(x) for x in excluded_raw) if excluded_raw else ()
    return {
        "has_fwl": True,
        "mode": mode,
        "min_continuous": float(st.session_state.get(f"{key_prefix}_fwl_min", DEFAULT_FWL_MIN_CONTINUOUS)),
        "exclude_discrete": excluded,
    }


def _clean_wells_df(
    df: pd.DataFrame,
    *,
    fwl_mode: str | None = None,
    fwl_min: float = DEFAULT_FWL_MIN_CONTINUOUS,
    fwl_exclude: tuple[float, ...] = (),
) -> pd.DataFrame:
    df = _normalize_columns(df.copy())
    if df.empty:
        return df
    numeric_candidates = [col for col in df.columns if col != df.columns[0]]
    for col in numeric_candidates:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    filter_cols = [c for c in numeric_candidates if c not in _OPTIONAL_WELL_COLS]
    if filter_cols:
        mask_bad = (df[filter_cols] <= -1).any(axis=1)
        df = df.loc[~mask_bad].dropna(subset=filter_cols)
    if "ACTNUM_GDM" in df.columns:
        df = df.loc[df["ACTNUM_GDM"] != 0]
    if fwl_mode in ("continuous", "discrete"):
        df = _apply_fwl_filter(
            df,
            {
                "has_fwl": True,
                "mode": fwl_mode,
                "min_continuous": fwl_min,
                "exclude_discrete": fwl_exclude,
            },
        )
    if "PC" in df.columns:
        df = df.loc[df["PC"] >= 0.01]
    if "Кнг_W" in df.columns:
        df.loc[df["Кнг_W"] > 1, "Кнг_W"] = df["Кнг_W"] / 100
    return df.reset_index(drop=True)


def _clean_prod_df(df: pd.DataFrame | None) -> pd.DataFrame | None:
    if df is None:
        return None
    df = _normalize_columns(df.copy())
    df = df.rename(columns=lambda x: str(x).replace("\n", ""))
    return df


@st.cache_data(show_spinner=False)
def _clean_wells_cached(
    df: pd.DataFrame,
    fwl_mode: str | None,
    fwl_min: float,
    fwl_exclude: tuple[float, ...],
) -> pd.DataFrame:
    return _clean_wells_df(df, fwl_mode=fwl_mode, fwl_min=fwl_min, fwl_exclude=fwl_exclude)


@st.cache_data(show_spinner=False)
def _clean_prod_cached(df: pd.DataFrame | None) -> pd.DataFrame | None:
    return _clean_prod_df(df)


def _pick_depth_column(df: pd.DataFrame) -> str | None:
    candidates = ["DEPT", "DEPTH", "MD", "TVD", "TVDSS", "Z_GDM", "GL_GDM", "H_GDM", "ГЛУБИНА", "Глубина"]
    upper_map = {c.upper(): c for c in df.columns}
    for cand in candidates:
        if cand.upper() in upper_map:
            return upper_map[cand.upper()]
    return None


def _series_1d(df: pd.DataFrame, col: str, *fallbacks: str) -> pd.Series:
    """Одномерный столбец как Series (если после rename дубли — берётся первый)."""
    for name in (col, *fallbacks):
        if name not in df.columns:
            continue
        val = df[name]
        if isinstance(val, pd.DataFrame):
            val = val.iloc[:, 0]
        return pd.to_numeric(val, errors="coerce")
    return pd.Series(np.nan, index=df.index)


def _well_convergence_percent_weighted(df: pd.DataFrame) -> float:
    """Средневзвешенный процент сходимости: веса из колонки weight (если есть)."""
    eps = 1e-6
    true_vals = _series_1d(df, "Кнг_W", "Кнг_hist")
    pred_vals = _series_1d(df, "Kng_model", "Кнг_model")
    w = _series_1d(df, "weight").fillna(1.0).to_numpy(dtype=float)
    valid = (true_vals.notna() & pred_vals.notna()).to_numpy()
    if valid.sum() == 0:
        return float("nan")
    t = true_vals.to_numpy(dtype=float)[valid]
    p = pred_vals.to_numpy(dtype=float)[valid]
    ww = w[valid]
    rel_err = np.abs(p - t) / np.maximum(np.abs(t), eps)
    point_score = np.clip(100.0 * (1.0 - rel_err), 0.0, 100.0)
    sw = float(np.sum(ww))
    if sw <= 0:
        return float("nan")
    return float(np.sum(ww * point_score) / sw)


def _filter_convergence_points(df: pd.DataFrame) -> pd.DataFrame:
    """
    Для кроссплотов: исключаем нули и выбросы Кнг_W по каждой скважине.
    """
    out = df.copy()
    out["Кнг_W"] = pd.to_numeric(out["Кнг_W"], errors="coerce")
    out["Kng_model"] = pd.to_numeric(out["Kng_model"], errors="coerce")
    out = out.dropna(subset=["Кнг_W", "Kng_model"])
    out = out[out["Кнг_W"] != 0]
    if out.empty or "WELL_NAME" not in out.columns:
        return out

    keep_idx: list[int] = []
    for _, g in out.groupby("WELL_NAME"):
        x = g["Кнг_W"].to_numpy(dtype=float)
        if len(x) < 6:
            keep_idx.extend(g.index.tolist())
            continue
        q1, q3 = np.quantile(x, [0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        keep = g[(g["Кнг_W"] >= lo) & (g["Кнг_W"] <= hi)]
        if len(keep) < max(5, int(0.5 * len(g))):
            ql, qh = np.quantile(x, [0.02, 0.98])
            keep = g[(g["Кнг_W"] >= ql) & (g["Кнг_W"] <= qh)]
        keep_idx.extend(keep.index.tolist())
    return out.loc[sorted(set(keep_idx))].copy()


def _exclude_clipped_kng_zeros(df: pd.DataFrame) -> pd.DataFrame:
    """Исключить точки, где Kng_model=0 при ненулевом Кнг_W (обрезка модели)."""
    out = df.copy()
    y_true = _series_1d(out, "Кнг_W", "Кнг_hist")
    y_pred = _series_1d(out, "Kng_model", "Кнг_model")
    clip = (y_pred == 0) & y_true.notna() & (y_true != 0)
    return out.loc[~clip].copy()


def _well_weighted_crossplot_df(df: pd.DataFrame) -> pd.DataFrame:
    if "WELL_NAME" not in df.columns:
        return pd.DataFrame()
    rows = []
    for well, g in df.groupby("WELL_NAME"):
        y_true = _series_1d(g, "Кнг_W", "Кнг_hist")
        y_pred = _series_1d(g, "Kng_model", "Кнг_model")
        w = _series_1d(g, "weight").fillna(1.0) if "weight" in g.columns else pd.Series(1.0, index=g.index)
        valid = y_true.notna() & y_pred.notna()
        if valid.sum() == 0:
            continue
        t = y_true[valid].to_numpy(dtype=float)
        p = y_pred[valid].to_numpy(dtype=float)
        ww = w[valid].to_numpy(dtype=float)
        ws = float(np.sum(ww))
        if ws <= 0:
            continue
        rows.append(
            {
                "WELL_NAME": str(well),
                "Кнг_W_wmean": float(np.sum(ww * t) / ws),
                "Kng_model_wmean": float(np.sum(ww * p) / ws),
                "convergence_percent": _well_convergence_percent_weighted(g),
                "points": int(valid.sum()),
                "avg_weight": float(np.mean(ww)),
            }
        )
    return pd.DataFrame(rows)


def _crossplot_hover_name_kw(df: pd.DataFrame) -> dict[str, str]:
    if "WELL_NAME" in df.columns:
        return {"hover_name": "WELL_NAME"}
    return {}


def _crossplot_hovertemplate(metrics_lines: str, *, with_well: bool) -> str:
    if with_well:
        return f"Скважина=%{{hovertext}}<br>{metrics_lines}<extra></extra>"
    return f"{metrics_lines}<extra></extra>"


def _apply_crossplot_hover(fig, metrics_lines: str, df: pd.DataFrame) -> None:
    fig.update_traces(
        hovertemplate=_crossplot_hovertemplate(metrics_lines, with_well="WELL_NAME" in df.columns)
    )


def _j_kng_interactive_hover(df: pd.DataFrame, depth_col: str | None) -> tuple[list[str], str]:
    """Колонки hover_data и шаблон подсказки для scatter Кнг (FWL, глубина, прочие поля)."""
    hover_cols: list[str] = []
    label_by_col: dict[str, str] = {}
    if "FWL_GDM" in df.columns:
        hover_cols.append("FWL_GDM")
        label_by_col["FWL_GDM"] = "FWL"
    if depth_col and depth_col in df.columns:
        hover_cols.append(depth_col)
        label_by_col[depth_col] = "Глубина"
    for c in ("PC", "PORO_GDM", "PERM_GDM", "SWL_GDM", "weight", "thickness"):
        if c in df.columns and c not in hover_cols:
            hover_cols.append(c)
            label_by_col[c] = c
    lines = ["Кнг_W=%{x:.3f}", "Kng_model=%{y:.3f}"]
    for i, col in enumerate(hover_cols):
        lines.append(f"{label_by_col[col]}=%{{customdata[{i}]:.3f}}")
    return hover_cols, "<br>".join(lines)


def _crossplot_points_from_snapshot(df_snap: pd.DataFrame) -> pd.DataFrame:
    """Точки для кроссплота: без Кнг_hist = 0."""
    if df_snap.empty or "Кнг_hist" not in df_snap.columns or "Кнг_model" not in df_snap.columns:
        return pd.DataFrame()
    y = pd.to_numeric(df_snap["Кнг_hist"], errors="coerce")
    p = pd.to_numeric(df_snap["Кнг_model"], errors="coerce")
    mask = y.notna() & p.notna() & (y.abs() > 1e-15)
    return df_snap.loc[mask].copy()


def _build_well_crossplot_table(df_snap: pd.DataFrame) -> pd.DataFrame:
    """Средневзвешенные по скважине точки для кроссплота (история vs модель)."""
    src = _crossplot_points_from_snapshot(df_snap)
    if src.empty:
        return pd.DataFrame()
    src = src.copy()
    src["Кнг_W"] = _series_1d(src, "Кнг_hist", "Кнг_W")
    src["Kng_model"] = _series_1d(src, "Кнг_model", "Kng_model")
    if "weight" not in src.columns:
        src["weight"] = 1.0
    cross = _well_weighted_crossplot_df(src)
    if cross.empty:
        return cross
    if "PVTNUM_GDM" in src.columns:
        well_region = (
            src.assign(_pvt_num=pd.to_numeric(src["PVTNUM_GDM"], errors="coerce"))
            .dropna(subset=["_pvt_num"])
            .groupby("WELL_NAME", as_index=False)["_pvt_num"]
            .median()
            .rename(columns={"_pvt_num": "Регион"})
        )
        well_region["Регион"] = well_region["Регион"].astype(int).astype(str)
        cross = cross.merge(well_region, on="WELL_NAME", how="left")
    return cross


def _well_crossplot_table_from_result(
    df: pd.DataFrame | None,
    hist_col: str = "Кнг_W",
    model_col: str = "Kng_model",
) -> pd.DataFrame:
    """Таблица точек кроссплота по всему результату расчёта (для разбивки по PVTNUM)."""
    if df is None or df.empty or hist_col not in df.columns or model_col not in df.columns:
        return pd.DataFrame()
    snap: dict[str, pd.Series] = {
        "Кнг_hist": pd.to_numeric(df[hist_col], errors="coerce"),
        "Кнг_model": pd.to_numeric(df[model_col], errors="coerce"),
    }
    if "WELL_NAME" in df.columns:
        snap["WELL_NAME"] = df["WELL_NAME"].astype(str)
    else:
        return pd.DataFrame()
    if "PVTNUM_GDM" in df.columns:
        snap["PVTNUM_GDM"] = df["PVTNUM_GDM"]
    if "weight" in df.columns:
        snap["weight"] = pd.to_numeric(df["weight"], errors="coerce")
    return _build_well_crossplot_table(pd.DataFrame(snap))


def _crossplot_qa_metrics_help_expander(key: str) -> None:
    """Краткие определения метрик таблицы «Невязка по кроссплоту»."""
    with st.expander("Что означают метрики в таблице", expanded=False):
        st.markdown(
            """
Каждая **скважина** на кроссплоте — одна точка: средневзвешенные по стволу Кнг истории и модели
(без ячеек с Кнг_hist = 0).

| Метрика | Смысл |
|---------|--------|
| **Скважин** | Число скважин, вошедших в расчёт по региону (или по всем регионам в первой строке). |
| **MAE** | Средняя абсолютная невязка «модель − история» по скважинам (в долях Кнг). Чем меньше, тем ближе к диагонали в среднем. |
| **SCORE** | 1 минус взвешенная средняя абсолютная невязка (как в таблице метрик качества). Ближе к **1** — лучше. |
| **Сходимость, %** | Насколько точки скважин в среднем близки к диагонали y = x (100 % — идеальное попадание). Удобна для сравнения с графиком. |
            """
        )


def _compute_well_crossplot_qa(
    cross_df: pd.DataFrame,
    x_col: str,
    y_col: str,
    region_col: str | None = None,
) -> pd.DataFrame:
    """
    Метрики невязки на кроссплоте «история — модель» (средневзвешенно по скважине).
    Веса — число ячеек скважины × средний вес наблюдений.
    """
    if cross_df.empty:
        return pd.DataFrame()

    def _row(g: pd.DataFrame, label: str) -> dict:
        x = pd.to_numeric(g[x_col], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(g[y_col], errors="coerce").to_numpy(dtype=float)
        w = pd.to_numeric(g.get("points", 1.0), errors="coerce").fillna(1.0).to_numpy(dtype=float)
        if "avg_weight" in g.columns:
            w = w * pd.to_numeric(g["avg_weight"], errors="coerce").fillna(1.0).to_numpy(dtype=float)
        m = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
        if m.sum() == 0:
            return {
                "Регион": label,
                "Скважин": 0,
                "MAE": np.nan,
                "SCORE": np.nan,
                "Сходимость, %": np.nan,
            }
        xt, yt, ww = x[m], y[m], w[m]
        err = yt - xt
        mae = float(np.average(np.abs(err), weights=ww))
        score = float(1.0 - (np.sum(ww * np.abs(err)) / np.sum(ww)))
        conv = pd.to_numeric(g.get("convergence_percent", np.nan), errors="coerce").to_numpy(dtype=float)[m]
        mean_conv = float(np.average(conv, weights=ww)) if np.isfinite(conv).any() else float("nan")
        return {
            "Регион": label,
            "Скважин": int(m.sum()),
            "MAE": mae,
            "SCORE": score,
            "Сходимость, %": mean_conv,
        }

    rows = [_row(cross_df, "Все регионы")]
    reg_col = region_col if region_col and region_col in cross_df.columns else None
    if reg_col is None and "Регион" in cross_df.columns:
        reg_col = "Регион"
    if reg_col:
        sub = cross_df.dropna(subset=[reg_col]).copy()
        for reg, g in sorted(sub.groupby(reg_col, dropna=False), key=lambda x: str(x[0])):
            rows.append(_row(g, str(reg)))
    out = pd.DataFrame(rows)
    out["_ord"] = pd.to_numeric(out["Регион"], errors="coerce")
    out.loc[out["Регион"] == "Все регионы", "_ord"] = -1
    return out.sort_values("_ord", na_position="last").drop(columns="_ord").reset_index(drop=True)


def _render_well_crossplot_qa_panel(
    cross_df: pd.DataFrame,
    x_col: str,
    y_col: str,
    region_col: str | None,
    block_key: str,
    method_label: str,
    *,
    show_metrics_help: bool = False,
    show_title: bool = True,
) -> None:
    """Таблица невязки по кроссплоту скважин: строка «Все регионы» и по каждому PVTNUM."""
    if show_metrics_help:
        _crossplot_qa_metrics_help_expander(key=f"{block_key}_help")
    reg_col = region_col if region_col and region_col in cross_df.columns else None
    if reg_col is None and "Регион" in cross_df.columns:
        reg_col = "Регион"
    qa = _compute_well_crossplot_qa(cross_df, x_col=x_col, y_col=y_col, region_col=reg_col)
    if qa.empty:
        st.info(f"{method_label}: нет данных для оценки невязки по кроссплоту.")
        return
    if show_title:
        st.markdown(f"**{method_label}**")
    st.dataframe(_round_df(qa.set_index("Регион")), use_container_width=True)
    st.download_button(
        f"Скачать таблицу ({method_label}, CSV)",
        data=_csv_bytes(qa),
        file_name=f"crossplot_wells_qa_{block_key}.csv",
        mime="text/csv",
        key=f"{block_key}_dl_cross_qa",
    )


def _compute_qa_metrics(df: pd.DataFrame, true_col: str, pred_col: str) -> pd.DataFrame:
    from sklearn.metrics import r2_score

    def _row(g: pd.DataFrame, label: int | str) -> dict:
        y_true = pd.to_numeric(g[true_col], errors="coerce").to_numpy()
        y_pred = pd.to_numeric(g[pred_col], errors="coerce").to_numpy()
        w = pd.to_numeric(g.get("weight", 1.0), errors="coerce").fillna(1.0).to_numpy()
        m = np.isfinite(y_true) & np.isfinite(y_pred) & np.isfinite(w)
        if m.sum() == 0:
            return {
                "PVTNUM_GDM": label,
                "MAE": np.nan,
                "RMSE": np.nan,
                "BIAS": np.nan,
                "R2": np.nan,
                "SCORE": np.nan,
            }
        yt, yp, ww = y_true[m], y_pred[m], w[m]
        err = yp - yt
        mae = float(np.average(np.abs(err), weights=ww))
        rmse = float(np.sqrt(np.average(err**2, weights=ww)))
        bias = float(np.average(err, weights=ww))
        score = float(1 - (np.sum(ww * np.abs(err)) / np.sum(ww)))
        r2 = float(r2_score(yt, yp)) if len(yt) > 1 else np.nan
        return {
            "PVTNUM_GDM": label,
            "MAE": mae,
            "RMSE": rmse,
            "BIAS": bias,
            "R2": r2,
            "SCORE": score,
        }

    rows = [_row(g, int(float(pvt_raw))) for pvt_raw, g in df.groupby("PVTNUM_GDM")]
    regional = pd.DataFrame(rows)
    if not regional.empty:
        regional = regional.sort_values(
            by="PVTNUM_GDM",
            key=lambda s: pd.to_numeric(s, errors="coerce"),
        ).reset_index(drop=True)
    global_row = _row(df, "Все регионы")
    return pd.concat([pd.DataFrame([global_row]), regional], ignore_index=True)


def _qa_metrics_from_snapshot(df_snap: pd.DataFrame) -> pd.DataFrame:
    """Метрики по снимку скважин — та же формула, что на вкладках J и БК (_compute_qa_metrics)."""
    if df_snap.empty or "Кнг_hist" not in df_snap.columns or "Кнг_model" not in df_snap.columns:
        return pd.DataFrame()
    work = df_snap.copy()
    if "PVTNUM_GDM" not in work.columns:
        work["PVTNUM_GDM"] = 0
    qa = _compute_qa_metrics(work, "Кнг_hist", "Кнг_model")

    def _n_points(g: pd.DataFrame) -> int:
        y = pd.to_numeric(g["Кнг_hist"], errors="coerce")
        p = pd.to_numeric(g["Кнг_model"], errors="coerce")
        w = pd.to_numeric(g.get("weight", 1.0), errors="coerce")
        return int((y.notna() & p.notna() & w.notna()).sum())

    n_list: list[int] = []
    for label in qa["PVTNUM_GDM"]:
        if str(label) == "Все регионы":
            n_list.append(_n_points(work))
        else:
            pvt_num = pd.to_numeric(work["PVTNUM_GDM"], errors="coerce")
            mask = pvt_num == float(label)
            n_list.append(_n_points(work.loc[mask]))
    qa.insert(1, "N_points", n_list)
    return qa


def _build_well_snapshot(df: pd.DataFrame, model_col: str) -> pd.DataFrame:
    if "WELL_NAME" not in df.columns or "Кнг_W" not in df.columns or model_col not in df.columns:
        return pd.DataFrame()
    work = df.copy()
    work["WELL_NAME"] = work["WELL_NAME"].astype(str)
    depth_col = _pick_depth_column(work)
    if depth_col is None:
        work["_AXIS"] = work.groupby("WELL_NAME").cumcount() + 1
        axis_kind = "index"
    else:
        work["_AXIS"] = pd.to_numeric(work[depth_col], errors="coerce")
        axis_kind = "depth"
    work["Кнг_hist"] = pd.to_numeric(work["Кнг_W"], errors="coerce")
    work["Кнг_model"] = pd.to_numeric(work[model_col], errors="coerce")
    keep_cols = ["WELL_NAME", "_AXIS", "Кнг_hist", "Кнг_model"]
    if "PVTNUM_GDM" in work.columns:
        keep_cols.append("PVTNUM_GDM")
    if "weight" in work.columns:
        keep_cols.append("weight")
    if "ACTNUM_GDM" in work.columns:
        keep_cols.append("ACTNUM_GDM")
    out = work[keep_cols].dropna(subset=["WELL_NAME", "_AXIS", "Кнг_hist", "Кнг_model"]).copy()
    if out.empty:
        return out
    out["AXIS_KIND"] = axis_kind
    out = out.sort_values(["WELL_NAME", "_AXIS"]).reset_index(drop=True)
    return out


def _save_well_snapshot(df: pd.DataFrame, model_col: str, method_tag: str) -> tuple[bool, str]:
    snap = _build_well_snapshot(df, model_col=model_col)
    if snap.empty:
        return False, "Не удалось сохранить: нет валидных скважинных точек (WELL_NAME/Кнг_W/модель)."
    now = pd.Timestamp.now()
    snap_id = f"{method_tag}-{now.strftime('%Y%m%d-%H%M%S')}"
    snap["METHOD"] = method_tag
    snap["SNAPSHOT_ID"] = snap_id
    snap["SNAPSHOT_LABEL"] = f"{method_tag} | {now.strftime('%Y-%m-%d %H:%M:%S')}"
    snap["SAVED_AT"] = now.isoformat()
    all_snaps = st.session_state.get("well_method_snapshots")
    if not isinstance(all_snaps, pd.DataFrame) or all_snaps.empty:
        st.session_state["well_method_snapshots"] = snap
    else:
        st.session_state["well_method_snapshots"] = pd.concat([all_snaps, snap], ignore_index=True)
    return True, f"Сохранено: {snap['WELL_NAME'].nunique()} скв., {len(snap)} точек ({method_tag})."


def _snapshot_catalog(method_tag: str) -> pd.DataFrame:
    snaps = st.session_state.get("well_method_snapshots")
    if not isinstance(snaps, pd.DataFrame) or snaps.empty:
        return pd.DataFrame()
    sub = snaps[snaps["METHOD"] == method_tag].copy()
    if sub.empty:
        return sub
    cat = (
        sub.groupby(["SNAPSHOT_ID", "SNAPSHOT_LABEL"], as_index=False)
        .agg(wells=("WELL_NAME", "nunique"), points=("WELL_NAME", "size"), saved_at=("SAVED_AT", "max"))
        .sort_values("saved_at", ascending=False)
        .reset_index(drop=True)
    )
    return cat


def _get_well_method_snapshots() -> pd.DataFrame:
    snaps = st.session_state.get("well_method_snapshots")
    if not isinstance(snaps, pd.DataFrame):
        return pd.DataFrame()
    return snaps.copy()


def _snapshot_counts_text() -> str:
    snaps = _get_well_method_snapshots()
    if snaps.empty:
        return "В памяти сессии снимков нет."
    n_j = int((snaps["METHOD"] == "J").sum()) if "METHOD" in snaps.columns else 0
    n_bc = int((snaps["METHOD"] == "BC").sum()) if "METHOD" in snaps.columns else 0
    cat_j = len(_snapshot_catalog("J"))
    cat_bc = len(_snapshot_catalog("BC"))
    return (
        f"В памяти: **{cat_j}** снимок(ов) J ({n_j} точек), **{cat_bc}** снимок(ов) БК ({n_bc} точек)."
    )


def _default_snapshot_filename() -> str:
    return f"snapshots_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"


def _default_snapshot_save_path() -> str:
    SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    return str((SNAPSHOTS_DIR / _default_snapshot_filename()).relative_to(PROJECT_ROOT))


def _resolve_snapshot_path(user_path: str, *, default_filename: str) -> Path:
    raw = (user_path or "").strip()
    if not raw:
        raise ValueError("Укажите путь к файлу (полный или относительно папки проекта).")
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    if p.exists() and p.is_dir():
        p = p / default_filename
    elif raw.endswith(("/", "\\")) or (not p.suffix and not p.exists()):
        p = p / default_filename
    return p.resolve()


def _normalize_loaded_snapshots(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    missing = [c for c in SNAPSHOT_REQUIRED_COLUMNS if c not in work.columns]
    if missing:
        raise ValueError(f"В файле нет обязательных столбцов: {', '.join(missing)}")
    work["METHOD"] = work["METHOD"].astype(str).str.upper()
    bad = set(work["METHOD"].unique()) - {"J", "BC"}
    if bad:
        raise ValueError(f"Недопустимые значения METHOD: {', '.join(sorted(bad))}")
    if "SAVED_AT" not in work.columns:
        work["SAVED_AT"] = pd.Timestamp.now().isoformat()
    if "AXIS_KIND" not in work.columns:
        work["AXIS_KIND"] = "depth"
    for col in ("Кнг_hist", "Кнг_model", "_AXIS"):
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work["WELL_NAME"] = work["WELL_NAME"].astype(str)
    work = work.dropna(subset=["WELL_NAME", "_AXIS", "Кнг_hist", "Кнг_model"])
    if work.empty:
        raise ValueError("После проверки не осталось валидных строк снимка.")
    return work.reset_index(drop=True)


def _read_snapshots_file(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Файл не найден: {path}")
    if path.suffix.lower() not in {".csv", ".txt"}:
        raise ValueError("Поддерживается файл .csv (UTF-8).")
    text = _decode_upload_text(path.read_bytes())
    df = _read_csv_text(text)
    return _normalize_loaded_snapshots(df)


def _write_snapshots_file(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def _gui_dialogs_available() -> bool:
    """Системный диалог выбора файла возможен только при локальном GUI (не CI / headless)."""
    if sys.platform == "darwin" and os.environ.get("CI"):
        return False
    headless = os.environ.get("STREAMLIT_SERVER_HEADLESS", "").lower() in ("1", "true", "yes")
    return not headless


def _pick_csv_save_path_macos(default_name: str, initial_dir: Path) -> str | None:
    dir_posix = str(initial_dir.resolve())
    name = default_name.replace("\\", "\\\\").replace('"', '\\"')
    folder = dir_posix.replace("\\", "\\\\").replace('"', '\\"')
    script = (
        f'set defaultLoc to POSIX file "{folder}"\n'
        f'set outPath to choose file name with prompt "Сохранить снимки как:" '
        f'default name "{name}" default location defaultLoc\n'
        "return POSIX path of outPath"
    )
    try:
        proc = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    path = (proc.stdout or "").strip()
    return path if path else None


def _pick_csv_save_path_windows(default_name: str, initial_dir: Path) -> str | None:
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.update_idletasks()
        try:
            root.attributes("-topmost", True)
        except tk.TclError:
            pass
        path = filedialog.asksaveasfilename(
            title="Сохранить снимки",
            defaultextension=".csv",
            initialfile=default_name,
            initialdir=str(initial_dir.resolve()),
            filetypes=[("CSV", "*.csv"), ("Все файлы", "*.*")],
        )
        root.destroy()
        return str(path) if path else None
    except Exception:
        return None


def _pick_csv_save_path(default_name: str) -> tuple[str | None, str | None]:
    """
    Диалог «Сохранить как…» для локального Streamlit.
    Возвращает (путь, сообщение_об_ошибке). На macOS tkinter часто роняет процесс Streamlit — используем osascript.
    """
    SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    initial_dir = SNAPSHOTS_DIR.resolve()
    if not _gui_dialogs_available():
        return None, (
            "Системный диалог недоступен (режим без GUI или удалённый сервер). "
            "Введите путь в поле ниже или нажмите «Скачать CSV в браузер»."
        )
    system = platform.system()
    if system == "Darwin":
        path = _pick_csv_save_path_macos(default_name, initial_dir)
        if path:
            return path, None
        return None, "Диалог отменён или macOS не разрешил выбор файла (нет доступа к экрану для терминала)."
    if system == "Windows":
        path = _pick_csv_save_path_windows(default_name, initial_dir)
        if path:
            return path, None
        return None, "Диалог отменён или окно выбора файла не открылось."
    return None, (
        "На этой ОС диалог «Сохранить как…» не настроен. "
        "Укажите путь вручную или скачайте CSV через браузер."
    )


def _merge_snapshots(existing: pd.DataFrame, loaded: pd.DataFrame) -> pd.DataFrame:
    """Добавить снимки из файла к запомненным в сессии; при том же SNAPSHOT_ID — из файла."""
    if existing.empty:
        return loaded.copy()
    if loaded.empty:
        return existing.copy()
    loaded_ids = set(loaded["SNAPSHOT_ID"].astype(str).unique())
    kept = existing[~existing["SNAPSHOT_ID"].astype(str).isin(loaded_ids)].copy()
    return pd.concat([kept, loaded], ignore_index=True)


def _commit_snapshots_to_session(
    loaded: pd.DataFrame,
    *,
    src: str,
    notice_key: str,
    rerun: bool = True,
) -> None:
    current = _get_well_method_snapshots()
    if current.empty:
        combined = loaded.copy()
        st.session_state[notice_key] = (
            f"Загружено {len(loaded)} строк из «{src}». "
            f"В памяти: {len(_snapshot_catalog('J'))} снимок(ов) J, {len(_snapshot_catalog('BC'))} снимок(ов) БК."
        )
    else:
        before_ids = set(current["SNAPSHOT_ID"].astype(str).unique())
        combined = _merge_snapshots(current, loaded)
        after_ids = set(combined["SNAPSHOT_ID"].astype(str).unique())
        added_ids = len(after_ids - before_ids)
        st.session_state[notice_key] = (
            f"Из «{src}» добавлено снимков: {added_ids} (строк: {len(loaded)}). "
            f"Снимки текущей сессии сохранены. "
            f"Всего: {len(_snapshot_catalog('J'))} J, {len(_snapshot_catalog('BC'))} БК."
        )
    st.session_state["well_method_snapshots"] = combined
    if rerun:
        st.rerun()


def _render_snapshot_disk_panel(key_prefix: str = "snap_disk") -> None:
    """Сохранение / загрузка каталога well_method_snapshots в CSV (без сворачиваемых блоков)."""
    st.markdown("### Снимки на диск")
    st.caption(
        "Сначала «Запомнить» на вкладках J и БК. Здесь — запись и чтение CSV; загрузка **добавляет** снимки к уже "
        "запомненным в сессии (при совпадении ID — версия из файла)."
    )
    notice_key = f"{key_prefix}_load_notice"
    if st.session_state.get(notice_key):
        st.success(st.session_state.pop(notice_key))

    st.markdown(_snapshot_counts_text())

    default_name = _default_snapshot_filename()
    save_path_key = f"{key_prefix}_save_path"
    save_path_pending_key = f"{key_prefix}_save_path_pending"
    load_path_key = f"{key_prefix}_load_path"

    pending_save_path = st.session_state.pop(save_path_pending_key, None)
    if pending_save_path is not None:
        st.session_state[save_path_key] = pending_save_path
    elif save_path_key not in st.session_state:
        st.session_state[save_path_key] = _default_snapshot_save_path()
    if load_path_key not in st.session_state:
        st.session_state[load_path_key] = str((SNAPSHOTS_DIR / "snapshots.csv").relative_to(PROJECT_ROOT))

    st.subheader("Сохранение")
    st.text_input(
        "Куда сохранить CSV",
        key=save_path_key,
        help="Полный или относительный путь, например data/snapshots/имя.csv",
    )
    snaps = _get_well_method_snapshots()
    b_save, b_pick, b_dl = st.columns([2, 2, 2])
    with b_save:
        if st.button("Сохранить в файл", type="primary", key=f"{key_prefix}_save_btn", use_container_width=True):
            if snaps.empty:
                st.warning("Нет снимков. Сначала «Запомнить» на вкладках J и/или БК.")
            else:
                try:
                    save_path_str = str(st.session_state.get(save_path_key, ""))
                    out_path = _resolve_snapshot_path(save_path_str, default_filename=default_name)
                    _write_snapshots_file(out_path, snaps)
                    if str(out_path) != save_path_str.strip():
                        st.session_state[save_path_pending_key] = str(out_path)
                        st.rerun()
                    else:
                        st.success(f"Сохранено {len(snaps)} строк → `{out_path}`")
                except (OSError, ValueError) as e:
                    st.error(str(e))
    with b_pick:
        if st.button("Выбрать путь…", key=f"{key_prefix}_pick_save", use_container_width=True):
            picked, err = _pick_csv_save_path(default_name)
            if picked:
                st.session_state[save_path_pending_key] = picked
                st.rerun()
            elif err:
                st.warning(err)
    with b_dl:
        if not snaps.empty:
            st.download_button(
                "Скачать CSV",
                data=_csv_bytes(snaps),
                file_name=default_name,
                mime="text/csv",
                key=f"{key_prefix}_download",
                use_container_width=True,
                help="Файл в папку «Загрузки» браузера.",
            )

    st.divider()
    st.subheader("Загрузка")
    up = st.file_uploader(
        "Выбрать CSV на компьютере",
        type=["csv"],
        key=f"{key_prefix}_upload",
        help="После выбора файла данные подгружаются сразу.",
    )
    if up is not None:
        upload_sig = f"{up.name}:{up.size}"
        if st.session_state.get(f"{key_prefix}_upload_sig") != upload_sig:
            try:
                loaded = _normalize_loaded_snapshots(_read_csv_text(_decode_upload_text(up.getvalue())))
                st.session_state[f"{key_prefix}_upload_sig"] = upload_sig
                _commit_snapshots_to_session(loaded, src=up.name, notice_key=notice_key)
            except (ValueError, pd.errors.ParserError) as e:
                st.error(str(e))

    load_col, load_btn_col = st.columns([4, 1])
    with load_col:
        load_path_str = st.text_input(
            "Или путь к CSV на диске",
            key=load_path_key,
            help="Относительно папки проекта или полный путь.",
        )
    with load_btn_col:
        st.write("")
        st.write("")
        if st.button("Загрузить", key=f"{key_prefix}_load_path_btn", use_container_width=True):
            try:
                in_path = _resolve_snapshot_path(load_path_str, default_filename="snapshots.csv")
                loaded = _read_snapshots_file(in_path)
                st.session_state[f"{key_prefix}_upload_sig"] = None
                _commit_snapshots_to_session(loaded, src=str(in_path), notice_key=notice_key)
            except (OSError, ValueError, pd.errors.ParserError) as e:
                st.error(str(e))


def _render_methods_comparison_block(block_key: str = "compare_methods") -> None:
    cat_j = _snapshot_catalog("J")
    cat_bc = _snapshot_catalog("BC")
    if cat_j.empty or cat_bc.empty:
        st.info("Сохраните результаты обоих методов кнопкой «Запомнить результаты по скважинам».")
        return

    def _crossplot_df(df_snap: pd.DataFrame) -> pd.DataFrame:
        return _crossplot_points_from_snapshot(df_snap)

    def _weighted_region_table(df_snap: pd.DataFrame) -> pd.DataFrame:
        if df_snap.empty:
            return pd.DataFrame()
        work = df_snap.copy()
        work["Кнг_hist"] = pd.to_numeric(work["Кнг_hist"], errors="coerce")
        work["Кнг_model"] = pd.to_numeric(work["Кнг_model"], errors="coerce")
        work["weight"] = pd.to_numeric(work.get("weight", 1.0), errors="coerce").fillna(1.0)
        work = work[np.isfinite(work["Кнг_hist"]) & np.isfinite(work["Кнг_model"]) & (work["weight"] > 0)]
        if work.empty:
            return pd.DataFrame()

        def _row(g: pd.DataFrame, label: str | int | float) -> dict | None:
            w = g["weight"].to_numpy(dtype=float)
            sw = float(np.sum(w))
            if sw <= 0:
                return None
            hist_w = float(np.sum(w * g["Кнг_hist"].to_numpy(dtype=float)) / sw)
            model_w = float(np.sum(w * g["Кнг_model"].to_numpy(dtype=float)) / sw)
            return {
                "Регион": label,
                "Средневзвешенное Кнг (история)": hist_w,
                "Средневзвешенное Кнг (модель)": model_w,
                "Дельта": model_w - hist_w,
                "Точек": int(len(g)),
            }

        rows: list[dict] = []
        all_row = _row(work, "Все регионы")
        if all_row:
            rows.append(all_row)

        region_col = "PVTNUM_GDM" if "PVTNUM_GDM" in work.columns else None
        if region_col is not None and work[region_col].notna().any():
            for region, g in work.groupby(region_col, dropna=False):
                r = _row(g, region)
                if r:
                    rows.append(r)

        if not rows:
            return pd.DataFrame()
        tab = pd.DataFrame(rows)
        tab["_ord"] = pd.to_numeric(tab["Регион"], errors="coerce")
        tab.loc[tab["Регион"].astype(str) == "Все регионы", "_ord"] = -1
        tab = tab.sort_values("_ord", na_position="last").drop(columns="_ord").reset_index(drop=True)
        return tab

    def _kng_hist_stats(arr: np.ndarray) -> list[float]:
        """Сводные по истории: без точек с Кнг_hist = 0 (нефтяные интервалы)."""
        a = arr[np.isfinite(arr) & (arr > 1e-15)]
        if a.size == 0:
            return [np.nan] * 5
        return [
            float(np.min(a)),
            float(np.max(a)),
            float(np.mean(a)),
            float(np.median(a)),
            float(np.std(a)),
        ]

    def _kng_model_stats(arr: np.ndarray) -> list[float]:
        a = arr[np.isfinite(arr)]
        if a.size == 0:
            return [np.nan] * 5
        return [
            float(np.nanmin(a)),
            float(np.nanmax(a)),
            float(np.nanmean(a)),
            float(np.nanmedian(a)),
            float(np.nanstd(a)),
        ]

    def _hist_percent(df_snap: pd.DataFrame, region_pick: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        if df_snap.empty:
            return pd.DataFrame(), pd.DataFrame()
        work = df_snap.copy()
        work["Кнг_hist"] = pd.to_numeric(work["Кнг_hist"], errors="coerce")
        work["Кнг_model"] = pd.to_numeric(work["Кнг_model"], errors="coerce")
        work = work[np.isfinite(work["Кнг_hist"]) & np.isfinite(work["Кнг_model"])]
        if work.empty:
            return pd.DataFrame(), pd.DataFrame()
        if (region_pick != "Все регионы") and ("PVTNUM_GDM" in work.columns):
            work = work[work["PVTNUM_GDM"].astype(str) == str(region_pick)]
        if work.empty:
            return pd.DataFrame(), pd.DataFrame()
        hist = work["Кнг_hist"].to_numpy(dtype=float)
        model = work["Кнг_model"].to_numpy(dtype=float)
        lo = float(np.nanmin(np.r_[hist, model]))
        hi = float(np.nanmax(np.r_[hist, model]))
        if not np.isfinite(lo) or not np.isfinite(hi):
            return pd.DataFrame(), pd.DataFrame()
        if hi <= lo:
            hi = lo + 1e-6
        step = 0.02
        lo_b = 0.5
        hi_b = step * np.ceil(hi / step)
        if hi_b <= lo_b:
            hi_b = lo_b + step
        bins = np.arange(lo_b, hi_b + step * 1.001, step)
        if len(bins) < 2:
            bins = np.array([lo_b, lo_b + step], dtype=float)
        h_hist, edges = np.histogram(hist, bins=bins)
        h_model, _ = np.histogram(model, bins=bins)
        centers = 0.5 * (edges[:-1] + edges[1:])
        denom_h = max(1, int(h_hist.sum()))
        denom_m = max(1, int(h_model.sum()))
        hist_df = pd.DataFrame(
            {
                "Нефтенасыщенность": centers,
                "История, %": 100.0 * h_hist / denom_h,
                "Модель, %": 100.0 * h_model / denom_m,
            }
        )
        stats = pd.DataFrame(
            {
                "Показатель": ["Минимум", "Максимум", "Среднее", "Медиана", "Станд. отклонение"],
                "История": _kng_hist_stats(hist),
                "Модель": _kng_model_stats(model),
            }
        )
        return hist_df, stats

    def _models_hist_percent(
        df_j: pd.DataFrame, df_bc: pd.DataFrame, region_pick: str
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        if df_j.empty or df_bc.empty:
            return pd.DataFrame(), pd.DataFrame()
        j = df_j.copy()
        b = df_bc.copy()
        for frame in (j, b):
            frame["_AXIS_R"] = pd.to_numeric(frame["_AXIS"], errors="coerce")
            frame["WELL_NAME"] = frame["WELL_NAME"].astype(str)
        mj = j[["WELL_NAME", "_AXIS_R", "Кнг_model"] + (["PVTNUM_GDM"] if "PVTNUM_GDM" in j.columns else [])].rename(
            columns={"Кнг_model": "J_model", "PVTNUM_GDM": "PVT_J"}
        )
        mb = b[["WELL_NAME", "_AXIS_R", "Кнг_model"] + (["PVTNUM_GDM"] if "PVTNUM_GDM" in b.columns else [])].rename(
            columns={"Кнг_model": "BC_model", "PVTNUM_GDM": "PVT_BC"}
        )
        m = mj.merge(mb, on=["WELL_NAME", "_AXIS_R"], how="inner")
        if m.empty:
            return pd.DataFrame(), pd.DataFrame()
        if "PVT_J" in m.columns:
            m["PVTNUM_GDM"] = m["PVT_J"]
        elif "PVT_BC" in m.columns:
            m["PVTNUM_GDM"] = m["PVT_BC"]
        if (region_pick != "Все регионы") and ("PVTNUM_GDM" in m.columns):
            m = m[pd.to_numeric(m["PVTNUM_GDM"], errors="coerce") == float(region_pick)]
        m["J_model"] = pd.to_numeric(m["J_model"], errors="coerce")
        m["BC_model"] = pd.to_numeric(m["BC_model"], errors="coerce")
        m = m[np.isfinite(m["J_model"]) & np.isfinite(m["BC_model"])]
        if m.empty:
            return pd.DataFrame(), pd.DataFrame()
        jv = m["J_model"].to_numpy(dtype=float)
        bv = m["BC_model"].to_numpy(dtype=float)
        lo = float(np.nanmin(np.r_[jv, bv]))
        hi = float(np.nanmax(np.r_[jv, bv]))
        if hi <= lo:
            hi = lo + 1e-6
        step = 0.02
        lo_b = 0.5
        hi_b = step * np.ceil(hi / step)
        if hi_b <= lo_b:
            hi_b = lo_b + step
        bins = np.arange(lo_b, hi_b + step * 1.001, step)
        if len(bins) < 2:
            bins = np.array([lo_b, lo_b + step], dtype=float)
        h_j, edges = np.histogram(jv, bins=bins)
        h_b, _ = np.histogram(bv, bins=bins)
        centers = 0.5 * (edges[:-1] + edges[1:])
        hist_df = pd.DataFrame(
            {
                "Нефтенасыщенность": centers,
                "J-функция, %": 100.0 * h_j / max(1, int(h_j.sum())),
                "Брукс-Кори, %": 100.0 * h_b / max(1, int(h_b.sum())),
            }
        )
        stats_df = pd.DataFrame(
            {
                "Показатель": ["Минимум", "Максимум", "Среднее", "Медиана", "Станд. отклонение"],
                "J-функция": [
                    float(np.nanmin(jv)),
                    float(np.nanmax(jv)),
                    float(np.nanmean(jv)),
                    float(np.nanmedian(jv)),
                    float(np.nanstd(jv)),
                ],
                "Брукс-Кори": [
                    float(np.nanmin(bv)),
                    float(np.nanmax(bv)),
                    float(np.nanmean(bv)),
                    float(np.nanmedian(bv)),
                    float(np.nanstd(bv)),
                ],
            }
        )
        return hist_df, stats_df

    def _corr(a: np.ndarray, b: np.ndarray) -> float:
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < 3:
            return float("nan")
        return float(np.corrcoef(a[m], b[m])[0, 1])

    def _trend_match(a: np.ndarray, b: np.ndarray) -> float:
        da = np.diff(a)
        db = np.diff(b)
        m = np.isfinite(da) & np.isfinite(db) & (np.abs(da) > 1e-9) & (np.abs(db) > 1e-9)
        if m.sum() == 0:
            return float("nan")
        return float(100.0 * np.mean(np.sign(da[m]) == np.sign(db[m])))

    j_labels = [f"{r.SNAPSHOT_LABEL} | скв:{int(r.wells)} тчк:{int(r.points)}" for r in cat_j.itertuples()]
    bc_labels = [f"{r.SNAPSHOT_LABEL} | скв:{int(r.wells)} тчк:{int(r.points)}" for r in cat_bc.itertuples()]
    map_j = dict(zip(j_labels, cat_j["SNAPSHOT_ID"]))
    map_bc = dict(zip(bc_labels, cat_bc["SNAPSHOT_ID"]))
    c1, c2 = st.columns(2)
    j_pick = c1.selectbox("Снимок J-функции", options=j_labels, key=f"{block_key}_j_pick")
    bc_pick = c2.selectbox("Снимок Брукса-Кори", options=bc_labels, key=f"{block_key}_bc_pick")
    sid_j = map_j[j_pick]
    sid_bc = map_bc[bc_pick]

    snaps = st.session_state.get("well_method_snapshots").copy()
    sj = snaps[(snaps["METHOD"] == "J") & (snaps["SNAPSHOT_ID"] == sid_j)].copy()
    sb = snaps[(snaps["METHOD"] == "BC") & (snaps["SNAPSHOT_ID"] == sid_bc)].copy()
    common_wells = sorted(set(sj["WELL_NAME"].astype(str)) & set(sb["WELL_NAME"].astype(str)))
    if not common_wells:
        st.warning("Между выбранными снимками нет общих скважин.")
        return
    st.caption(f"Общих скважин: {len(common_wells)}")

    st.markdown("### Метрики по каждому методу (по горизонтам, Регион)")
    st.caption(
        "Те же взвешенные MAE/RMSE/BIAS/R²/SCORE, что на вкладках J и БК после расчёта "
        "(по сохранённому снимку). На кроссплотах ниже по-прежнему не учитываются точки с Кнг_hist = 0."
    )
    tab_j = _qa_metrics_from_snapshot(sj)
    tab_b = _qa_metrics_from_snapshot(sb)
    cjm, cbm = st.columns(2)
    cjm.dataframe(_round_df(tab_j.rename(columns={"PVTNUM_GDM": "Регион"}).set_index("Регион")), use_container_width=True)
    cbm.dataframe(_round_df(tab_b.rename(columns={"PVTNUM_GDM": "Регион"}).set_index("Регион")), use_container_width=True)

    st.markdown("### Кроссплоты по методам")
    c3, c4 = st.columns(2)
    sj_x = _crossplot_df(sj)
    sb_x = _crossplot_df(sb)
    if sj_x.empty:
        c3.info("Нет точек для кроссплота J после исключения Кнг_hist = 0.")
    else:
        sj_plot = sj_x.copy()
        if "PVTNUM_GDM" in sj_plot.columns:
            sj_plot["Регион"] = pd.to_numeric(sj_plot["PVTNUM_GDM"], errors="coerce").astype("Int64").astype(str)
        fig_j = px.scatter(
            sj_plot,
            x="Кнг_hist",
            y="Кнг_model",
            color="Регион" if "Регион" in sj_plot.columns else None,
            color_discrete_sequence=px.colors.qualitative.Dark24,
            hover_data=[c for c in ["_AXIS", "Регион"] if c in sj_plot.columns],
            title="J: расчетное(историческое)",
            opacity=0.65,
            **_crossplot_hover_name_kw(sj_plot),
        )
        fig_j.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(color="red", dash="dash"))
        _apply_crossplot_hover(fig_j, "Кнг_hist=%{x:.3f}<br>Кнг_model=%{y:.3f}", sj_plot)
        fig_j.update_layout(legend_title_text="Регион")
        c3.plotly_chart(fig_j, use_container_width=True)
    if sb_x.empty:
        c4.info("Нет точек для кроссплота БК после исключения Кнг_hist = 0.")
    else:
        sb_plot = sb_x.copy()
        if "PVTNUM_GDM" in sb_plot.columns:
            sb_plot["Регион"] = pd.to_numeric(sb_plot["PVTNUM_GDM"], errors="coerce").astype("Int64").astype(str)
        fig_bc = px.scatter(
            sb_plot,
            x="Кнг_hist",
            y="Кнг_model",
            color="Регион" if "Регион" in sb_plot.columns else None,
            color_discrete_sequence=px.colors.qualitative.Dark24,
            hover_data=[c for c in ["_AXIS", "Регион"] if c in sb_plot.columns],
            title="БК: расчетное(историческое)",
            opacity=0.65,
            **_crossplot_hover_name_kw(sb_plot),
        )
        fig_bc.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(color="red", dash="dash"))
        _apply_crossplot_hover(fig_bc, "Кнг_hist=%{x:.3f}<br>Кнг_model=%{y:.3f}", sb_plot)
        fig_bc.update_layout(legend_title_text="Регион")
        c4.plotly_chart(fig_bc, use_container_width=True)

    st.markdown("### Региональные средневзвешенные значения и распределения")
    meth_tab_j, meth_tab_bc, meth_tab_cmp = st.tabs(["J-функция", "Брукс-Кори", "J vs БК"])

    with meth_tab_j:
        tab_reg_j = _weighted_region_table(sj)
        if tab_reg_j.empty:
            st.info("Недостаточно данных для региональной сводки J-функции.")
        else:
            st.dataframe(_round_df(tab_reg_j), use_container_width=True)
            reg_opts = [str(x) for x in tab_reg_j["Регион"].tolist()]
            reg_pick = st.selectbox("Регион для гистограммы (J)", options=reg_opts, key=f"{block_key}_hist_reg_j")
            hist_df, stats_df = _hist_percent(sj, reg_pick)
            if hist_df.empty:
                st.info("Нет данных для гистограммы J-функции.")
            else:
                cc1, cc2 = st.columns([2, 1])
                fig_h = px.bar(
                    hist_df.melt(id_vars="Нефтенасыщенность", var_name="Источник", value_name="Проценты"),
                    x="Нефтенасыщенность",
                    y="Проценты",
                    color="Источник",
                    barmode="overlay",
                    opacity=0.6,
                    title=f"J-функция: распределение Кнг ({reg_pick})",
                )
                fig_h.update_xaxes(range=[0.5, None], dtick=0.02, tickformat=".2f")
                cc1.plotly_chart(fig_h, use_container_width=True)
                cc1.download_button(
                    "Скачать данные гистограммы (J, CSV)",
                    data=_csv_bytes(hist_df),
                    file_name=f"hist_j_{str(reg_pick).replace(' ', '_')}.csv",
                    mime="text/csv",
                    key=f"{block_key}_dl_hist_j",
                )
                cc2.markdown("<div style='height: 42px;'></div>", unsafe_allow_html=True)
                cc2.dataframe(_round_df(stats_df), use_container_width=True)
                cc2.download_button(
                    "Скачать статистику (J, CSV)",
                    data=_csv_bytes(stats_df),
                    file_name=f"hist_j_stats_{str(reg_pick).replace(' ', '_')}.csv",
                    mime="text/csv",
                    key=f"{block_key}_dl_hist_j_stats",
                )

    with meth_tab_bc:
        tab_reg_b = _weighted_region_table(sb)
        if tab_reg_b.empty:
            st.info("Недостаточно данных для региональной сводки Брукса-Кори.")
        else:
            st.dataframe(_round_df(tab_reg_b), use_container_width=True)
            reg_opts = [str(x) for x in tab_reg_b["Регион"].tolist()]
            reg_pick = st.selectbox("Регион для гистограммы (БК)", options=reg_opts, key=f"{block_key}_hist_reg_bc")
            hist_df, stats_df = _hist_percent(sb, reg_pick)
            if hist_df.empty:
                st.info("Нет данных для гистограммы Брукса-Кори.")
            else:
                cc1, cc2 = st.columns([2, 1])
                fig_h = px.bar(
                    hist_df.melt(id_vars="Нефтенасыщенность", var_name="Источник", value_name="Проценты"),
                    x="Нефтенасыщенность",
                    y="Проценты",
                    color="Источник",
                    barmode="overlay",
                    opacity=0.6,
                    title=f"Брукс-Кори: распределение Кнг ({reg_pick})",
                )
                fig_h.update_xaxes(range=[0.5, None], dtick=0.02, tickformat=".2f")
                cc1.plotly_chart(fig_h, use_container_width=True)
                cc1.download_button(
                    "Скачать данные гистограммы (БК, CSV)",
                    data=_csv_bytes(hist_df),
                    file_name=f"hist_bc_{str(reg_pick).replace(' ', '_')}.csv",
                    mime="text/csv",
                    key=f"{block_key}_dl_hist_bc",
                )
                cc2.markdown("<div style='height: 42px;'></div>", unsafe_allow_html=True)
                cc2.dataframe(_round_df(stats_df), use_container_width=True)
                cc2.download_button(
                    "Скачать статистику (БК, CSV)",
                    data=_csv_bytes(stats_df),
                    file_name=f"hist_bc_stats_{str(reg_pick).replace(' ', '_')}.csv",
                    mime="text/csv",
                    key=f"{block_key}_dl_hist_bc_stats",
                )

    with meth_tab_cmp:
        pvt_vals = sorted(
            set(
                [str(x) for x in pd.to_numeric(sj.get("PVTNUM_GDM"), errors="coerce").dropna().astype(int).tolist()]
                + [str(x) for x in pd.to_numeric(sb.get("PVTNUM_GDM"), errors="coerce").dropna().astype(int).tolist()]
            )
        )
        reg_opts = ["Все регионы"] + pvt_vals
        reg_pick = st.selectbox("Регион для гистограммы (J vs БК)", options=reg_opts, key=f"{block_key}_hist_reg_jbc")
        hist_df, stats_df = _models_hist_percent(sj, sb, reg_pick)
        if hist_df.empty:
            st.info("Недостаточно общих точек J и БК для построения гистограммы.")
        else:
            c1, c2 = st.columns([2, 1])
            fig_cmp = px.bar(
                hist_df.melt(id_vars="Нефтенасыщенность", var_name="Модель", value_name="Проценты"),
                x="Нефтенасыщенность",
                y="Проценты",
                color="Модель",
                barmode="overlay",
                opacity=0.6,
                title=f"Распределения предсказанной Кнг: J vs БК ({reg_pick})",
            )
            fig_cmp.update_xaxes(range=[0.5, None], dtick=0.02, tickformat=".2f")
            c1.plotly_chart(fig_cmp, use_container_width=True)
            c1.download_button(
                "Скачать данные гистограммы (J_vs_БК, CSV)",
                data=_csv_bytes(hist_df),
                file_name=f"hist_j_bc_{str(reg_pick).replace(' ', '_')}.csv",
                mime="text/csv",
                key=f"{block_key}_dl_hist_jbc",
            )
            c2.markdown("<div style='height: 42px;'></div>", unsafe_allow_html=True)
            c2.dataframe(_round_df(stats_df), use_container_width=True)
            c2.download_button(
                "Скачать статистику (J_vs_БК, CSV)",
                data=_csv_bytes(stats_df),
                file_name=f"hist_j_bc_stats_{str(reg_pick).replace(' ', '_')}.csv",
                mime="text/csv",
                key=f"{block_key}_dl_hist_jbc_stats",
            )

    st.markdown("### Кроссплоты по скважинам (средневзвешенно, все регионы)")
    cw1, cw2 = st.columns(2)

    sj_cross_src = _crossplot_df(sj)
    if sj_cross_src.empty:
        cw1.info("Недостаточно данных для кроссплота по скважинам (J).")
    else:
        sj_cross_src = sj_cross_src.rename(columns={"Кнг_hist": "Кнг_W", "Кнг_model": "Kng_model"})
        if "weight" not in sj_cross_src.columns:
            sj_cross_src["weight"] = 1.0
        sj_cross = _well_weighted_crossplot_df(sj_cross_src)
        if sj_cross.empty:
            cw1.info("Недостаточно данных для кроссплота по скважинам (J).")
        else:
            if "PVTNUM_GDM" in sj_cross_src.columns:
                well_region_j = (
                    sj_cross_src.assign(_pvt_num=pd.to_numeric(sj_cross_src["PVTNUM_GDM"], errors="coerce"))
                    .dropna(subset=["_pvt_num"])
                    .groupby("WELL_NAME", as_index=False)["_pvt_num"]
                    .median()
                    .rename(columns={"_pvt_num": "Регион"})
                )
                well_region_j["Регион"] = well_region_j["Регион"].astype(int).astype(str)
                sj_cross = sj_cross.merge(well_region_j, on="WELL_NAME", how="left")
            fig_jw = px.scatter(
                sj_cross,
                x="Кнг_W_wmean",
                y="Kng_model_wmean",
                color="Регион" if "Регион" in sj_cross.columns else None,
                color_discrete_sequence=px.colors.qualitative.Dark24,
                hover_data={
                    "Регион": True,
                    "points": ":.3f",
                    "avg_weight": ":.3f",
                    "convergence_percent": ":.3f",
                },
                title="J: кроссплот по скважинам",
                opacity=0.85,
                **_crossplot_hover_name_kw(sj_cross),
            )
            _apply_crossplot_hover(fig_jw, "Кнг_hist_wmean=%{x:.3f}<br>Кнг_J_wmean=%{y:.3f}", sj_cross)
            fig_jw.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(color="red", dash="dash"))
            fig_jw.update_layout(
                xaxis_title="Кнг_hist (средневзвеш.)",
                yaxis_title="Кнг_J (средневзвеш.)",
                legend_title_text="Регион",
            )
            cw1.plotly_chart(fig_jw, use_container_width=True)
            cw1.caption(f"Скважин (точек кроссплота): {len(sj_cross)}")

    sb_cross_src = _crossplot_df(sb)
    if sb_cross_src.empty:
        cw2.info("Недостаточно данных для кроссплота по скважинам (БК).")
    else:
        sb_cross_src = sb_cross_src.rename(columns={"Кнг_hist": "Кнг_W", "Кнг_model": "Kng_model"})
        if "weight" not in sb_cross_src.columns:
            sb_cross_src["weight"] = 1.0
        sb_cross = _well_weighted_crossplot_df(sb_cross_src).rename(columns={"Kng_model_wmean": "Kng_BC_wmean"})
        if sb_cross.empty:
            cw2.info("Недостаточно данных для кроссплота по скважинам (БК).")
        else:
            if "PVTNUM_GDM" in sb_cross_src.columns:
                well_region_b = (
                    sb_cross_src.assign(_pvt_num=pd.to_numeric(sb_cross_src["PVTNUM_GDM"], errors="coerce"))
                    .dropna(subset=["_pvt_num"])
                    .groupby("WELL_NAME", as_index=False)["_pvt_num"]
                    .median()
                    .rename(columns={"_pvt_num": "Регион"})
                )
                well_region_b["Регион"] = well_region_b["Регион"].astype(int).astype(str)
                sb_cross = sb_cross.merge(well_region_b, on="WELL_NAME", how="left")
            fig_bw = px.scatter(
                sb_cross,
                x="Кнг_W_wmean",
                y="Kng_BC_wmean",
                color="Регион" if "Регион" in sb_cross.columns else None,
                color_discrete_sequence=px.colors.qualitative.Dark24,
                hover_data={
                    "Регион": True,
                    "points": ":.3f",
                    "avg_weight": ":.3f",
                    "convergence_percent": ":.3f",
                },
                title="БК: кроссплот по скважинам",
                opacity=0.85,
                **_crossplot_hover_name_kw(sb_cross),
            )
            _apply_crossplot_hover(fig_bw, "Кнг_hist_wmean=%{x:.3f}<br>Кнг_БК_wmean=%{y:.3f}", sb_cross)
            fig_bw.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(color="red", dash="dash"))
            fig_bw.update_layout(
                xaxis_title="Кнг_hist (средневзвеш.)",
                yaxis_title="Кнг_БК (средневзвеш.)",
                legend_title_text="Регион",
            )
            cw2.plotly_chart(fig_bw, use_container_width=True)
            cw2.caption(f"Скважин (точек кроссплота): {len(sb_cross)}")

    st.markdown("### Невязка по кроссплоту (скважины)")
    st.caption(
        "Одна точка на скважину (средневзвешенные Кнг); первая строка — **все регионы**, "
        "далее — по PVTNUM. Ячейки с Кнг_hist = 0 не учитываются."
    )
    _crossplot_qa_metrics_help_expander(key=f"{block_key}_cross_qa_help")
    sj_cross_tbl = _build_well_crossplot_table(sj)
    sb_cross_tbl = _build_well_crossplot_table(sb).rename(columns={"Kng_model_wmean": "Kng_BC_wmean"})
    cjq, cbq = st.columns(2)
    with cjq:
        _render_well_crossplot_qa_panel(
            sj_cross_tbl,
            x_col="Кнг_W_wmean",
            y_col="Kng_model_wmean",
            region_col="Регион",
            block_key=f"{block_key}_j",
            method_label="J-функция",
            show_title=False,
        )
    with cbq:
        _render_well_crossplot_qa_panel(
            sb_cross_tbl,
            x_col="Кнг_W_wmean",
            y_col="Kng_BC_wmean",
            region_col="Регион",
            block_key=f"{block_key}_bc",
            method_label="Брукса — Кори",
            show_title=False,
        )

    st.markdown("### Поскважинное сравнение и кластеры согласованности")
    st.caption(
        "Для каждой скважины на общей сетке по глубине сравниваются **тренды** рассчитанных кривых "
        "J-функции и Брукса–Кори (корреляция и совпадение знака приращений). Связь с историей РИГИС "
        "приведена для справки."
    )
    rows = []
    for well in common_wells:
        wj = sj[sj["WELL_NAME"].astype(str) == well].copy().sort_values("_AXIS")
        wb = sb[sb["WELL_NAME"].astype(str) == well].copy().sort_values("_AXIS")
        if wj.empty or wb.empty:
            continue
        tj = wj.groupby("_AXIS", as_index=False).agg(Кнг_hist=("Кнг_hist", "mean"), Кнг_model=("Кнг_model", "mean"))
        tb = wb.groupby("_AXIS", as_index=False).agg(Кнг_hist=("Кнг_hist", "mean"), Кнг_model=("Кнг_model", "mean"))
        x_lo = max(float(tj["_AXIS"].min()), float(tb["_AXIS"].min()))
        x_hi = min(float(tj["_AXIS"].max()), float(tb["_AXIS"].max()))
        if x_hi <= x_lo:
            grid = np.array(sorted(set(tj["_AXIS"].tolist() + tb["_AXIS"].tolist())), dtype=float)
        else:
            grid = np.linspace(x_lo, x_hi, int(max(30, min(180, 2 * min(len(tj), len(tb))))))
        if len(grid) < 5:
            continue
        hj = np.interp(grid, tj["_AXIS"].to_numpy(dtype=float), tj["Кнг_hist"].to_numpy(dtype=float))
        hb = np.interp(grid, tb["_AXIS"].to_numpy(dtype=float), tb["Кнг_hist"].to_numpy(dtype=float))
        hist = 0.5 * (hj + hb)
        pj = np.interp(grid, tj["_AXIS"].to_numpy(dtype=float), tj["Кнг_model"].to_numpy(dtype=float))
        pb = np.interp(grid, tb["_AXIS"].to_numpy(dtype=float), tb["Кнг_model"].to_numpy(dtype=float))
        rows.append(
            {
                "WELL_NAME": str(well),
                "corr_j_bc": _corr(pj, pb),
                "trend_j_bc": _trend_match(pj, pb),
                "corr_j_hist": _corr(pj, hist),
                "corr_bc_hist": _corr(pb, hist),
                "points_interp": int(len(grid)),
            }
        )
    cmp_df = pd.DataFrame(rows)
    if cmp_df.empty:
        st.warning("Недостаточно данных для поскважинного сравнения.")
        return

    cmp_df["good_match"] = (cmp_df["corr_j_bc"] >= 0.8) & (cmp_df["trend_j_bc"] >= 65.0)
    st.metric("Скважины с хорошим совпадением J и БК (по трендам)", int(cmp_df["good_match"].sum()))
    disp_cols = ["WELL_NAME", "corr_j_bc", "trend_j_bc", "corr_j_hist", "corr_bc_hist", "good_match", "points_interp"]
    with st.expander("Как интерпретировать таблицу сравнения", expanded=False):
        st.markdown(
            "- `WELL_NAME` — название скважины.\n"
            "- `corr_j_bc` — корреляция между профилями J и БК (ближе к 1 — лучше согласие).\n"
            "- `trend_j_bc` — доля совпадающих направлений изменения J и БК по глубине, %.\n"
            "- `corr_j_hist` — корреляция профиля J с историческим профилем.\n"
            "- `corr_bc_hist` — корреляция профиля БК с историческим профилем.\n"
            "- `good_match` — флаг хорошего совпадения (истина, если `corr_j_bc >= 0.8` и `trend_j_bc >= 65%`).\n"
            "- `points_interp` — число интерполированных точек, по которым считались метрики."
        )
    st.dataframe(
        _round_df(cmp_df[disp_cols].sort_values(["good_match", "corr_j_bc", "trend_j_bc"], ascending=[False, False, False])),
        use_container_width=True,
    )

    feat = cmp_df[["corr_j_bc", "trend_j_bc"]].copy()
    feat = feat.fillna(feat.median(numeric_only=True))
    try:
        from sklearn.cluster import KMeans

        k = int(max(2, min(4, len(feat))))
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        cmp_df["cluster"] = km.fit_predict(feat).astype(str)
    except Exception:
        cmp_df["cluster"] = np.where(cmp_df["good_match"], "good", "other")

    fig_cluster = px.scatter(
        cmp_df,
        x="corr_j_bc",
        y="trend_j_bc",
        color="cluster",
        symbol="good_match",
        hover_data=["corr_j_hist", "corr_bc_hist", "points_interp"],
        title="Кластеры: согласованность J-функции и Брукса–Кори (корреляция и тренд)",
        **_crossplot_hover_name_kw(cmp_df),
    )
    _apply_crossplot_hover(fig_cluster, "corr_j_bc=%{x:.3f}<br>trend_j_bc=%{y:.3f}", cmp_df)
    with st.expander("Как интерпретировать график кластеров", expanded=False):
        st.markdown(
            "- Каждая точка — одна скважина.\n"
            "- Ось X (`corr_j_bc`) показывает, насколько форма кривых J и БК похожа.\n"
            "- Ось Y (`trend_j_bc`) показывает долю совпадения направления изменений по глубине, %.\n"
            "- Чем правее и выше точка, тем лучше согласованность методов по скважине.\n"
            "- `cluster` — метка кластера (группа скважин с похожей комбинацией `corr_j_bc` и `trend_j_bc`).\n"
            "- `good_match=True` — скважина в зоне хорошего согласия: `corr_j_bc >= 0.8` и `trend_j_bc >= 65%`.\n"
            "- `good_match=False` (или `0`) — хотя бы один из критериев не выполнен, согласованность ниже целевой."
        )
    st.plotly_chart(fig_cluster, use_container_width=True)

    well = st.selectbox("Скважина для детального сравнения", options=sorted(cmp_df["WELL_NAME"].unique().tolist()), key=f"{block_key}_well")
    wj = sj[sj["WELL_NAME"].astype(str) == well].copy().sort_values("_AXIS")
    wb = sb[sb["WELL_NAME"].astype(str) == well].copy().sort_values("_AXIS")
    if wj.empty or wb.empty:
        return
    tj = wj.groupby("_AXIS", as_index=False).agg(Кнг_hist=("Кнг_hist", "mean"), Кнг_model=("Кнг_model", "mean"))
    tb = wb.groupby("_AXIS", as_index=False).agg(Кнг_hist=("Кнг_hist", "mean"), Кнг_model=("Кнг_model", "mean"))
    x_lo = max(float(tj["_AXIS"].min()), float(tb["_AXIS"].min()))
    x_hi = min(float(tj["_AXIS"].max()), float(tb["_AXIS"].max()))
    if x_hi <= x_lo:
        grid = np.array(sorted(set(tj["_AXIS"].tolist() + tb["_AXIS"].tolist())), dtype=float)
    else:
        grid = np.linspace(x_lo, x_hi, int(max(40, min(250, 2 * min(len(tj), len(tb))))))
    hj = np.interp(grid, tj["_AXIS"].to_numpy(dtype=float), tj["Кнг_hist"].to_numpy(dtype=float))
    hb = np.interp(grid, tb["_AXIS"].to_numpy(dtype=float), tb["Кнг_hist"].to_numpy(dtype=float))
    hist = 0.5 * (hj + hb)
    pj = np.interp(grid, tj["_AXIS"].to_numpy(dtype=float), tj["Кнг_model"].to_numpy(dtype=float))
    pb = np.interp(grid, tb["_AXIS"].to_numpy(dtype=float), tb["Кнг_model"].to_numpy(dtype=float))
    plot_df = pd.DataFrame({"_AXIS": grid, "Кн историческая": hist, "Кн J-функция": pj, "Кн Брукса-Кори": pb}).melt(
        id_vars="_AXIS", var_name="Кривая", value_name="Кн"
    )
    fig = px.line(plot_df, x="Кн", y="_AXIS", color="Кривая", title=f"Скважина {well}: сравнение профилей")
    fig.update_traces(hovertemplate="Кн=%{x:.3f}<br>Глубина/индекс=%{y:.3f}<br>Кривая=%{fullData.name}<extra></extra>")
    if (wj["AXIS_KIND"].iloc[0] == "depth") and (wb["AXIS_KIND"].iloc[0] == "depth"):
        fig.update_yaxes(autorange="reversed", title="Глубина")
    else:
        fig.update_yaxes(title="Индекс точки")
    st.plotly_chart(fig, use_container_width=True)


def _well_pvts(df: pd.DataFrame) -> list[int]:
    """Уникальные PVTNUM_GDM только из текущей таблицы скважин (после очистки)."""
    if "PVTNUM_GDM" not in df.columns:
        return []
    s = pd.to_numeric(df["PVTNUM_GDM"], errors="coerce")
    s = s[s.notna()]
    if s.empty:
        return []
    return sorted(s.astype(int).unique().tolist())


def _pvt_summary_table(df: pd.DataFrame, pvts: list[int]) -> pd.DataFrame:
    if not pvts or "PVTNUM_GDM" not in df.columns:
        return pd.DataFrame()
    pvt_s = pd.to_numeric(df["PVTNUM_GDM"], errors="coerce").astype("Int64")
    rows = []
    for pvt in pvts:
        mask = pvt_s == pvt
        rows.append(
            {
                "PVTNUM": pvt,
                "Строк": int(mask.sum()),
                "Скважин": int(df.loc[mask, "WELL_NAME"].nunique()) if "WELL_NAME" in df.columns else 0,
            }
        )
    return pd.DataFrame(rows)


def _sync_pvt_session_state(wells_path: str, pvts: list[int]) -> None:
    """Сбросить привязки лаборатории/границ к регионам, которых нет в текущем файле скважин."""
    pvt_set = set(int(p) for p in pvts)
    sig = (str(wells_path), tuple(pvts))
    if st.session_state.get("_wells_pvt_sig") == sig:
        return
    st.session_state["_wells_pvt_sig"] = sig
    _clear_pvt_horizon_multiselects()
    for key in ("auto_bounds_by_pvt", "auto_preview_by_pvt"):
        stored = st.session_state.get(key)
        if isinstance(stored, dict):
            st.session_state[key] = {int(k): v for k, v in stored.items() if int(k) in pvt_set}


def _default_bounds_for_pvts(pvts: list[int]) -> dict[int, dict[str, tuple[float, float]]]:
    return {pvt: {"a": (0.05, 0.30), "b": (-3.0, -0.5), "sigma": (25.0, 35.0)} for pvt in pvts}


def _bounds_ui_manual(pvts: list[int], *, df_wells: pd.DataFrame | None = None) -> dict[int, dict[str, tuple[float, float]]]:
    st.subheader("Ограничения коэффициентов по регионам (PVTNUM)")
    if not pvts:
        st.warning("В данных скважин не найдено ни одного региона PVTNUM_GDM.")
        return {}
    if df_wells is not None:
        st.caption(
            f"Регионы берутся **только** из загруженного файла скважин ({len(pvts)}): "
            + ", ".join(str(p) for p in pvts)
        )
        summary = _pvt_summary_table(df_wells, pvts)
        if not summary.empty:
            st.dataframe(summary, use_container_width=True, hide_index=True)
    st.caption("Для каждого региона задайте диапазоны a, b и sigma.")
    bounds = _default_bounds_for_pvts(pvts)
    for pvt in pvts:
        with st.expander(f"PVTNUM {pvt}", expanded=False):
            c1, c2 = st.columns(2)
            with c1:
                a_min = st.number_input(f"a min | PVT {pvt}", value=bounds[pvt]["a"][0], key=f"a_min_{pvt}")
                b_min = st.number_input(f"b min | PVT {pvt}", value=bounds[pvt]["b"][0], key=f"b_min_{pvt}")
                s_min = st.number_input(f"sigma min | PVT {pvt}", value=bounds[pvt]["sigma"][0], key=f"s_min_{pvt}")
            with c2:
                a_max = st.number_input(f"a max | PVT {pvt}", value=bounds[pvt]["a"][1], key=f"a_max_{pvt}")
                b_max = st.number_input(f"b max | PVT {pvt}", value=bounds[pvt]["b"][1], key=f"b_max_{pvt}")
                s_max = st.number_input(f"sigma max | PVT {pvt}", value=bounds[pvt]["sigma"][1], key=f"s_max_{pvt}")
            if a_min >= a_max or b_min >= b_max or s_min >= s_max:
                st.error("Минимум должен быть меньше максимума.")
            bounds[pvt] = {"a": (a_min, a_max), "b": (b_min, b_max), "sigma": (s_min, s_max)}
    return bounds


def _manual_params_ui(pvts: list[int]) -> dict[int, tuple[float, float, float]]:
    st.subheader("Заданные коэффициенты a, b, sigma (без оптимизации)")
    out: dict[int, tuple[float, float, float]] = {}
    for pvt in pvts:
        with st.expander(f"PVTNUM {pvt}", expanded=False):
            a = st.number_input(f"a | PVT {pvt}", value=0.15, format="%.6f", key=f"fix_a_{pvt}")
            b = st.number_input(f"b | PVT {pvt}", value=-1.0, format="%.6f", key=f"fix_b_{pvt}")
            s = st.number_input(f"sigma | PVT {pvt}", value=30.0, format="%.4f", key=f"fix_s_{pvt}")
            out[pvt] = (float(a), float(b), float(s))
    return out


def _validate_columns(df_wells: pd.DataFrame, df_prod: pd.DataFrame | None) -> list[str]:
    errors: list[str] = []
    required = {"WELL_NAME", "PVTNUM_GDM", "PORO_GDM", "PERM_GDM", "PC", "SWL_GDM", "Кнг_W"}
    missing = sorted(required - set(df_wells.columns))
    if missing:
        errors.append(f"В файле скважин не хватает колонок: {missing}")
    if df_prod is not None and not df_prod.empty:
        prod_ok = any(col in df_prod.columns for col in ["Ствол скважины", "WELL_NAME"]) and any(
            col in df_prod.columns for col in ["Экспл. объект", "PVTNUM_GDM"]
        )
        if not prod_ok:
            errors.append("Файл добычи должен содержать колонки скважины и региона (PVTNUM).")
    return errors


def _coalesce_path(*keys: str) -> str | None:
    for k in keys:
        v = st.session_state.get(k)
        if isinstance(v, str) and v.strip():
            return v
    return None


def _scroll_page_top() -> None:
    """Прокрутка в начало страницы (с повторами после отрисовки контента Streamlit)."""
    components.html(
        """
        <script>
        (function() {
          const doc = window.parent.document;
          const win = window.parent;
          if (!doc) return;

          if (doc.activeElement && typeof doc.activeElement.blur === 'function') {
            doc.activeElement.blur();
          }

          function scrollAllToTop() {
            const seen = new Set();
            const candidates = [
              doc.scrollingElement,
              doc.documentElement,
              doc.body,
              doc.querySelector('section.main'),
              doc.querySelector('[data-testid="stAppViewContainer"]'),
              doc.querySelector('[data-testid="stMain"]'),
              doc.querySelector('[data-testid="stMainBlockContainer"]'),
              doc.querySelector('.main'),
              doc.querySelector('.block-container'),
            ].filter(Boolean);

            candidates.forEach((el) => {
              if (seen.has(el)) return;
              seen.add(el);
              try {
                el.scrollTop = 0;
                if (typeof el.scrollTo === 'function') {
                  el.scrollTo({ top: 0, left: 0, behavior: 'auto' });
                }
              } catch (e) {}
            });

            try {
              if (win && typeof win.scrollTo === 'function') {
                win.scrollTo(0, 0);
              }
            } catch (e) {}
          }

          scrollAllToTop();
          [0, 50, 120, 250, 450, 700].forEach((ms) => setTimeout(scrollAllToTop, ms));
        })();
        </script>
        """,
        height=0,
        width=0,
    )


SCROLL_STORAGE_BC = "scroll_bc_main"


def _mark_scroll_to_top_pending() -> None:
    st.session_state["_scroll_to_top_pending"] = True


def _scroll_to_top_if_pending(*, finish: bool = False) -> None:
    """Прокрутка вверх после смены раздела; finish=True — в конце вкладки (после отрисовки)."""
    if not st.session_state.get("_scroll_to_top_pending"):
        return
    _scroll_page_top()
    if finish:
        st.session_state.pop("_scroll_to_top_pending", None)


def _clear_preserved_scroll(*storage_keys: str) -> None:
    keys_js = ", ".join(repr(k) for k in storage_keys)
    components.html(
        f"""
        <script>
        (function() {{
          const keys = [{keys_js}];
          keys.forEach((k) => {{
            try {{ sessionStorage.removeItem(k); }} catch (e) {{}}
          }});
        }})();
        </script>
        """,
        height=0,
        width=0,
    )


def _preserve_scroll_position(storage_key: str = SCROLL_STORAGE_BC, *, restore: bool = True) -> None:
    """Сохраняет прокрутку; восстановление только внутри той же вкладки (restore=True)."""
    restore_js = "true" if restore else "false"
    components.html(
        f"""
        <script>
        (function() {{
          const doc = window.parent.document;
          if (!doc) return;
          const KEY = '{storage_key}';
          const doRestore = {restore_js};
          function scrollEl() {{
            return doc.querySelector('[data-testid="stAppViewContainer"]')
                || doc.querySelector('section.main')
                || doc.scrollingElement
                || doc.documentElement;
          }}
          const el = scrollEl();
          if (!el) return;

          if (doRestore) {{
            const saved = sessionStorage.getItem(KEY);
            if (saved !== null && saved !== '') {{
              const y = parseInt(saved, 10);
              if (!Number.isNaN(y)) {{
                const apply = () => {{ try {{ el.scrollTop = y; }} catch (e) {{}} }};
                apply();
                requestAnimationFrame(apply);
                setTimeout(apply, 0);
                setTimeout(apply, 80);
                setTimeout(apply, 200);
              }}
            }}
          }} else {{
            try {{ sessionStorage.setItem(KEY, '0'); }} catch (e) {{}}
          }}

          const flag = '__scroll_listener_' + KEY;
          if (!window[flag]) {{
            window[flag] = true;
            el.addEventListener('scroll', () => {{
              sessionStorage.setItem(KEY, String(el.scrollTop));
            }}, {{ passive: true }});
          }}
        }})();
        </script>
        """,
        height=0,
        width=0,
    )


def _ui_lock(is_locked: bool, lock_id: str) -> None:
    components.html(
        f"""
        <script>
        (function() {{
          const doc = window.parent.document;
          if (!doc) return;
          const lockId = '{lock_id}';
          const locked = {str(is_locked).lower()};
          const prev = doc.getElementById(lockId);
          if (!locked) {{
            if (prev) prev.remove();
            doc.querySelectorAll('[data-ui-lock-disabled=\"1\"]').forEach((el) => {{
              try {{
                el.disabled = false;
                el.removeAttribute('data-ui-lock-disabled');
              }} catch (e) {{}}
            }});
            return;
          }}
          if (!prev) {{
            const ov = doc.createElement('div');
            ov.id = lockId;
            ov.style.position = 'fixed';
            ov.style.inset = '0';
            ov.style.background = 'rgba(255,255,255,0.06)';
            ov.style.zIndex = '2147483000';
            ov.style.cursor = 'wait';
            ov.style.pointerEvents = 'auto';
            doc.body.appendChild(ov);
          }}
          doc.querySelectorAll('button, input, select, textarea').forEach((el) => {{
            try {{
              if (!el.disabled) {{
                el.disabled = true;
                el.setAttribute('data-ui-lock-disabled', '1');
              }}
            }} catch (e) {{}}
          }});
        }})();
        </script>
        """,
        height=0,
        width=0,
    )


def _safe_guess_col(cols: list[str], candidates: list[str]) -> str:
    if not cols:
        return ""
    try:
        return _guess_col(cols, candidates)
    except Exception:
        return cols[0]


def _csv_bytes(df: pd.DataFrame) -> bytes:
    out = df.copy()
    num_cols = out.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 0:
        out[num_cols] = out[num_cols].round(3)
    return out.to_csv(index=False).encode("utf-8-sig")


def _round_df(df: pd.DataFrame, digits: int = 3) -> pd.DataFrame:
    out = df.copy()
    num_cols = out.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 0:
        out[num_cols] = out[num_cols].round(digits)
    return out


def _fmt_float3(x: object) -> str:
    try:
        xf = float(x)  # type: ignore[arg-type]
        if not np.isfinite(xf):
            return "—"
        return f"{xf:.3f}"
    except (TypeError, ValueError):
        return "—"


# Общее сопоставление колонок скважин/добычи между вкладками J и БК
SHARED_WELLS_COL_SIG = "shared_wells_cols_sig"
SHARED_WELLS_MAP_SAVED = "shared_wells_map_saved"
SHARED_PROD_COL_SIG = "shared_prod_cols_sig"
SHARED_PROD_MAP_SAVED = "shared_prod_map_saved"
SHARED_WELL_PREVIEW = "shared_well_preview"


def _wells_file_uploader_key() -> str:
    n = int(st.session_state.get("shared_wells_uploader_nonce", 0))
    return f"shared_wells_file_{n}"


def _prod_file_uploader_key() -> str:
    n = int(st.session_state.get("shared_prod_uploader_nonce", 0))
    return f"shared_prod_file_{n}"


def _clear_leverett_results() -> None:
    for k in (
        "leverett_result_df",
        "leverett_params_df",
        "leverett_qa_df",
        "j_timing_df",
        "j_total_elapsed_sec",
    ):
        st.session_state.pop(k, None)


def _clear_methods_tabs_results() -> None:
    """Сброс таблиц и графиков расчёта на вкладках J и Брукса-Кори."""
    _clear_leverett_results()
    for k in (
        "bc_result_df",
        "bc_params_df",
        "bc_qa_df",
        "bc_timing_df",
        "bc_total_elapsed_sec",
        "bc_meta",
        "bc_busy",
        "bc_manual_preview_on",
    ):
        st.session_state.pop(k, None)
    st.session_state.pop("j_busy", None)


def _shared_well_selectbox(wells: list[str], label: str = "Скважина") -> str:
    """Общий выбор скважины для профиля на вкладках J и Брукса-Кори."""
    if not wells:
        return ""
    saved = st.session_state.get(SHARED_WELL_PREVIEW)
    if saved not in wells:
        st.session_state[SHARED_WELL_PREVIEW] = wells[0]
    return st.selectbox(label, wells, key=SHARED_WELL_PREVIEW)


def _clear_pvt_horizon_multiselects() -> None:
    st.session_state.pop("pvt_horizon_map", None)
    st.session_state.pop("bc_horizon_map_ctx", None)
    st.session_state.pop("_pending_pvt_hor_fill", None)
    for k in list(st.session_state.keys()):
        if str(k).startswith("pvt_hor_map_"):
            st.session_state.pop(k, None)


def _pvt_horizon_map_from_session(pvts: list[int]) -> dict[int, list[str]]:
    """Привязки горизонт→PVTNUM из session_state (как на вкладке J)."""
    stored = st.session_state.get("pvt_horizon_map") or {}
    out: dict[int, list[str]] = {}
    for p in pvts:
        pi = int(p)
        sel = stored.get(pi, stored.get(p))
        if not sel:
            sel = st.session_state.get(f"pvt_hor_map_{pi}_bc") or st.session_state.get(f"pvt_hor_map_{pi}", [])
        out[pi] = [str(h) for h in (sel or [])]
    return out


def _lab_j_horizon_universe() -> list[str]:
    """Коды горизонта из сохранённого лабораторного облака J (после «Применить фильтр»)."""
    meta = st.session_state.get("lab_meta") or {}
    if meta.get("horizons"):
        return sorted(str(h) for h in meta["horizons"])
    cloud = st.session_state.get("lab_cloud")
    if isinstance(cloud, pd.DataFrame) and not cloud.empty and "lab_horizon" in cloud.columns:
        return sorted(cloud["lab_horizon"].astype(str).unique().tolist())
    return []


def _render_pvt_horizon_mapping(
    pvts: list[int],
    hors_universe: list[str],
    *,
    title: str,
    key_suffix: str = "",
) -> dict[int, list[str]]:
    """Мультивыбор горизонтов по PVTNUM; сохраняет pvt_horizon_map в session_state."""
    if not pvts or not hors_universe:
        return {}
    stored = st.session_state.get("pvt_horizon_map") or {}
    st.subheader(title)
    pvt_horizon_map: dict[int, list[str]] = {}
    for pvt in pvts:
        pi = int(pvt)
        prev = stored.get(pi, stored.get(pvt, []))
        default = [str(h) for h in (prev or []) if str(h) in hors_universe]
        pvt_horizon_map[pi] = st.multiselect(
            f"Горизонты для PVTNUM {pi}",
            options=hors_universe,
            default=default,
            key=f"pvt_hor_map_{pi}{key_suffix}",
        )
    st.session_state["pvt_horizon_map"] = pvt_horizon_map
    return pvt_horizon_map


def _bc_pvt_horizon_mapping_fragment() -> None:
    ctx = st.session_state.get("bc_horizon_map_ctx") or {}
    pvts = [int(p) for p in (ctx.get("pvts") or [])]
    hors_universe = [str(h) for h in (ctx.get("hors_universe") or [])]
    if not pvts or not hors_universe:
        return
    _render_pvt_horizon_mapping(
        pvts,
        hors_universe,
        title="Связь горизонт -> PVTNUM (Брукса-Кори)",
        key_suffix="_bc",
    )


_bc_pvt_horizon_mapping_run = (
    st.fragment(_bc_pvt_horizon_mapping_fragment)
    if hasattr(st, "fragment")
    else _bc_pvt_horizon_mapping_fragment
)


def _bc_well_preview_fragment() -> None:
    """Профиль скважины — отдельный fragment, без перерисовки всей вкладки БК."""
    bc_res = st.session_state.get("bc_result_df")
    if not isinstance(bc_res, pd.DataFrame) or bc_res.empty:
        return
    psel = st.session_state.get("bc_plot_pvt")
    if psel is None:
        pvt_opts = sorted(pd.to_numeric(bc_res["PVTNUM_GDM"], errors="coerce").dropna().astype(int).unique().tolist())
        if not pvt_opts:
            return
        psel = pvt_opts[0]
    g = bc_res[pd.to_numeric(bc_res["PVTNUM_GDM"], errors="coerce") == float(psel)].copy()
    g = g.dropna(subset=["Кнг_W", "Kng_BC_model"])
    g_conv = _filter_convergence_points(g.rename(columns={"Kng_BC_model": "Kng_model"})).rename(
        columns={"Kng_model": "Kng_BC_model"}
    )
    if g_conv.empty or "WELL_NAME" not in g_conv.columns:
        return
    st.subheader("Просмотр скважины (Брукса-Кори)")
    wells = sorted(g_conv["WELL_NAME"].astype(str).unique().tolist())
    well = _shared_well_selectbox(wells)
    wd = g_conv[g_conv["WELL_NAME"].astype(str) == well].copy()
    dcol = _pick_depth_column(wd)
    if dcol is None:
        st.warning("Не найдена колонка глубины для скважины.")
    elif "ACTNUM_GDM" not in wd.columns:
        st.warning("В данных отсутствует ACTNUM_GDM.")
    else:
        wd[dcol] = pd.to_numeric(wd[dcol], errors="coerce")
        wd = wd.dropna(subset=[dcol]).sort_values(dcol).reset_index(drop=True)
        curve = wd[[dcol, "ACTNUM_GDM", "Кнг_W", "Kng_BC_model"]].rename(
            columns={"Кнг_W": "Кн РИГИС", "Kng_BC_model": "Кн Брукса-Кори"}
        )
        melt = curve.melt(
            id_vars=[dcol],
            value_vars=["ACTNUM_GDM", "Кн РИГИС", "Кн Брукса-Кори"],
            var_name="Кривая",
            value_name="Значение",
        )
        fig_prof = px.line(melt, x="Значение", y=dcol, color="Кривая", title=f"Скважина {well}: вертикальный профиль (БК)")
        fig_prof.update_traces(hovertemplate="Значение=%{x:.3f}<br>Глубина=%{y:.3f}<br>Кривая=%{fullData.name}<extra></extra>")
        fig_prof.update_yaxes(autorange="reversed")
        st.plotly_chart(fig_prof, use_container_width=True)


def _bc_results_dashboard_fragment() -> None:
    """Таблицы и графики результатов БК (без блока просмотра скважины)."""
    bc_res = st.session_state.get("bc_result_df")
    bc_params = st.session_state.get("bc_params_df")
    bc_qa = st.session_state.get("bc_qa_df")
    bc_timing = st.session_state.get("bc_timing_df")
    bc_total_elapsed = st.session_state.get("bc_total_elapsed_sec")
    if not isinstance(bc_res, pd.DataFrame) or not isinstance(bc_params, pd.DataFrame) or bc_res.empty:
        return

    cpar, cqa = st.columns(2)
    with cpar:
        st.subheader("Параметры Брукса-Кори по регионам")
        st.dataframe(_round_df(bc_params), use_container_width=True)
        if isinstance(bc_timing, pd.DataFrame) and not bc_timing.empty:
            st.subheader("Статистика времени расчета БК")
            bc_timing_ru = bc_timing.rename(
                columns={
                    "PVTNUM_GDM": "Регион (PVTNUM)",
                    "rows_geo": "Строк геологии",
                    "rows_lab": "Лабораторных точек",
                    "elapsed_sec": "Время расчета, сек",
                }
            )
            bc_timing_ru["Регион (PVTNUM)"] = bc_timing_ru["Регион (PVTNUM)"].astype(str)
            st.dataframe(
                _round_df(bc_timing_ru.set_index("Регион (PVTNUM)")),
                use_container_width=True,
                height=min(600, 35 * (len(bc_timing_ru) + 1)),
            )
            if bc_total_elapsed is not None:
                st.caption(f"Общее время расчета БК: {float(bc_total_elapsed):.2f} с")
    with cqa:
        st.subheader("Метрики качества Брукса-Кори")
        if isinstance(bc_qa, pd.DataFrame) and not bc_qa.empty:
            bc_qa_show = bc_qa.copy()
            bc_qa_show["PVTNUM_GDM"] = bc_qa_show["PVTNUM_GDM"].astype(str)
            st.dataframe(_round_df(bc_qa_show.set_index("PVTNUM_GDM")), use_container_width=True)
            global_bc = bc_qa[bc_qa["PVTNUM_GDM"].astype(str) == "Все регионы"]
            if not global_bc.empty and np.isfinite(global_bc.iloc[0]["SCORE"]):
                st.metric("SCORE (все регионы)", f"{float(global_bc.iloc[0]['SCORE']):.3f}")
            else:
                reg_scores = bc_qa.loc[bc_qa["PVTNUM_GDM"].astype(str) != "Все регионы", "SCORE"]
                if len(reg_scores):
                    st.metric("SCORE (среднее по регионам)", f"{float(reg_scores.mean()):.3f}")
        else:
            st.info("Метрики пока недоступны.")
    csv_bc_result = bc_res.to_csv(index=False).encode("utf-8")
    csv_bc_params = bc_params.to_csv(index=False).encode("utf-8")
    csv_bc_qa = (bc_qa if isinstance(bc_qa, pd.DataFrame) else pd.DataFrame()).to_csv(index=False).encode("utf-8")
    cd1, cd2, cd3 = st.columns(3)
    cd1.download_button("Скачать результат", data=csv_bc_result, file_name="brooks_corey_result.csv", mime="text/csv")
    cd2.download_button("Скачать параметры", data=csv_bc_params, file_name="brooks_corey_params.csv", mime="text/csv")
    cd3.download_button("Скачать метрики", data=csv_bc_qa, file_name="brooks_corey_metrics.csv", mime="text/csv")
    if st.button("Запомнить результаты по скважинам (Брукса-Кори)", key="save_bc_snapshot"):
        ok, msg = _save_well_snapshot(bc_res, model_col="Kng_BC_model", method_tag="BC")
        if ok:
            st.success(msg)
        else:
            st.warning(msg)

    pvt_opts = sorted(pd.to_numeric(bc_res["PVTNUM_GDM"], errors="coerce").dropna().astype(int).unique().tolist())
    psel = st.selectbox("Регион для графиков БК", pvt_opts, key="bc_plot_pvt")
    g = bc_res[pd.to_numeric(bc_res["PVTNUM_GDM"], errors="coerce") == float(psel)].copy()
    g = g.dropna(subset=["Кнг_W", "Kng_BC_model"])
    if g.empty:
        return

    bc_meta = st.session_state.get("bc_meta", {})
    m = bc_meta.get(psel, {})
    lab_pvt = m.get("lab", pd.DataFrame()).copy()
    if not lab_pvt.empty:
        st.subheader("Оптимальные зависимости Брукса-Кори по облакам")
        horizons_for_plot = sorted(lab_pvt["HORIZON"].astype(str).unique().tolist())
        selected_h = st.multiselect(
            "Горизонты для отображения зависимостей",
            options=horizons_for_plot,
            default=horizons_for_plot,
            key="bc_h_plot",
        )
        lab_plot = lab_pvt[lab_pvt["HORIZON"].astype(str).isin(selected_h)] if selected_h else lab_pvt
        prow = bc_params[bc_params["PVTNUM_GDM"].astype(int) == int(psel)]
        if not prow.empty and not lab_plot.empty:
            pp = prow.iloc[0]
            env = m.get("envelopes", {})
            fig1 = _plot_bc_cloud(
                lab_plot,
                "PORO_LAB_FRAC",
                "SWL_LAB",
                f"PVT {psel}: swl = a·exp(b·Кп)",
                lower_ab=env.get("swl", {}).get("lower"),
                upper_ab=env.get("swl", {}).get("upper"),
                opt_ab=(float(pp["a_swl"]), float(pp["b_swl"])),
                curve_kind="swl_exp_ab",
            )
            fig2 = _plot_bc_cloud(
                lab_plot,
                "SWL_LAB",
                "PERM_LAB",
                f"PVT {psel}: Кпр(Кво)",
                lower_ab=env.get("perm", {}).get("lower"),
                upper_ab=env.get("perm", {}).get("upper"),
                opt_ab=(float(pp["a_perm"]), float(pp["b_perm"])),
            )
            fig3 = _plot_bc_cloud(
                lab_plot,
                "perm_poro",
                "PVIT_LAB",
                f"PVT {psel}: pvit(perm_poro)",
                lower_ab=env.get("pvit", {}).get("lower"),
                upper_ab=env.get("pvit", {}).get("upper"),
                opt_ab=(float(pp["a_pvit"]), float(pp["b_pvit"])),
            )
            fig4 = _plot_bc_cloud(
                lab_plot,
                "perm_poro",
                "N_LAB",
                f"PVT {psel}: n(perm_poro)",
                lower_ab=env.get("n", {}).get("lower"),
                upper_ab=env.get("n", {}).get("upper"),
                opt_ab=(float(pp["a_n"]), float(pp["b_n"])),
            )
            c1, c2 = st.columns(2)
            c1.plotly_chart(fig1, use_container_width=True)
            c2.plotly_chart(fig2, use_container_width=True)
            c3, c4 = st.columns(2)
            c3.plotly_chart(fig3, use_container_width=True)
            c4.plotly_chart(fig4, use_container_width=True)

    st.subheader("Кроссплоты Брукса-Кори")
    g_conv = _filter_convergence_points(g.rename(columns={"Kng_BC_model": "Kng_model"})).rename(
        columns={"Kng_model": "Kng_BC_model"}
    )
    if g_conv.empty:
        st.warning("Нет валидных точек сходимости для кроссплотов БК.")
        return
    bc_color_mode = st.radio("Палитра БК", options=("По весу", "По толщине"), horizontal=True, key="bc_scatter_color")
    depth_col = _pick_depth_column(g_conv)
    if bc_color_mode == "По толщине" and depth_col is not None and "WELL_NAME" in g_conv.columns:
        g_conv[depth_col] = pd.to_numeric(g_conv[depth_col], errors="coerce")
        tdf = g_conv.dropna(subset=[depth_col]).groupby("WELL_NAME")[depth_col].agg(["min", "max"]).reset_index()
        tdf["thickness"] = tdf["max"] - tdf["min"]
        g_conv = g_conv.merge(tdf[["WELL_NAME", "thickness"]], on="WELL_NAME", how="left")
        bc_color = "thickness"
    else:
        bc_color = "weight" if "weight" in g_conv.columns else None

    fig = px.scatter(
        g_conv,
        x="Кнг_W",
        y="Kng_BC_model",
        color=bc_color,
        color_continuous_scale="Viridis",
        hover_data={
            c: ":.3f"
            for c in ["PC", "PORO_GDM", "Kng_BC_model", "Кнг_W", "thickness", "weight"]
            if c in g_conv.columns
        },
        title=f"PVT {psel}: предсказанное(историческое) (Брукса-Кори)",
        opacity=0.75,
        **_crossplot_hover_name_kw(g_conv),
    )
    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(color="red", dash="dash"))
    _apply_crossplot_hover(fig, "Кнг_W=%{x:.3f}<br>Kng_BC_model=%{y:.3f}", g_conv)
    st.plotly_chart(fig, use_container_width=True)

    if "WELL_NAME" in g_conv.columns:
        cross = _well_weighted_crossplot_df(g_conv.rename(columns={"Kng_BC_model": "Kng_model"})).rename(
            columns={"Kng_model_wmean": "Kng_BC_wmean"}
        )
        if not cross.empty:
            figw = px.scatter(
                cross,
                x="Кнг_W_wmean",
                y="Kng_BC_wmean",
                color="convergence_percent",
                color_continuous_scale="Turbo",
                hover_data={
                    "points": ":.3f",
                    "avg_weight": ":.3f",
                    "convergence_percent": ":.3f",
                },
                title=f"PVT {psel}: кроссплот по скважинам (БК, средневзвешенно)",
                **_crossplot_hover_name_kw(cross),
            )
            _apply_crossplot_hover(figw, "Кнг_W_wmean=%{x:.3f}<br>Kng_BC_wmean=%{y:.3f}", cross)
            figw.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(color="red", dash="dash"))
            st.plotly_chart(figw, use_container_width=True)
            st.markdown("#### Невязка по кроссплоту (скважины)")
            cross_all_bc = _well_crossplot_table_from_result(bc_res, "Кнг_W", "Kng_BC_model")
            _render_well_crossplot_qa_panel(
                cross_all_bc,
                x_col="Кнг_W_wmean",
                y_col="Kng_model_wmean",
                region_col="Регион",
                block_key="bc_all_pvt",
                method_label="Брукса — Кори",
                show_metrics_help=True,
            )


_bc_well_preview_run = st.fragment(_bc_well_preview_fragment) if hasattr(st, "fragment") else _bc_well_preview_fragment
_bc_results_dashboard_run = (
    st.fragment(_bc_results_dashboard_fragment) if hasattr(st, "fragment") else _bc_results_dashboard_fragment
)


def _clear_lab_j_session() -> None:
    """Сброс сохранённого облака J(Swn), фильтров лаборатории и результатов J/БК."""
    for k in (
        "lab_cloud",
        "lab_cloud_ready",
        "lab_meta",
        "lab_trend_fit",
        "lab_source_df",
        "_pending_pvt_hor_fill",
        "_lab_matrix_upload_id",
        "_lab_kkd_upload_id",
        "_bc_lab_upload_id",
    ):
        st.session_state.pop(k, None)
    for k in ("lab_sel_areas", "lab_sel_hors", "_lab_areas_sig"):
        st.session_state.pop(k, None)
    _clear_methods_tabs_results()


def _lab_selectbox_index(cols: list[str], saved: str | None, guess: str | None) -> int:
    if saved and saved in cols:
        return cols.index(saved)
    if guess and guess in cols:
        return cols.index(guess)
    return 0


def _restore_lab_filter_widgets(areas: list[str], horizons: list[str]) -> None:
    """Восстановить мультивыбор площадей/горизонтов после «Применить фильтр»."""
    meta = st.session_state.get("lab_meta") or {}
    if not st.session_state.get("lab_cloud_ready"):
        return
    saved_a = meta.get("areas") or []
    saved_h = meta.get("horizons") or []
    st.session_state["lab_sel_areas"] = [a for a in saved_a if a in areas]
    st.session_state["lab_sel_hors"] = [h for h in saved_h if h in horizons]


def _clear_shared_wells_files() -> None:
    for k in ("wells_file_path", "shared_wells_file_path", "bc_wells_path", "_active_wells_path"):
        st.session_state.pop(k, None)
    st.session_state["shared_wells_uploader_nonce"] = int(st.session_state.get("shared_wells_uploader_nonce", 0)) + 1
    st.session_state.pop(SHARED_WELLS_COL_SIG, None)
    st.session_state.pop("_wells_pvt_sig", None)
    st.session_state.pop("auto_bounds_by_pvt", None)
    st.session_state.pop("auto_preview_by_pvt", None)
    _clear_methods_tabs_results()


def _clear_shared_prod_files() -> None:
    for k in ("prod_file_path", "shared_prod_file_path", "bc_prod_path"):
        st.session_state.pop(k, None)
    st.session_state["shared_prod_uploader_nonce"] = int(st.session_state.get("shared_prod_uploader_nonce", 0)) + 1
    st.session_state.pop(SHARED_PROD_COL_SIG, None)


def _map_uploaded_wells_df(raw_wells: pd.DataFrame, *, key_prefix: str, title: str) -> pd.DataFrame:
    df = _normalize_columns(raw_wells.copy())
    cols = list(df.columns)
    st.subheader(title)
    st.caption("Сопоставления подставляются автоматически по именам колонок; при необходимости скорректируйте вручную.")
    if not cols:
        return df
    sig = tuple(sorted(cols))
    prev_sig = st.session_state.get(SHARED_WELLS_COL_SIG)
    if prev_sig != sig:
        st.session_state[SHARED_WELLS_COL_SIG] = sig
        st.session_state[SHARED_WELLS_MAP_SAVED] = {}
        st.session_state.pop("_wells_pvt_sig", None)
        st.session_state.pop("shared_map_wells_Perf_GDM", None)
    saved_map = st.session_state.get(SHARED_WELLS_MAP_SAVED) or st.session_state.get(f"{key_prefix}_wells_map_saved", {})
    mapping_rules: list[tuple[str, str, list[str]]] = [
        ("WELL_NAME", "Скважина", ["WELL_NAME", "СКВАЖ", "WELL", "STVOL"]),
        ("PVTNUM_GDM", "Регион", ["PVTNUM", "PVT", "ЭКСПЛ", "OBJECT", "OBJ"]),
        ("PORO_GDM", "Пористость", ["PORO", "ПОР", "PHI"]),
        ("PERM_GDM", "Проницаемость", ["PERM", "ПРОНИ", "KPR", "K_PR"]),
        ("PC", "Капиллярное давление", ["PC", "КАПИЛ", "P_CAP"]),
        ("SWL_GDM", "Кво", ["SWL", "SWI", "SW_MIN", "КВО"]),
        ("Кнг_W", "Нефтенасыщенность", ["КНГ", "KNG", "KN", "SOIL", "НЕФТЕНАС", "RIGIS"]),
    ]
    defaults = {}
    for target, _, hints in mapping_rules:
        cand = saved_map.get(target)
        if isinstance(cand, str) and cand in cols:
            defaults[target] = cand
        elif target == "Кнг_W":
            gk = _guess_kng_w_column(cols)
            defaults[target] = gk if gk else _safe_guess_col(cols, hints)
        else:
            defaults[target] = _safe_guess_col(cols, hints)
    pick: dict[str, str] = {}
    row1_rules = mapping_rules[:4]
    row2_rules = mapping_rules[4:]

    row1 = st.columns(4)
    for i, (target, ui_label, _) in enumerate(row1_rules):
        default = defaults[target]
        idx = cols.index(default) if default in cols else 0
        pick[target] = row1[i].selectbox(
            ui_label,
            options=cols,
            index=idx,
            key=f"shared_map_wells_{target}",
        )

    row2 = st.columns(4)
    for i, (target, ui_label, _) in enumerate(row2_rules):
        default = defaults[target]
        idx = cols.index(default) if default in cols else 0
        pick[target] = row2[i].selectbox(
            ui_label,
            options=cols,
            index=idx,
            key=f"shared_map_wells_{target}",
        )

    perf_not_used = "не использовать"
    if st.session_state.get("shared_map_wells_Perf_GDM") == "<не использовать>":
        st.session_state.pop("shared_map_wells_Perf_GDM", None)
    perf_saved = saved_map.get("Perf_GDM")
    perf_options = [perf_not_used] + cols
    if isinstance(perf_saved, str) and perf_saved in cols:
        perf_idx = perf_options.index(perf_saved)
    else:
        perf_idx = 0
    perf_pick = row2[3].selectbox("Перфорация", options=perf_options, index=perf_idx, key="shared_map_wells_Perf_GDM")
    if perf_pick != perf_not_used:
        pick["Perf_GDM"] = perf_pick
    if len(set(pick.values())) != len(pick):
        st.error("Для обязательных полей скважин выбраны повторяющиеся колонки. Выберите уникальные соответствия.")
        return pd.DataFrame()
    st.session_state[SHARED_WELLS_MAP_SAVED] = pick.copy()
    rename_map = {src: dst for dst, src in pick.items()}
    mapped = df.rename(columns=rename_map)
    return mapped


def _map_uploaded_prod_df(raw_prod: pd.DataFrame | None, *, key_prefix: str, title: str) -> pd.DataFrame | None:
    if raw_prod is None or raw_prod.empty:
        return None
    df = _normalize_columns(raw_prod.copy())
    cols = list(df.columns)
    st.subheader(title)
    st.caption("Опционально: сопоставьте колонки файла добычи в таблице ниже, затем проверьте предпросмотр.")
    if not cols:
        return df
    sig = tuple(sorted(cols))
    prev_sig = st.session_state.get(SHARED_PROD_COL_SIG)
    if prev_sig != sig:
        st.session_state[SHARED_PROD_COL_SIG] = sig
        st.session_state[SHARED_PROD_MAP_SAVED] = {}
    saved_map = st.session_state.get(SHARED_PROD_MAP_SAVED) or st.session_state.get(f"{key_prefix}_prod_map_saved", {})
    rules: list[tuple[str, str, list[str]]] = [
        ("WELL_NAME", "Скважина", ["WELL_NAME", "СТВОЛ", "СКВАЖ", "WELL"]),
        ("PVTNUM_GDM", "Регион", ["PVTNUM", "ЭКСПЛ", "ОБЪЕКТ", "OBJECT"]),
    ]
    defaults = {}
    for target, _, hints in rules:
        cand = saved_map.get(target)
        defaults[target] = cand if isinstance(cand, str) and cand in cols else _safe_guess_col(cols, hints)
    picks: dict[str, str] = {}
    cols_ui = st.columns(2)
    for i, (target, ui_label, _) in enumerate(rules):
        host = cols_ui[i % 2]
        default = defaults[target]
        idx = cols.index(default) if default in cols else 0
        picks[target] = host.selectbox(ui_label, options=cols, index=idx, key=f"shared_map_prod_{target}")
    st.caption("Предпросмотр загруженного файла добычи:")
    st.dataframe(_round_df(df.head(12)), use_container_width=True)
    if len(set(picks.values())) != len(picks):
        st.error("Для файла добычи выбраны повторяющиеся колонки сопоставления.")
        return pd.DataFrame()
    st.session_state[SHARED_PROD_MAP_SAVED] = picks.copy()
    rename_map = {src: dst for dst, src in picks.items()}
    return df.rename(columns=rename_map)


def _prod_obj_key(v: object) -> str:
    """Стабильный ключ для кода эксплуатационного объекта в файле добычи."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "__NA__"
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        fv = float(v)
        if np.isfinite(fv) and abs(fv - round(fv)) < 1e-9:
            return str(int(round(fv)))
        return format(fv, "g")
    s = str(v).strip()
    return s if s else "__EMPTY__"


def _prod_code_resolves_to_model_pvt(v: object, model_pvt_set: set[int]) -> int | None:
    """Если код в добыче уже совпадает с целочисленным PVTNUM модели — вернуть его."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    try:
        iv = int(float(str(v).strip()))
        if iv in model_pvt_set:
            return iv
    except (ValueError, TypeError, OverflowError):
        pass
    return None


def _pvtmap_widget_key(prefix: str, prod_path_tag: str, rk: str) -> str:
    h = hashlib.md5(f"{prefix}|{prod_path_tag}|{rk}".encode("utf-8", errors="replace")).hexdigest()[:18]
    return f"pvtmap_{prefix}_{h}"


def _apply_prod_pvt_mapping_ui(
    df_prod: pd.DataFrame | None,
    model_pvts: list[int],
    *,
    key_prefix: str,
    prod_path: str | None,
    title: str = "Соответствие кода объекта в добыче → PVTNUM модели",
) -> pd.DataFrame | None:
    """
    В файле добычи колонка «регион» может содержать коды эксплуатационных объектов,
    не совпадающие с PVTNUM_GDM скважин. Здесь для каждого уникального кода задаётся PVTNUM,
    к которому относить строку при расчёте весов (слияние по WELL_NAME + PVTNUM_GDM).
    """
    if df_prod is None or df_prod.empty or not model_pvts:
        return df_prod
    if "PVTNUM_GDM" not in df_prod.columns:
        return df_prod
    tag = str(prod_path or "")
    wset = set(int(p) for p in model_pvts)
    uniq = df_prod["PVTNUM_GDM"].dropna().unique().tolist()
    uniq_sorted = sorted(uniq, key=lambda x: (_prod_obj_key(x), str(x)))
    st.subheader(title)
    st.caption(
        "Если в добыче указаны другие обозначения объектов, чем **PVTNUM_GDM** в файле скважин, "
        "выберите для каждого кода соответствующий **PVTNUM** модели. Иначе веса по добыче не сольются со скважинами."
    )
    for u in uniq_sorted:
        rk = _prod_obj_key(u)
        auto = _prod_code_resolves_to_model_pvt(u, wset)
        default_pvt = auto if auto is not None else int(model_pvts[0])
        idx = model_pvts.index(default_pvt) if default_pvt in model_pvts else 0
        wkey = _pvtmap_widget_key(key_prefix, tag, rk)
        st.selectbox(
            f"Код в файле добычи: `{u}` → PVTNUM",
            options=model_pvts,
            index=idx,
            format_func=lambda x: f"{int(x)}",
            key=wkey,
            help="Этот PVTNUM должен совпадать с регионом скважины в геомодели для корректного merge весов.",
        )
    out = df_prod.copy()

    def _cell(v: object) -> object:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return v
        rk = _prod_obj_key(v)
        wkey = _pvtmap_widget_key(key_prefix, tag, rk)
        if wkey in st.session_state:
            return int(st.session_state[wkey])
        r = _prod_code_resolves_to_model_pvt(v, wset)
        return int(r) if r is not None else v

    out["PVTNUM_GDM"] = out["PVTNUM_GDM"].map(_cell)
    return out


@st.cache_data(show_spinner=False)
def _load_kkd_cached(db_path: str, mtime: float) -> pd.DataFrame:
    _ = mtime
    return load_kkd_dataframe(db_path)


def _guess_col(cols: list[str], candidates: list[str]) -> str:
    upper = {c.upper(): c for c in cols}
    for cand in candidates:
        for uc, orig in upper.items():
            if cand in uc:
                return orig
    return cols[0]


def _guess_bc_lab_n_column(cols: list[str]) -> str:
    """Столбец показателя n в лабораторной таблице БК: по умолчанию колонка «n»."""
    for c in cols:
        if str(c).strip() == "n":
            return c
    for c in cols:
        if str(c).strip().lower() == "n":
            return c
    return _safe_guess_col(cols, ["N", "N_LAB", "ПОКАЗАТЕЛЬ"])


def _guess_kng_w_column(cols: list[str]) -> str:
    """
    Автовыбор столбца нефтенасыщенности (Кнг): Kn, Кн, soil, КНГ, нефтенасыщенность и т.п.
    """
    if not cols:
        return ""
    scored: list[tuple[int, str]] = []
    for c in cols:
        raw = str(c).strip()
        u = raw.upper().replace(" ", "").replace("_", "")
        score = 0
        if "КНГ" in u or "KNG" in u:
            score += 8
        if "НЕФТЕНАС" in raw.upper():
            score += 8
        if "SOIL" in u:
            score += 6
        if "OILSAT" in u or "SOIL" in raw.upper():
            score += 4
        if u in ("KN", "КН") or u.endswith("KN") or u.endswith("КН"):
            score += 5
        if "КН" in raw.upper() and "КНГ" not in raw.upper() and len(raw) <= 12:
            score += 3
        if "НЕФТ" in raw.upper() and "КВО" not in raw.upper() and "ВОД" not in raw.upper():
            score += 2
        if "RIGIS" in u:
            score += 2
        if score > 0:
            scored.append((score, c))
    if scored:
        scored.sort(key=lambda t: (-t[0], len(str(t[1]))))
        return str(scored[0][1])
    return _guess_col(cols, ["КНГ", "KNG", "KN", "SOIL", "НЕФТЕНАС", "RIGIS", "КНН"])


def _lab_counts_by_pvt(
    pvts: list[int],
    cloud: pd.DataFrame | None,
    pvt_horizon_map: dict[int, list[str]],
) -> dict[int, int]:
    """Число лабораторных точек J–Swn по горизонтам, привязанным к каждому PVTNUM."""
    out = {int(p): 0 for p in pvts}
    if cloud is None or cloud.empty:
        return out
    if not {"lab_horizon", "Swn", "J_lab"}.issubset(cloud.columns):
        return out
    for pvt in pvts:
        hs = pvt_horizon_map.get(pvt) or []
        if not hs:
            continue
        hs_set = {str(h).strip() for h in hs}
        sub = cloud.loc[cloud["lab_horizon"].astype(str).str.strip().isin(hs_set)]
        swn = pd.to_numeric(sub["Swn"], errors="coerce")
        jj = pd.to_numeric(sub["J_lab"], errors="coerce")
        out[int(pvt)] = int((swn.notna() & jj.notna() & (swn > 0) & (jj > 0)).sum())
    return out


def _build_j_envelopes_by_pvt(
    pvts: list[int],
    cloud: pd.DataFrame | None,
    pvt_horizon_map: dict[int, list[str]],
) -> dict[int, dict]:
    """
    Коридор J(Swn)=a·Swn^b по лаборатории для штрафа в optimize (нижняя/верхняя + диапазон Swn).
    """
    out: dict[int, dict] = {}
    if cloud is None or cloud.empty:
        return out
    need = {"lab_horizon", "Swn", "J_lab"}
    if not need.issubset(cloud.columns):
        return out
    for pvt in pvts:
        hs = pvt_horizon_map.get(pvt) or []
        if not hs:
            continue
        hs_set = {str(h).strip() for h in hs}
        sub = cloud.loc[cloud["lab_horizon"].astype(str).str.strip().isin(hs_set)]
        if len(sub) < 5:
            continue
        swn = pd.to_numeric(sub["Swn"], errors="coerce").to_numpy(dtype=float)
        jj = pd.to_numeric(sub["J_lab"], errors="coerce").to_numpy(dtype=float)
        m = np.isfinite(swn) & np.isfinite(jj) & (swn > 0) & (jj > 0)
        if int(m.sum()) < 5:
            continue
        info = auto_ab_bounds_from_cloud(swn[m], jj[m])
        lo = info.get("lower") or {}
        up = info.get("upper") or {}
        try:
            alo, blo = float(lo["a"]), float(lo["b"])
            ahi, bhi = float(up["a"]), float(up["b"])
        except (KeyError, TypeError, ValueError):
            continue
        if not all(np.isfinite([alo, blo, ahi, bhi])):
            continue
        s_pos = swn[m]
        out[int(pvt)] = {
            "lower": {"a": alo, "b": blo},
            "upper": {"a": ahi, "b": bhi},
            "s_min": float(np.min(s_pos)),
            "s_max": float(np.max(s_pos)),
        }
    return out


def _add_pvit_n_if_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Если в лабораторной таблице нет Pvit/n, пытаемся рассчитать их по группам
    (скважина+образец) из зависимости ln(Pc)=ln(Pvit)-n*ln(Swn).
    """
    out = df.copy()
    if "Pvit" in out.columns and "n" in out.columns:
        return out

    cols = list(out.columns)
    swn_col = "Swn" if "Swn" in out.columns else ("Swn.1" if "Swn.1" in out.columns else None)
    pc_col = None
    for c in cols:
        cc = str(c).replace("\n", " ").strip().lower()
        if "капиллярное давление" in cc:
            pc_col = c
            break
    well_col = "Скважина" if "Скважина" in out.columns else None
    sample_cols = [c for c in ["Номер образца", "Порядковый номер образца"] if c in out.columns]
    if swn_col is None or pc_col is None or well_col is None or not sample_cols:
        return out

    out["_swn_tmp"] = pd.to_numeric(out[swn_col], errors="coerce")
    out["_pc_tmp"] = pd.to_numeric(out[pc_col], errors="coerce")
    pvit = pd.Series(np.nan, index=out.index, dtype=float)
    nval = pd.Series(np.nan, index=out.index, dtype=float)
    group_cols = [well_col] + sample_cols

    for _, g in out.groupby(group_cols, dropna=False):
        m = np.isfinite(g["_swn_tmp"]) & np.isfinite(g["_pc_tmp"]) & (g["_swn_tmp"] > 0) & (g["_pc_tmp"] > 0)
        gg = g.loc[m]
        if len(gg) < 3:
            continue
        x = np.log(gg["_swn_tmp"].to_numpy(dtype=float))
        y = np.log(gg["_pc_tmp"].to_numpy(dtype=float))
        slope, intercept = np.polyfit(x, y, 1)
        n_est = -float(slope)
        pvit_est = float(np.exp(intercept))
        if np.isfinite(n_est) and np.isfinite(pvit_est) and n_est > 0 and pvit_est > 0:
            pvit.loc[g.index] = pvit_est
            nval.loc[g.index] = n_est

    out["Pvit"] = pvit
    out["n"] = nval
    out = out.drop(columns=["_swn_tmp", "_pc_tmp"])
    return out


def _get_bc_source_df() -> tuple[pd.DataFrame | None, str]:
    """
    Приоритет источников БК:
    1) Только пользовательский файл БК (вкладка «Лаборатория»), если он загружен — без подмешивания ККД/БД.
    2) Иначе — данные из БД ККД (sqlite).
    """
    user = st.session_state.get("bc_user_upload_df")
    if isinstance(user, pd.DataFrame) and not user.empty:
        return _add_pvit_n_if_missing(_normalize_columns(user)), "пользовательский файл БК"

    try:
        db_file = sqlite_path()
        if db_file.is_file():
            kkd_df = _load_kkd_cached(str(db_file), db_file.stat().st_mtime)
            if isinstance(kkd_df, pd.DataFrame) and not kkd_df.empty:
                return _add_pvit_n_if_missing(_normalize_columns(kkd_df)), "файл/БД ККД"
    except Exception:
        pass
    return None, ""


def _power_curve_from_ab(x: np.ndarray, a: float, b: float) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        y = a * (x**b)
    return np.nan_to_num(y, nan=np.nan, posinf=np.nan, neginf=np.nan)


def _swl_a_exp_b_poro_curve(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """swl = a*exp(b*poro) для графика (как в расчёте)."""
    p = np.asarray(x, dtype=float)
    z = np.clip(p * float(b), -60.0, 60.0)
    with np.errstate(over="ignore"):
        y = float(a) * np.exp(z)
    return np.clip(y, 1e-12, 1.0)


def _plot_bc_cloud(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    lower_ab: tuple[float, float] | None = None,
    upper_ab: tuple[float, float] | None = None,
    opt_ab: tuple[float, float] | None = None,
    *,
    curve_kind: str = "power",
) -> go.Figure:
    fig = px.scatter(df, x=x_col, y=y_col, color="HORIZON" if "HORIZON" in df.columns else None, opacity=0.65, title=title)
    fig.update_layout(legend_title_text="Регион")
    fig.update_traces(hovertemplate=f"{x_col}=%{{x:.3f}}<br>{y_col}=%{{y:.3f}}<extra></extra>")
    x = pd.to_numeric(df[x_col], errors="coerce").to_numpy(dtype=float)
    x = x[np.isfinite(x) & (x > 0)]
    if len(x) == 0:
        return fig
    grid = np.linspace(float(np.min(x)), float(np.max(x)), 150)
    if curve_kind == "swl_exp_ab":
        if lower_ab is not None:
            yl = _swl_a_exp_b_poro_curve(grid, lower_ab[0], lower_ab[1])
            fig.add_trace(
                go.Scatter(
                    x=grid,
                    y=yl,
                    mode="lines",
                    name=f"Нижняя swl: a={lower_ab[0]:.3g}, b={lower_ab[1]:.3g}",
                    line=dict(color="steelblue", dash="dash"),
                )
            )
        if upper_ab is not None:
            yu = _swl_a_exp_b_poro_curve(grid, upper_ab[0], upper_ab[1])
            fig.add_trace(
                go.Scatter(
                    x=grid,
                    y=yu,
                    mode="lines",
                    name=f"Верхняя swl: a={upper_ab[0]:.3g}, b={upper_ab[1]:.3g}",
                    line=dict(color="darkorange", dash="dashdot"),
                )
            )
        if opt_ab is not None:
            yo = _swl_a_exp_b_poro_curve(grid, opt_ab[0], opt_ab[1])
            fig.add_trace(
                go.Scatter(
                    x=grid,
                    y=yo,
                    mode="lines",
                    name=f"swl=a·exp(b·Кп): a={opt_ab[0]:.3g}, b={opt_ab[1]:.3g}",
                    line=dict(color="crimson", width=3),
                )
            )
        return fig
    if lower_ab is not None:
        yl = _power_curve_from_ab(grid, lower_ab[0], lower_ab[1])
        fig.add_trace(go.Scatter(x=grid, y=yl, mode="lines", name=f"Нижняя: a={lower_ab[0]:.3g}, b={lower_ab[1]:.3g}", line=dict(color="steelblue", dash="dash")))
    if upper_ab is not None:
        yu = _power_curve_from_ab(grid, upper_ab[0], upper_ab[1])
        fig.add_trace(go.Scatter(x=grid, y=yu, mode="lines", name=f"Верхняя: a={upper_ab[0]:.3g}, b={upper_ab[1]:.3g}", line=dict(color="darkorange", dash="dashdot")))
    if opt_ab is not None:
        yo = _power_curve_from_ab(grid, opt_ab[0], opt_ab[1])
        fig.add_trace(go.Scatter(x=grid, y=yo, mode="lines", name=f"Оптимальная: a={opt_ab[0]:.3g}, b={opt_ab[1]:.3g}", line=dict(color="crimson", width=3)))
    return fig


def _fig_j_swn_lab(
    cloud: pd.DataFrame,
    title: str,
    trend_fit: dict | None = None,
    extra_lines: list[dict] | None = None,
    optimal: dict | None = None,
    low_swn_threshold: float = DEFAULT_LOW_SWN_THRESHOLD,
    j_cap_at_low_swn: float = DEFAULT_J_CAP_AT_LOW_SWN,
) -> go.Figure:
    """cloud: колонки Swn, J_lab, lab_area."""
    fig = px.scatter(
        cloud,
        x="Swn",
        y="J_lab",
        color="lab_area",
        hover_data=[c for c in ("lab_horizon", "Swn", "J_lab") if c in cloud.columns],
        title=title,
        opacity=0.65,
    )
    s_min = float(cloud["Swn"].min())
    s_max = float(cloud["Swn"].max())
    if s_max <= s_min:
        s_max = s_min + 1e-6
    grid = np.linspace(s_min, s_max, 120)
    thr = float(low_swn_threshold)
    cap = float(j_cap_at_low_swn)
    if thr > 0 and s_min <= thr <= s_max:
        grid = np.unique(np.sort(np.concatenate([grid, [thr]])))

    def _j_on_grid(a: float, b: float) -> np.ndarray:
        return apply_low_swn_j_cap(grid, j_power_from_swn(grid, a, b), swn_threshold=thr, j_cap=cap)

    ann_text = []
    if trend_fit and np.isfinite(trend_fit.get("a", np.nan)) and np.isfinite(trend_fit.get("b", np.nan)):
        a, b = trend_fit["a"], trend_fit["b"]
        fig.add_trace(
            go.Scatter(
                x=grid,
                y=_j_on_grid(a, b),
                mode="lines",
                name=f"Тренд лаб.: J = {a:.4g}·Swn^{b:.4g}",
                line=dict(width=2, color="black"),
            )
        )
        ann_text.append(
            f"Лаб. тренд: J = {a:.4g}·Swn<sup>{b:.4g}</sup> "
            f"(при Swn≤{thr:g} — J≤{cap:g})"
        )

    if extra_lines:
        for line in extra_lines:
            name = line.get("name", "линия")
            dash = line.get("dash", "dash")
            col = line.get("color", "gray")
            if "x" in line and "y" in line:
                xs = np.asarray(line["x"], dtype=float)
                ys = np.asarray(line["y"], dtype=float)
                fig.add_trace(
                    go.Scatter(
                        x=xs,
                        y=ys,
                        mode="lines",
                        name=name,
                        line=dict(width=2, dash=dash, color=col),
                    )
                )
            else:
                aa, bb = line["a"], line["b"]
                fig.add_trace(
                    go.Scatter(
                        x=grid,
                        y=_j_on_grid(aa, bb),
                        mode="lines",
                        name=name,
                        line=dict(width=2, dash=dash, color=col),
                    )
                )

    if optimal and np.isfinite(optimal.get("a", np.nan)) and np.isfinite(optimal.get("b", np.nan)):
        a, b = optimal["a"], optimal["b"]
        y_opt = _j_on_grid(a, b)
        fig.add_trace(
            go.Scatter(
                x=grid,
                y=y_opt,
                mode="lines",
                name=f"Модель (опт.): J = {a:.4g}·Swn^{b:.4g}",
                line=dict(width=3, color="crimson"),
            )
        )
        ann_text.append(
            f"Опт. модель: J = {a:.4g}·Swn<sup>{b:.4g}</sup> "
            f"(при Swn≤{low_swn_threshold:g} — J≤{j_cap_at_low_swn:g})"
        )

    if ann_text:
        fig.update_layout(annotations=[dict(x=0.02, y=0.98, xref="paper", yref="paper", showarrow=False, align="left", text="<br>".join(ann_text), font=dict(size=12))])
    fig.update_layout(xaxis_title="Swn", yaxis_title="J (лаб.)")
    return fig


def laboratory_tab() -> None:
    st.title("Лаборатория")
    st.caption(
        "Источник облака J–Swn: либо база ККД (`data/ККД_БД.xlsx` → SQLite), либо Excel с матрицей ступеней "
        "(первая строка — группы «водонасыщенность» и «J функция», вторая — подстолбцы «1 ст.» … «N ст.»)."
    )

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    st.subheader("Лаборатория: данные для метода Брукса–Кори")
    st.caption("Можно загрузить сразу, без настройки облака J(Swn).")
    bc_file = st.file_uploader(
        "Загрузить таблицу лабораторных точек (файл зависимости_облако_точек)",
        type=["csv", "xlsx", "xls"],
        key="bc_lab_upload",
    )
    if bc_file is not None:
        _bc_lab_uid = f"{bc_file.name}_{bc_file.size}"
        if st.session_state.get("_bc_lab_upload_id") != _bc_lab_uid:
            _clear_methods_tabs_results()
            st.session_state["_bc_lab_upload_id"] = _bc_lab_uid
        try:
            bc_df = _read_table(bc_file)
            bc_df = _add_pvit_n_if_missing(_normalize_columns(bc_df))
            st.session_state["bc_user_upload_df"] = bc_df
            st.session_state["bc_user_upload_rev"] = int(time.time() * 1000) % 1_000_000_000
            st.success(f"Данные Брукса-Кори загружены: {len(bc_df)} строк (используются только они, без ККД из БД).")
        except Exception as e:
            st.error(f"Ошибка загрузки данных Брукса-Кори: {e}")

    st.divider()

    st.subheader("Лаборатория: данные для метода J функции")
    lab_src = st.radio(
        "Источник лабораторных точек для J(Swn)",
        options=("matrix", "kkd"),
        index=0,
        format_func=lambda x: "База ККД (Excel → SQLite)" if x == "kkd" else "Excel: матрица ступеней (Sw + J)",
        horizontal=True,
        key="lab_cloud_source_mode",
    )
    _prev_lab_src = st.session_state.get("_lab_src_mode_sig")
    if _prev_lab_src is not None and _prev_lab_src != lab_src:
        _clear_lab_j_session()
    st.session_state["_lab_src_mode_sig"] = lab_src

    df: pd.DataFrame | None = None

    if lab_src == "kkd":
        if excel_path().is_file() and not sqlite_path().is_file():
            try:
                build_kkd_sqlite()
            except Exception:
                pass
        up = st.file_uploader("Загрузить Excel ККД (сохранится как data/ККД_БД.xlsx)", type=["xlsx", "xls"], key="kkd_upload")
        if up is not None:
            _kkd_uid = f"{up.name}_{up.size}"
            if st.session_state.get("_lab_kkd_upload_id") != _kkd_uid:
                _clear_lab_j_session()
                st.session_state["_lab_kkd_upload_id"] = _kkd_uid
            target = DATA_DIR / "ККД_БД.xlsx"
            target.write_bytes(up.getbuffer())
            st.success(f"Файл сохранён: {target}")

        c1, c2 = st.columns(2)
        with c1:
            if st.button("Пересобрать SQLite из Excel", type="primary"):
                try:
                    dbp = build_kkd_sqlite()
                    _clear_lab_j_session()
                    st.success(f"База обновлена: {dbp}")
                except FileNotFoundError as e:
                    st.error(str(e))
        with c2:
            st.caption(f"Excel: `{excel_path()}`  |  SQLite: `{sqlite_path()}`")

        try:
            db_file = sqlite_path()
            df = _load_kkd_cached(str(db_file), db_file.stat().st_mtime)
            st.session_state["lab_source_df"] = df
        except FileNotFoundError:
            st.info(
                "База ещё не создана. Положите `ККД_БД.xlsx` в папку `data` или загрузите файл выше, "
                "затем нажмите «Пересобрать SQLite из Excel». Либо выберите источник «матрица ступеней»."
            )
            df = None
    else:
        st.markdown(
            "Загрузите `.xlsx` / `.xls`, где **первая строка** задаёт объединённые заголовки "
            "(блок водонасыщенности и блок J), **вторая** — подписи ступеней («1 ст.», …). "
            "Таблица разворачивается: для каждой строки считаются `Sw_min`/`Sw_max` по всем ступеням, "
            "`Swn = (Sw − Sw_min)/(Sw_max − Sw_min)`; если Swn получается 0 или 1 — в ячейку записывается пропуск."
        )
        mat_up = st.file_uploader("Файл матрицы ступеней", type=["xlsx", "xls"], key="lab_matrix_stairs_upload")
        cached_df = st.session_state.get("lab_source_df")
        if mat_up is None:
            if st.session_state.get("lab_cloud_ready") and isinstance(cached_df, pd.DataFrame) and not cached_df.empty:
                df = cached_df
            else:
                if st.session_state.get("_lab_matrix_upload_id") and not st.session_state.get("lab_cloud_ready"):
                    _clear_lab_j_session()
                st.info("Выберите файл Excel с матрицей ступеней — или переключитесь на источник «База ККД».")
                df = None
        else:
            _mat_uid = f"{mat_up.name}_{mat_up.size}"
            if st.session_state.get("_lab_matrix_upload_id") != _mat_uid:
                _clear_lab_j_session()
                st.session_state["_lab_matrix_upload_id"] = _mat_uid
            _def_kw_w = ", ".join(DEFAULT_MATRIX_WATER_KEYWORDS)
            _def_kw_j = ", ".join(DEFAULT_MATRIX_J_KEYWORDS)
            if "lab_matrix_kw_water_text" not in st.session_state:
                st.session_state["lab_matrix_kw_water_text"] = _def_kw_w
            if "lab_matrix_kw_j_text" not in st.session_state:
                st.session_state["lab_matrix_kw_j_text"] = _def_kw_j
            with st.expander("Подписи блоков (первая строка Excel)", expanded=False):
                st.caption(
                    "Укажите подстроки, которые встречаются в **объединённой первой строке** заголовка для блока "
                    "водонасыщенности и для блока J. Разделитель — запятая, точка с запятой или новая строка. "
                    "Колонка попадает в блок только если совпала первая строка **и** вторая похожа на ступень "
                    "(«1 ст.», …). **Площадь**, **№ скв.**, **Фация**, **№ обр.**, **Код горизонта** (в т.ч. под объединённым заголовком J) "
                    "всегда считаются метаданными."
                )
                st.text_area(
                    "Ключевые слова блока водонасыщенности",
                    height=70,
                    key="lab_matrix_kw_water_text",
                    help=f"По умолчанию: {_def_kw_w}",
                )
                st.text_area(
                    "Ключевые слова блока J",
                    height=70,
                    key="lab_matrix_kw_j_text",
                    help=f"По умолчанию: {_def_kw_j}",
                )
            water_kw = parse_matrix_block_keywords(
                st.session_state.get("lab_matrix_kw_water_text"), DEFAULT_MATRIX_WATER_KEYWORDS
            )
            j_kw = parse_matrix_block_keywords(st.session_state.get("lab_matrix_kw_j_text"), DEFAULT_MATRIX_J_KEYWORDS)
            try:
                raw = pd.read_excel(io.BytesIO(mat_up.getbuffer()), header=[0, 1], engine="openpyxl")
            except Exception:
                try:
                    raw = pd.read_excel(io.BytesIO(mat_up.getbuffer()), header=[0, 1])
                except Exception as e2:
                    st.error(f"Не удалось прочитать Excel: {e2}")
                    return
            if not is_likely_j_matrix_stairs_format(raw, water_keywords=water_kw, j_keywords=j_kw):
                st.error(
                    "Файл не распознан как матрица ступеней: нет двухуровневого заголовка или не найдены столбцы "
                    "ступеней по заданным ключам. Проверьте первые две строки листа или расширьте ключевые слова выше."
                )
                return
            try:
                sw_c, j_c, meta_c = classify_j_matrix_stairs_columns(raw, water_keywords=water_kw, j_keywords=j_kw)
                st.caption(
                    f"Распознано: водонасыщенность — {len(sw_c)} столбцов, J — {len(j_c)} столбцов, "
                    f"метаданные — {len(meta_c)} столбцов."
                )
                df = transform_j_stairs_wide_to_long(raw, water_keywords=water_kw, j_keywords=j_kw)
            except Exception as e:
                st.error(f"Ошибка преобразования в длинный формат: {e}")
                return
            if df.empty:
                st.error("После разворота не осталось строк.")
                return
            st.success(f"Разворот матрицы: {len(df)} строк (ступени × исходные образцы).")
            st.session_state["lab_source_df"] = df
            with st.expander("Предпросмотр (первые 20 строк)", expanded=False):
                st.dataframe(df.head(20), use_container_width=True)

    if df is None:
        if st.session_state.get("lab_cloud_ready"):
            cloud = st.session_state.get("lab_cloud")
            if isinstance(cloud, pd.DataFrame) and not cloud.empty:
                st.success(
                    f"Для вкладки «Подбор J» сохранено **{len(cloud)}** точек лаборатории. "
                    "Чтобы изменить фильтр — снова выберите источник и файл ниже."
                )
                st.subheader("График J(Swn) (лаборатория)")
                fit = st.session_state.get("lab_trend_fit") or {}
                fig = _fig_j_swn_lab(cloud, title="Лабораторные данные", trend_fit=fit, extra_lines=None, optimal=None)
                st.plotly_chart(fig, use_container_width=True)
        return

    cols = list(df.columns)
    lab_meta_saved = st.session_state.get("lab_meta") or {}

    area_guess = guess_area_column(cols)
    hor_guess = guess_horizon_column(cols)
    j_guess = guess_j_column(cols)
    _, swn_list = pick_second_swn_column(cols)

    st.subheader("Сопоставление колонок")
    c1, c2, c3, c4 = st.columns(4)
    area_col = c1.selectbox(
        "Колонка площади",
        options=cols,
        index=_lab_selectbox_index(cols, lab_meta_saved.get("area_col"), area_guess),
        key="lab_map_area_col",
    )
    hor_col = c2.selectbox(
        "Колонка кода горизонта",
        options=cols,
        index=_lab_selectbox_index(cols, lab_meta_saved.get("horizon_col"), hor_guess),
        key="lab_map_hor_col",
    )
    j_col = c3.selectbox(
        "J (функция Леверетта)",
        options=cols,
        index=_lab_selectbox_index(cols, lab_meta_saved.get("j_col"), j_guess),
        key="lab_map_j_col",
    )
    default_swn = "Swn" if "Swn" in cols else cols[0]
    swn_col = c4.selectbox(
        "Swn",
        options=cols,
        index=_lab_selectbox_index(cols, lab_meta_saved.get("swn_col"), default_swn),
        key="lab_map_swn_col",
        help=(f"Найденные кандидаты SWn: {swn_list}" if swn_list else None),
    )

    areas = sorted(pd.Series(df[area_col]).dropna().astype(str).str.strip().unique().tolist())
    horizons = sorted(pd.Series(df[hor_col]).dropna().astype(str).str.strip().unique().tolist())

    st.subheader("Фильтр данных")
    st.caption("Площади и горизонты сохраняются в рамках сессии (в том числе после перехода на другие вкладки).")
    if st.session_state.get("lab_cloud_ready"):
        _restore_lab_filter_widgets(areas, horizons)
    else:
        areas_sig = (area_col, tuple(areas))
        if st.session_state.get("_lab_areas_sig") != areas_sig:
            st.session_state["lab_sel_areas"] = areas.copy()
            st.session_state["_lab_areas_sig"] = areas_sig
        elif not st.session_state.get("lab_sel_areas"):
            st.session_state["lab_sel_areas"] = areas.copy()
        else:
            kept = [a for a in st.session_state["lab_sel_areas"] if a in areas]
            st.session_state["lab_sel_areas"] = kept if kept else areas.copy()
    if not st.session_state.get("lab_cloud_ready"):
        if "lab_sel_hors" not in st.session_state:
            st.session_state["lab_sel_hors"] = horizons.copy()
        else:
            st.session_state["lab_sel_hors"] = [h for h in st.session_state["lab_sel_hors"] if h in horizons]
            if not st.session_state["lab_sel_hors"] and horizons:
                st.session_state["lab_sel_hors"] = horizons.copy()
    cfa1, cfa2 = st.columns([1, 3])
    all_selected = set(st.session_state.get("lab_sel_areas", [])) == set(areas) and len(areas) > 0
    if cfa1.button("Выбрать все (площади)" if not all_selected else "Снять все (площади)", key="lab_toggle_all_areas"):
        st.session_state["lab_sel_areas"] = [] if all_selected else areas.copy()
    sel_areas = cfa2.multiselect("Площади", options=areas, key="lab_sel_areas")
    sel_hors = st.multiselect(
        "Коды горизонтов (можно несколько)",
        options=horizons,
        key="lab_sel_hors",
    )

    if st.button("Применить фильтр и сохранить выбор для вкладки «Подбор»", type="primary"):
        if not sel_areas or not sel_hors:
            st.error("Выберите хотя бы одну площадь и один код горизонта.")
            return
        filt = filter_lab_df(df, area_col, hor_col, sel_areas, sel_hors)
        filt = filt.assign(
            lab_area=filt[area_col].astype(str).str.strip(),
            lab_horizon=filt[hor_col].astype(str).str.strip(),
            Swn=pd.to_numeric(filt[swn_col], errors="coerce"),
            J_lab=pd.to_numeric(filt[j_col], errors="coerce"),
        )
        filt = filt.dropna(subset=["Swn", "J_lab"])
        filt = filt[(filt["Swn"] > 0) & (filt["J_lab"] > 0)]
        if filt.empty:
            st.error("После фильтрации не осталось валидных точек (Swn>0, J>0).")
            return
        st.session_state["lab_cloud"] = filt.reset_index(drop=True)
        st.session_state["lab_cloud_ready"] = True
        st.session_state["lab_meta"] = {
            "area_col": area_col,
            "horizon_col": hor_col,
            "swn_col": swn_col,
            "j_col": j_col,
            "areas": sel_areas,
            "horizons": sel_hors,
            "source_mode": lab_src,
        }
        st.session_state["lab_source_df"] = df
        _clear_methods_tabs_results()
        _clear_pvt_horizon_multiselects()
        fit = fit_power_j_swn(filt["Swn"].to_numpy(), filt["J_lab"].to_numpy())
        st.session_state["lab_trend_fit"] = fit
        st.success(
            f"Сохранено точек: {_fmt_float3(len(filt))}. Лаб. тренд: "
            f"a={_fmt_float3(fit.get('a'))}, b={_fmt_float3(fit.get('b'))}, R²={_fmt_float3(fit.get('r2'))}"
        )

    cloud = st.session_state.get("lab_cloud")
    if cloud is not None and not cloud.empty:
        st.subheader("График J(Swn) (лаборатория)")
        fit = st.session_state.get("lab_trend_fit") or {}
        fig = _fig_j_swn_lab(cloud, title="Лабораторные данные", trend_fit=fit, extra_lines=None, optimal=None)
        st.plotly_chart(fig, use_container_width=True)


def leverett_tab() -> None:
    st.title("Подбор J функции Леверетта")
    _scroll_to_top_if_pending()
    st.write("Загрузите данные скважин, задайте ограничения или автограницы, выполните подбор.")

    lab_ready = bool(st.session_state.get("lab_cloud_ready"))
    cloud: pd.DataFrame | None = st.session_state.get("lab_cloud")
    lab_hors = _lab_j_horizon_universe()

    with st.sidebar:
        st.header("Загрузка данных")
        wells_file = st.file_uploader(
            "Файл скважин", type=["csv", "xlsx", "xls", "txt"], key=_wells_file_uploader_key()
        )
        prod_file = st.file_uploader(
            "Файл добычи (опционально для весов)", type=["csv", "xlsx", "xls"], key=_prod_file_uploader_key()
        )
        c_clr1, c_clr2 = st.columns(2)
        if c_clr1.button("Сбросить скважины", help="Убрать сохранённый файл скважин из расчёта", key="j_clr_wells"):
            _clear_shared_wells_files()
            st.rerun()
        if c_clr2.button("Сбросить добычу", help="Убрать сохранённый файл добычи из расчёта", key="j_clr_prod"):
            _clear_shared_prod_files()
            st.rerun()
        optimizer_method = st.selectbox(
            "Метод оптимизации",
            options=["differential_evolution", "pso", "dual_annealing"],
            format_func=lambda x: {
                "differential_evolution": "Дифференциальная эволюция",
                "pso": "Рой частиц (PSO)",
                "dual_annealing": "Dual Annealing",
            }[x],
            index=0,
        )
        use_perf_weights = st.checkbox(
            "Учитывать перфорации (Perf_GDM) в весах",
            value=False,
            key="j_use_perf_weights",
            help="Если выключено, колонка Perf_GDM игнорируется при расчете весов.",
        )
        maxiter = st.slider("Итерации оптимизации", min_value=20, max_value=300, value=200, step=10)
        popsize = st.slider("Размер популяции", min_value=8, max_value=40, value=20, step=1)
        st.markdown("**Ограничение J при малых Swn (только обучение)**")
        low_swn_threshold = st.number_input(
            "Порог Swn",
            min_value=1e-6,
            max_value=0.5,
            value=DEFAULT_LOW_SWN_THRESHOLD,
            step=0.001,
            format="%.4f",
            key="j_low_swn_threshold",
            help="При Swn не выше этого порога в целевой функции J(Swn)=a·Swn^b ограничивается сверху.",
        )
        j_cap_at_low_swn = st.number_input(
            "Макс. J при Swn ≤ порога",
            min_value=0.1,
            max_value=10000.0,
            value=DEFAULT_J_CAP_AT_LOW_SWN,
            step=0.5,
            key="j_cap_at_low_swn",
            help="Если расчётная J выше — при обучении используется это значение (по умолчанию 20).",
        )

    if wells_file is not None:
        p = _persist_uploaded_file(wells_file, "wells")
        if st.session_state.get("_active_wells_path") != p:
            _clear_leverett_results()
            _clear_pvt_horizon_multiselects()
        st.session_state["_active_wells_path"] = p
        st.session_state["wells_file_path"] = p
        st.session_state["shared_wells_file_path"] = p
    if prod_file is not None:
        p = _persist_uploaded_file(prod_file, "prod")
        st.session_state["prod_file_path"] = p
        st.session_state["shared_prod_file_path"] = p

    wells_path = _coalesce_path("shared_wells_file_path", "wells_file_path", "bc_wells_path")

    if lab_ready and lab_hors:
        if not wells_path:
            st.subheader("Связь код горизонта -> PVTNUM")
            st.warning(
                "Лабораторное облако J сохранено. Загрузите **файл скважин** в боковой панели — "
                "затем появятся поля привязки горизонтов к PVTNUM из выгрузки."
            )
        else:
            st.caption(
                f"Лаборатория: сохранено {len(cloud) if isinstance(cloud, pd.DataFrame) else 0} точек, "
                f"кодов горизонта: {len(lab_hors)}."
            )
    elif not lab_ready:
        st.info(
            "Для привязки горизонтов к PVTNUM сначала во вкладке **«Лаборатория»** выберите данные "
            "и нажмите **«Применить фильтр и сохранить выбор для вкладки «Подбор»»**."
        )

    if not wells_path:
        st.info("Загрузите файл скважин, чтобы продолжить.")
        return

    try:
        raw_wells = _read_table_from_path(wells_path)
        raw_prod = None
        prod_path = _coalesce_path("shared_prod_file_path", "prod_file_path", "bc_prod_path")
        if prod_path:
            raw_prod = _read_table_from_path(prod_path)
        mapped_wells = _map_uploaded_wells_df(raw_wells, key_prefix="j", title="Сопоставление колонок файла скважин")
        if mapped_wells.empty:
            return
        mapped_prod = _map_uploaded_prod_df(raw_prod, key_prefix="j", title="Сопоставление колонок файла добычи")
        if isinstance(mapped_prod, pd.DataFrame) and mapped_prod.empty:
            return
        fwl_settings = _fwl_filter_settings_ui(mapped_wells, key_prefix="shared")
        df_wells = _clean_wells_cached(
            mapped_wells,
            fwl_mode=fwl_settings["mode"] if fwl_settings["has_fwl"] else None,
            fwl_min=float(fwl_settings["min_continuous"]),
            fwl_exclude=tuple(fwl_settings["exclude_discrete"]),
        )
        if (not use_perf_weights) and ("Perf_GDM" in df_wells.columns):
            df_wells = df_wells.drop(columns=["Perf_GDM"])
        df_prod = _clean_prod_cached(mapped_prod)
    except Exception as e:
        st.error(f"Ошибка чтения файла: {e}")
        return
    st.caption(
        f"Выбранный файл скважин: `{Path(wells_path).name}`"
        + (f" | Выбранный файл добычи: `{Path(prod_path).name}`" if prod_path else " | Выбранный файл добычи: не выбран")
    )

    validation_errors = _validate_columns(df_wells, df_prod)
    if validation_errors:
        for msg in validation_errors:
            st.error(msg)
        return

    pvts = _well_pvts(df_wells)
    if not pvts:
        st.error("Не удалось определить регионы PVTNUM_GDM.")
        return

    _sync_pvt_session_state(wells_path, pvts)

    df_prod_weights = df_prod
    if df_prod is not None and not df_prod.empty and "PVTNUM_GDM" in df_prod.columns:
        df_prod_weights = _apply_prod_pvt_mapping_ui(
            df_prod,
            pvts,
            key_prefix="shared",
            prod_path=str(prod_path) if prod_path else "",
        )

    st.subheader("Предпросмотр данных")
    st.dataframe(_round_df(df_wells.head(30)), use_container_width=True)
    st.caption(f"Строк после очистки: {len(df_wells)}")

    pvt_horizon_map: dict[int, list[str]] = {}
    if lab_ready and lab_hors:
        pvt_horizon_map = _render_pvt_horizon_mapping(
            pvts,
            lab_hors,
            title="Связь код горизонта -> PVTNUM",
        )

    st.subheader("Режим границ a, b")
    if lab_ready and cloud is not None and not cloud.empty:
        bounds_mode = st.radio(
            "Как задавать amin/amax/bmin/bmax",
            options=("Автоподбор по лабораторному облаку", "Ручной ввод"),
            index=0,
            horizontal=True,
        )
    else:
        bounds_mode = "Ручной ввод"
        st.caption("Автоподбор границ доступен после выбора данных во вкладке «Лаборатория».")

    bounds_by_pvt: dict[int, dict[str, tuple[float, float]]] = {}
    auto_preview: dict[int, dict] = {}

    if bounds_mode == "Ручной ввод":
        bounds_by_pvt = _bounds_ui_manual(pvts, df_wells=df_wells)
    else:
        if st.button("Подобрать границы a, b автоматически по облаку", type="primary"):
            if cloud is None or cloud.empty:
                st.error("Нет данных лаборатории в session_state.")
            else:
                if any(not (pvt_horizon_map.get(p) or []) for p in pvts):
                    st.error("Для автоподбора выберите хотя бы один код горизонта для каждого PVTNUM из файла скважин.")
                else:
                    auto_preview = {}
                    new_bounds: dict[int, dict[str, tuple[float, float]]] = {}
                    for pvt in pvts:
                        hs = pvt_horizon_map[pvt]
                        sub = cloud[cloud["lab_horizon"].isin(hs)]
                        if len(sub) < 5:
                            st.warning(f"PVT {pvt}: мало точек в лаборатории ({len(sub)}).")
                        swn = sub["Swn"].to_numpy(dtype=float)
                        jj = sub["J_lab"].to_numpy(dtype=float)
                        info = auto_ab_bounds_from_cloud(swn, jj)
                        amin, amax = info["a_bounds"]
                        bmin, bmax = info["b_bounds"]
                        new_bounds[pvt] = {"a": (amin, amax), "b": (bmin, bmax), "sigma": (25.0, 35.0)}
                        auto_preview[pvt] = info
                    st.session_state["auto_bounds_by_pvt"] = new_bounds
                    st.session_state["auto_preview_by_pvt"] = auto_preview
                    st.success("Границы рассчитаны. Ниже можно скорректировать sigma.")

        stored_bounds = {
            int(p): b
            for p, b in (st.session_state.get("auto_bounds_by_pvt") or {}).items()
            if int(p) in pvts
        }
        if stored_bounds:
            st.markdown("**Текущие автограницы (можно отредактировать sigma)**")
            for pvt in pvts:
                if pvt not in stored_bounds:
                    continue
                bnd = stored_bounds[pvt]
                with st.expander(f"PVTNUM {pvt} (авто a,b)", expanded=False):
                    st.write(
                        f"a: [{bnd['a'][0]:.6g} ; {bnd['a'][1]:.6g}]  |  b: [{bnd['b'][0]:.6g} ; {bnd['b'][1]:.6g}]"
                    )
                    s1 = st.number_input(f"sigma min | PVT {pvt}", value=bnd["sigma"][0], key=f"auto_smin_{pvt}")
                    s2 = st.number_input(f"sigma max | PVT {pvt}", value=bnd["sigma"][1], key=f"auto_smax_{pvt}")
                    stored_bounds[pvt] = {
                        "a": bnd["a"],
                        "b": bnd["b"],
                        "sigma": (float(s1), float(s2)),
                    }
            st.session_state["auto_bounds_by_pvt"] = stored_bounds
            bounds_by_pvt = {p: stored_bounds[p] for p in pvts if p in stored_bounds}

        preview = st.session_state.get("auto_preview_by_pvt") or {}
        if preview:
            st.subheader("Графики облака и огибающих по каждому PVTNUM")
            for pvt in pvts:
                if pvt not in preview:
                    continue
                info = preview[pvt]
                hs = (st.session_state.get("pvt_horizon_map") or {}).get(pvt) or []
                if cloud is None or not hs:
                    continue
                sub = cloud[cloud["lab_horizon"].isin(hs)]
                if sub.empty:
                    continue
                fit = fit_power_j_swn(sub["Swn"].to_numpy(), sub["J_lab"].to_numpy())
                plot = info.get("plot") or {}
                if plot.get("upper") and plot.get("lower"):
                    pl, pu = plot["lower"], plot["upper"]
                    lines = [
                        {
                            "x": pl["x"],
                            "y": pl["y"],
                            "name": f"Нижняя: J = {pl['a']:.3g}·Swn^{pl['b']:.3g}",
                            "dash": "dash",
                            "color": "steelblue",
                        },
                        {
                            "x": pu["x"],
                            "y": pu["y"],
                            "name": f"Верхняя: J = {pu['a']:.3g}·Swn^{pu['b']:.3g}",
                            "dash": "dashdot",
                            "color": "darkorange",
                        },
                    ]
                else:
                    lines = [
                        {"a": info["lower"]["a"], "b": info["lower"]["b"], "name": "Нижняя огибающая", "dash": "dash", "color": "steelblue"},
                        {"a": info["upper"]["a"], "b": info["upper"]["b"], "name": "Верхняя огибающая", "dash": "dashdot", "color": "darkorange"},
                    ]
                fig = _fig_j_swn_lab(
                    sub.assign(lab_area=sub["lab_area"].astype(str)),
                    title=f"PVT {pvt}: лабораторное облако и огибающие",
                    trend_fit=fit,
                    extra_lines=lines,
                    optimal=None,
                    low_swn_threshold=float(
                        st.session_state.get("j_low_swn_threshold", DEFAULT_LOW_SWN_THRESHOLD)
                    ),
                    j_cap_at_low_swn=float(st.session_state.get("j_cap_at_low_swn", DEFAULT_J_CAP_AT_LOW_SWN)),
                )
                st.plotly_chart(fig, use_container_width=True)

    use_fixed = st.checkbox("Рассчитать с заданными коэффициентами a, b, sigma (без оптимизации)", value=False)
    fixed_params = _manual_params_ui(pvts) if use_fixed else None

    j_busy = bool(st.session_state.get("j_busy", False))
    _ui_lock(j_busy, "ui_lock_j")
    run_opt = st.button("Рассчитать оптимальные коэффициенты", type="primary", disabled=j_busy)
    run_fix = st.button("Рассчитать с заданными коэффициентами", type="secondary", disabled=j_busy) if use_fixed else False
    if j_busy:
        st.warning("Идет расчет J-функции... Пожалуйста, дождитесь завершения.")

    def _merge_bounds(partial: dict[int, dict[str, tuple[float, float]]]) -> dict[int, dict[str, tuple[float, float]]]:
        merged = _default_bounds_for_pvts(pvts)
        merged.update({int(p): b for p, b in partial.items() if int(p) in pvts})
        return merged

    j_envelope_arg: dict[int, dict] | None = None
    j_lab_counts: dict[int, int] | None = None
    if (
        lab_ready
        and cloud is not None
        and not cloud.empty
        and pvt_horizon_map
        and not any(not (pvt_horizon_map.get(p) or []) for p in pvts)
    ):
        j_lab_counts = _lab_counts_by_pvt(pvts, cloud, pvt_horizon_map)
        _je = _build_j_envelopes_by_pvt(pvts, cloud, pvt_horizon_map)
        if _je:
            j_envelope_arg = _je

    if run_fix:
        if not use_fixed:
            st.error("Включите «Рассчитать с заданными коэффициентами» и задайте a, b, sigma.")
        elif lab_ready and cloud is not None and any(not (pvt_horizon_map.get(p) or []) for p in pvts):
            st.error("Для режима заданных коэффициентов выберите соответствие горизонтов для каждого PVTNUM.")
        else:
            st.session_state["j_busy"] = True
            _ui_lock(True, "ui_lock_j")
            with st.spinner("Применяются заданные коэффициенты..."):
                try:
                    t0_j = time.perf_counter()
                    result_df, params_df, qa_df, j_timing_df = run_pipeline(
                        df_wells=df_wells,
                        df_prod=df_prod_weights,
                        bounds_by_pvt=_merge_bounds(bounds_by_pvt),
                        maxiter=maxiter,
                        popsize=popsize,
                        fixed_params=fixed_params,
                        optimizer_method=optimizer_method,
                        j_envelope_by_pvt=j_envelope_arg,
                        lab_counts_by_pvt=j_lab_counts,
                        low_swn_threshold=float(low_swn_threshold),
                        j_cap_at_low_swn=float(j_cap_at_low_swn),
                    )
                except Exception as e:
                    st.error(f"Ошибка расчета: {e}")
                else:
                    j_elapsed = float(time.perf_counter() - t0_j)
                    if not j_timing_df.empty:
                        mask_all = j_timing_df["PVTNUM_GDM"] == "Все регионы"
                        if mask_all.any():
                            j_timing_df.loc[mask_all, "elapsed_sec"] = j_elapsed
                    st.session_state["leverett_result_df"] = result_df
                    st.session_state["leverett_params_df"] = params_df
                    st.session_state["leverett_qa_df"] = qa_df
                    st.session_state["j_timing_df"] = j_timing_df
                    st.session_state["j_total_elapsed_sec"] = j_elapsed
                    st.success("Расчет с заданными коэффициентами завершен.")
                finally:
                    st.session_state["j_busy"] = False
                    _ui_lock(False, "ui_lock_j")

    if run_opt:
        if bounds_mode == "Автоподбор по лабораторному облаку" and not bounds_by_pvt:
            st.error("Сначала нажмите «Подобрать границы a, b автоматически по облаку».")
        else:
            st.session_state["j_busy"] = True
            _ui_lock(True, "ui_lock_j")
            with st.spinner("Выполняется подбор коэффициентов..."):
                try:
                    t0_j = time.perf_counter()
                    result_df, params_df, qa_df, j_timing_df = run_pipeline(
                        df_wells=df_wells,
                        df_prod=df_prod_weights,
                        bounds_by_pvt=_merge_bounds(bounds_by_pvt),
                        maxiter=maxiter,
                        popsize=popsize,
                        fixed_params=None,
                        optimizer_method=optimizer_method,
                        j_envelope_by_pvt=j_envelope_arg,
                        lab_counts_by_pvt=j_lab_counts,
                        low_swn_threshold=float(low_swn_threshold),
                        j_cap_at_low_swn=float(j_cap_at_low_swn),
                    )
                except Exception as e:
                    st.error(f"Ошибка расчета: {e}")
                else:
                    j_elapsed = float(time.perf_counter() - t0_j)
                    if not j_timing_df.empty:
                        mask_all = j_timing_df["PVTNUM_GDM"] == "Все регионы"
                        if mask_all.any():
                            j_timing_df.loc[mask_all, "elapsed_sec"] = j_elapsed
                    st.session_state["leverett_result_df"] = result_df
                    st.session_state["leverett_params_df"] = params_df
                    st.session_state["leverett_qa_df"] = qa_df
                    st.session_state["j_timing_df"] = j_timing_df
                    st.session_state["j_total_elapsed_sec"] = j_elapsed
                    st.success("Расчет завершен.")
                finally:
                    st.session_state["j_busy"] = False
                    _ui_lock(False, "ui_lock_j")

    result_df = st.session_state.get("leverett_result_df")
    params_df = st.session_state.get("leverett_params_df")
    qa_df = st.session_state.get("leverett_qa_df")
    if result_df is None or params_df is None or qa_df is None:
        st.info("Запустите расчет, чтобы увидеть результаты и графики.")
        return

    j_total_elapsed = st.session_state.get("j_total_elapsed_sec")
    j_timing_df = st.session_state.get("j_timing_df")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Параметры")
        st.dataframe(_round_df(params_df), use_container_width=True)
        if isinstance(j_timing_df, pd.DataFrame) and not j_timing_df.empty:
            st.subheader("Статистика времени расчета J")
            j_timing_ru = j_timing_df.rename(
                columns={
                    "PVTNUM_GDM": "Регион (PVTNUM)",
                    "rows_geo": "Строк геологии",
                    "rows_lab": "Лабораторных точек",
                    "elapsed_sec": "Время расчета, сек",
                }
            )
            st.dataframe(
                _round_df(j_timing_ru.set_index("Регион (PVTNUM)")),
                use_container_width=True,
            )
    with c2:
        st.subheader("Метрики")
        qa_show = qa_df.copy()
        qa_show["PVTNUM_GDM"] = qa_show["PVTNUM_GDM"].astype(str)
        st.dataframe(_round_df(qa_show.set_index("PVTNUM_GDM")), use_container_width=True)
        global_j = qa_df[qa_df["PVTNUM_GDM"].astype(str) == "Все регионы"]
        if not global_j.empty and np.isfinite(global_j.iloc[0]["SCORE"]):
            st.metric("SCORE (все регионы)", f"{float(global_j.iloc[0]['SCORE']):.3f}")
        elif not qa_df.empty:
            reg_scores = qa_df.loc[qa_df["PVTNUM_GDM"].astype(str) != "Все регионы", "SCORE"]
            if len(reg_scores):
                st.metric("SCORE (среднее по регионам)", f"{float(reg_scores.mean()):.3f}")
    if j_total_elapsed is not None:
        st.caption(f"Общее время расчета J-функции: {float(j_total_elapsed):.2f} с")
    csv_result = result_df.to_csv(index=False).encode("utf-8")
    csv_params = params_df.to_csv(index=False).encode("utf-8")
    csv_qa = qa_df.to_csv(index=False).encode("utf-8")
    c3, c4, c5 = st.columns(3)
    c3.download_button("Скачать результат", data=csv_result, file_name="leverett_result.csv", mime="text/csv")
    c4.download_button("Скачать параметры", data=csv_params, file_name="leverett_params.csv", mime="text/csv")
    c5.download_button("Скачать метрики", data=csv_qa, file_name="leverett_metrics.csv", mime="text/csv")
    if st.button("Запомнить результаты по скважинам (J-функция)", key="save_j_snapshot"):
        ok, msg = _save_well_snapshot(result_df, model_col="Kng_model", method_tag="J")
        if ok:
            st.success(msg)
        else:
            st.warning(msg)

    if lab_ready and cloud is not None and not cloud.empty:
        st.subheader("J(Swn): лаборатория + степенная модель с оптимальными a, b")
        region = st.selectbox(
            "Регион для графика J–Swn",
            options=sorted(pd.to_numeric(params_df["PVTNUM_GDM"], errors="coerce").dropna().astype(int).unique()),
            key="jsw_region",
        )
        prow = params_df[params_df["PVTNUM_GDM"].astype(int) == int(region)]
        if not prow.empty:
            a_opt = float(prow.iloc[0]["a"])
            b_opt = float(prow.iloc[0]["b"])
            hs = (st.session_state.get("pvt_horizon_map") or {}).get(int(region)) or sorted(
                cloud["lab_horizon"].astype(str).unique().tolist()
            )
            sub = cloud[cloud["lab_horizon"].isin(hs)] if hs else cloud
            lab_fit = fit_power_j_swn(sub["Swn"].to_numpy(), sub["J_lab"].to_numpy())
            fig = _fig_j_swn_lab(
                sub.assign(lab_area=sub["lab_area"].astype(str)),
                title=f"PVT {region}: лаборатория и J = a·Swn^b (оптимальные a,b)",
                trend_fit=lab_fit,
                extra_lines=None,
                optimal={"a": a_opt, "b": b_opt},
                low_swn_threshold=float(
                    st.session_state.get("j_low_swn_threshold", DEFAULT_LOW_SWN_THRESHOLD)
                ),
                j_cap_at_low_swn=float(st.session_state.get("j_cap_at_low_swn", DEFAULT_J_CAP_AT_LOW_SWN)),
            )
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Интерактивные графики (Кнг)")
    pvt_series = pd.to_numeric(result_df["PVTNUM_GDM"], errors="coerce")
    region_options = sorted(pvt_series.dropna().astype(int).unique().tolist())
    if not region_options:
        st.warning("В результатах отсутствуют валидные регионы PVTNUM_GDM.")
        return

    region = st.selectbox("Регион (PVTNUM_GDM)", options=region_options)
    region_mask = pd.to_numeric(result_df["PVTNUM_GDM"], errors="coerce") == float(region)
    region_df_raw = result_df.loc[region_mask].copy()
    region_df = _filter_convergence_points(region_df_raw)

    if region_df.empty:
        st.warning("Для выбранного региона нет валидных точек сходимости (без нулей/выбросов Кнг_W).")
        return

    depth_col = _pick_depth_column(region_df)

    color_mode = st.radio(
        "Палитра точек на графике",
        options=("По весу", "По толщине"),
        horizontal=True,
        key="scatter_color_mode",
    )

    depth_for_thickness = depth_col
    if color_mode == "По толщине":
        if depth_for_thickness is None:
            st.warning("Колонка глубины не найдена, палитра автоматически переключена на веса.")
            color_mode = "По весу"
        elif "WELL_NAME" not in region_df.columns:
            st.warning("Нет колонки WELL_NAME для расчета толщин, палитра переключена на веса.")
            color_mode = "По весу"
        else:
            region_df[depth_for_thickness] = pd.to_numeric(region_df[depth_for_thickness], errors="coerce")
            thickness_df = (
                region_df.dropna(subset=[depth_for_thickness])
                .groupby("WELL_NAME")[depth_for_thickness]
                .agg(["min", "max"])
                .reset_index()
            )
            thickness_df["thickness"] = thickness_df["max"] - thickness_df["min"]
            region_df = region_df.merge(thickness_df[["WELL_NAME", "thickness"]], on="WELL_NAME", how="left")

    color_col = "weight"
    if color_mode == "По толщине" and "thickness" in region_df.columns:
        color_col = "thickness"

    hover_cols, hover_metrics = _j_kng_interactive_hover(region_df, depth_col)
    fig_scatter = px.scatter(
        region_df,
        x="Кнг_W",
        y="Kng_model",
        color=color_col if color_col in region_df.columns else None,
        color_continuous_scale="Viridis",
        hover_data=hover_cols if hover_cols else None,
        title=f"PVT {region}: предсказанное Кнг(историческое) ({'вес' if color_col == 'weight' else 'толщина'})",
        opacity=0.7,
        **_crossplot_hover_name_kw(region_df),
    )
    fig_scatter.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(color="red", dash="dash"))
    _apply_crossplot_hover(fig_scatter, hover_metrics, region_df)
    fig_scatter.update_layout(xaxis_title="Кнг историческое (ГИС)", yaxis_title="Кнг предсказанное")
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.subheader("Кроссплот по скважинам (средневзвешенные значения)")
    if "WELL_NAME" in region_df.columns:
        region_cross = _exclude_clipped_kng_zeros(region_df)
        cross_df = _well_weighted_crossplot_df(region_cross)
        if cross_df.empty:
            st.info("Недостаточно данных для кроссплота по скважинам.")
        else:
            fig_well_cross = px.scatter(
                cross_df,
                x="Кнг_W_wmean",
                y="Kng_model_wmean",
                color="convergence_percent",
                color_continuous_scale="Turbo",
                hover_data={
                    "points": ":.3f",
                    "avg_weight": ":.3f",
                    "convergence_percent": ":.3f",
                },
                title=f"PVT {region}: кроссплот по скважинам (средневзвешенно)",
                opacity=0.85,
                **_crossplot_hover_name_kw(cross_df),
            )
            _apply_crossplot_hover(
                fig_well_cross, "Кнг_W_wmean=%{x:.3f}<br>Kng_model_wmean=%{y:.3f}", cross_df
            )
            fig_well_cross.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(color="red", dash="dash"))
            fig_well_cross.update_layout(
                xaxis_title="Кнг_W (средневзвеш.)",
                yaxis_title="Kng_model (средневзвеш.)",
            )
            st.plotly_chart(fig_well_cross, use_container_width=True)
            st.markdown("#### Невязка по кроссплоту (скважины)")
            cross_all_j = _well_crossplot_table_from_result(
                _exclude_clipped_kng_zeros(result_df), "Кнг_W", "Kng_model"
            )
            _render_well_crossplot_qa_panel(
                cross_all_j,
                x_col="Кнг_W_wmean",
                y_col="Kng_model_wmean",
                region_col="Регион",
                block_key=f"j_all_pvt",
                method_label="J-функция",
                show_metrics_help=True,
            )

    st.markdown(f"#### Аналитика по выбранному региону (PVTNUM={region})")
    region_wells_count = region_df["WELL_NAME"].astype(str).nunique() if "WELL_NAME" in region_df.columns else 0
    region_points_count = len(region_df)
    m1, m2 = st.columns(2)
    m1.metric(f"Уникальных скважин (PVTNUM={region})", int(region_wells_count))
    m2.metric(f"Всего точек (PVTNUM={region})", int(region_points_count))

    if "weight" in result_df.columns and "WELL_NAME" in result_df.columns:
        weight_summary = (
            result_df.groupby("WELL_NAME", as_index=False)
            .agg(avg_weight=("weight", "mean"), max_weight=("weight", "max"), points=("weight", "size"))
            .sort_values(["avg_weight", "max_weight"], ascending=False)
        )
        st.markdown("**Скважины с наибольшими весами (в целом по выборке)**")
        weight_summary_ru = weight_summary.rename(
            columns={
                "WELL_NAME": "Скважина",
                "avg_weight": "Средний вес",
                "max_weight": "Максимальный вес",
                "points": "Количество точек",
            }
        )
        st.dataframe(_round_df(weight_summary_ru.head(15)), use_container_width=True)

    if not qa_df.empty:
        qa_row = qa_df[qa_df["PVTNUM_GDM"] == int(region)]
        if not qa_row.empty:
            qa_row = qa_row.iloc[0]
            recs = []
            if qa_row["R2"] < 0.5:
                recs.append("Низкий R2: сузьте диапазоны a/b или увеличьте maxiter/popsize.")
            bias = float(qa_row["BIAS"])
            if abs(bias) > 0.08:
                if bias > 0:
                    recs.append("BIAS > 0 (модель завышает Кн): попробуйте повысить sigma или немного снизить a.")
                else:
                    recs.append("BIAS < 0 (модель занижает Кн): попробуйте понизить sigma или немного повысить a.")
            if qa_row["RMSE"] > 0.15:
                recs.append("Повышенный RMSE: проверьте выбросы PC/PERM/PORO и очистку данных.")
            if qa_row["SCORE"] < 0.75:
                recs.append("Низкий SCORE: проверьте соответствие горизонтов PVTNUM и пересмотрите границы a/b.")
            if not recs:
                recs.append("Качество выглядит стабильным; при автограницах можно сузить горизонты в лаборатории.")
            st.markdown("**Рекомендации**")
            for rec in recs:
                st.write(f"- {rec}")

    if "WELL_NAME" in region_df.columns:
        wells = sorted(region_df["WELL_NAME"].astype(str).unique().tolist())
        st.subheader("Просмотр скважины")
        well = _shared_well_selectbox(wells)
        well_df = region_df[region_df["WELL_NAME"].astype(str) == well].copy()
        well_depth_col = depth_col or _pick_depth_column(well_df)
        if well_depth_col is None:
            st.warning("Не найдена колонка глубины (например DEPTH/DEPT).")
        elif "ACTNUM_GDM" not in well_df.columns:
            st.warning("В данных отсутствует ACTNUM_GDM для детального графика.")
        else:
            well_df[well_depth_col] = pd.to_numeric(well_df[well_depth_col], errors="coerce")
            well_df = well_df.dropna(subset=[well_depth_col]).sort_values(well_depth_col).reset_index(drop=True)
            curve_cols = [well_depth_col, "ACTNUM_GDM", "Кнг_W", "Kng_model"]
            if "FWL_GDM" in well_df.columns:
                curve_cols.insert(1, "FWL_GDM")
            curve_df = well_df[curve_cols].copy()
            curve_df = curve_df.rename(columns={"Кнг_W": "Кн РИГИС", "Kng_model": "Кн J-функция"})
            id_vars = [well_depth_col]
            if "FWL_GDM" in curve_df.columns:
                id_vars.append("FWL_GDM")
            chart_df = curve_df.melt(
                id_vars=id_vars,
                value_vars=["ACTNUM_GDM", "Кн РИГИС", "Кн J-функция"],
                var_name="Кривая",
                value_name="Значение",
            )
            fig_well = px.line(
                chart_df,
                x="Значение",
                y=well_depth_col,
                color="Кривая",
                title=f"Скважина {well}: вертикальный профиль",
            )
            fig_well.update_traces(mode="lines")
            if "FWL_GDM" in chart_df.columns:
                fig_well.update_traces(
                    hovertemplate=(
                        "Значение=%{x:.3f}<br>"
                        f"Глубина=%{{y:.3f}}<br>"
                        "FWL=%{customdata[0]:.3f}<br>"
                        "Кривая=%{fullData.name}<extra></extra>"
                    )
                )
            else:
                fig_well.update_traces(
                    hovertemplate=(
                        "Значение=%{x:.3f}<br>"
                        f"Глубина=%{{y:.3f}}<br>"
                        "Кривая=%{fullData.name}<extra></extra>"
                    )
                )
            fig_well.update_yaxes(autorange="reversed")
            fig_well.update_layout(xaxis_title="Значение", yaxis_title="Глубина")
            st.plotly_chart(fig_well, use_container_width=True)

            conv_percent = _well_convergence_percent_weighted(well_df)
            if np.isfinite(conv_percent):
                st.metric("Сходимость для скважины (средневзвеш.), %", f"{conv_percent:.2f}")


def brooks_corey_tab() -> None:
    st.title("Метод Брукса-Кори")
    _scroll_to_top_if_pending()

    bc_lab_df, bc_source = _get_bc_source_df()
    if bc_lab_df is None or bc_lab_df.empty:
        st.info(
            "Нет данных для Брукса-Кори: во вкладке «Лаборатория» вверху страницы загрузите таблицу для БК "
            "или настройте облако J(Swn) и примените фильтр, либо убедитесь, что доступна БД ККД."
        )
        _scroll_to_top_if_pending(finish=True)
        return
    st.caption(f"Источник лабораторных данных БК: {bc_source}")

    # Для ККД — фильтр из «Лаборатория». Для своего файла БК — только он и фильтры ниже по колонкам файла.
    use_lab_meta_filter = bc_source != "пользовательский файл БК"
    lab_meta = st.session_state.get("lab_meta") or {}
    area_col_meta = lab_meta.get("area_col")
    horizon_col_meta = lab_meta.get("horizon_col")
    selected_areas = lab_meta.get("areas") or []
    selected_horizons = lab_meta.get("horizons") or []
    if use_lab_meta_filter:
        if not horizon_col_meta or not selected_horizons:
            st.warning(
                "Для расчета Брукса-Кори сначала во вкладке «Лаборатория» выберите горизонты "
                "и нажмите «Применить фильтр и сохранить выбор…»."
            )
            return
        if horizon_col_meta not in bc_lab_df.columns:
            st.error(
                "В лабораторном источнике БК нет колонки горизонта из последнего выбора во вкладке «Лаборатория». "
                "Переоткройте «Лаборатория», выберите корректные колонки и сохраните фильтр."
            )
            return
        bc_lab_df = bc_lab_df.copy()
        bc_lab_df[horizon_col_meta] = bc_lab_df[horizon_col_meta].astype(str).str.strip()
        hmask = bc_lab_df[horizon_col_meta].isin([str(x).strip() for x in selected_horizons])
        if area_col_meta and area_col_meta in bc_lab_df.columns and selected_areas:
            bc_lab_df[area_col_meta] = bc_lab_df[area_col_meta].astype(str).str.strip()
            amask = bc_lab_df[area_col_meta].isin([str(x).strip() for x in selected_areas])
            bc_lab_df = bc_lab_df[hmask & amask].copy()
            st.caption("Для БК применен фильтр: горизонты + площади (из Лаборатории).")
        else:
            bc_lab_df = bc_lab_df[hmask].copy()
            st.caption("Для БК применен фильтр только по горизонтам (из Лаборатории).")
    if bc_lab_df.empty:
        st.error("После применения фильтра из вкладки «Лаборатория» не осталось данных для Брукса-Кори.")
        return
    st.caption(f"Для БК используется отфильтрованная лабораторная выборка: {len(bc_lab_df)} строк.")

    with st.sidebar:
        st.header("Геологическая модель для Брукса-Кори")
        st.number_input(
            "Макс. проницаемость Кпр, мД (лаборатория и модель)",
            min_value=10.0,
            max_value=100_000.0,
            value=float(st.session_state.get("bc_perm_cap", 5000.0)),
            step=100.0,
            key="bc_perm_cap",
            help="Значения проницаемости обрезаются сверху при подготовке лабораторных данных и при расчёте Кпр.",
        )
        wells_file = st.file_uploader(
            "Файл скважин (для БК)", type=["csv", "xlsx", "xls", "txt"], key=_wells_file_uploader_key()
        )
        prod_file = st.file_uploader(
            "Файл добычи (для весов БК, опционально)", type=["csv", "xlsx", "xls"], key=_prod_file_uploader_key()
        )
        bcw1, bcw2 = st.columns(2)
        if bcw1.button("Сбросить скважины", key="bc_clr_wells"):
            _clear_shared_wells_files()
            st.rerun()
        if bcw2.button("Сбросить добычу", key="bc_clr_prod"):
            _clear_shared_prod_files()
            st.rerun()
        maxiter = st.slider("Итерации БК", min_value=50, max_value=350, value=200, step=10, key="bc_maxiter")
        popsize = st.slider("Популяция БК", min_value=10, max_value=40, value=20, step=1, key="bc_popsize")
        bc_optimizer_method = st.selectbox(
            "Метод оптимизации БК",
            options=["differential_evolution", "pso", "dual_annealing"],
            format_func=lambda x: {
                "differential_evolution": "Дифференциальная эволюция",
                "pso": "Рой частиц (PSO)",
                "dual_annealing": "Dual Annealing",
            }[x],
            index=0,
            key="bc_optimizer_method",
        )
        use_perf_weights = st.checkbox(
            "Учитывать перфорации (Perf_GDM) в весах БК",
            value=False,
            key="bc_use_perf_weights",
            help="Если выключено, колонка Perf_GDM игнорируется при расчете весов БК.",
        )

    if wells_file is not None:
        p = _persist_uploaded_file(wells_file, "bc_wells")
        st.session_state["bc_wells_path"] = p
        st.session_state["shared_wells_file_path"] = p
    if prod_file is not None:
        p = _persist_uploaded_file(prod_file, "bc_prod")
        st.session_state["bc_prod_path"] = p
        st.session_state["shared_prod_file_path"] = p
    wells_path = _coalesce_path("shared_wells_file_path", "bc_wells_path", "wells_file_path")
    if not wells_path:
        st.info("Загрузите файл скважин для расчета Брукса-Кори.")
        return

    try:
        raw_geo = _read_table_from_path(wells_path)
        prod_path = _coalesce_path("shared_prod_file_path", "bc_prod_path", "prod_file_path")
        raw_prod = _read_table_from_path(prod_path) if prod_path else None
        mapped_geo = _map_uploaded_wells_df(raw_geo, key_prefix="bc", title="Сопоставление колонок файла скважин (БК)")
        if mapped_geo.empty:
            return
        mapped_prod = _map_uploaded_prod_df(raw_prod, key_prefix="bc", title="Сопоставление колонок файла добычи (БК)")
        if isinstance(mapped_prod, pd.DataFrame) and mapped_prod.empty:
            return
        if "FWL_GDM" in mapped_geo.columns:
            fwl_settings = _fwl_filter_settings_ui(mapped_geo, key_prefix="shared")
        else:
            fwl_settings = {"has_fwl": False, "mode": None, "min_continuous": DEFAULT_FWL_MIN_CONTINUOUS, "exclude_discrete": ()}
        df_geo_raw = _clean_wells_cached(
            mapped_geo,
            fwl_mode=fwl_settings["mode"] if fwl_settings["has_fwl"] else None,
            fwl_min=float(fwl_settings["min_continuous"]),
            fwl_exclude=tuple(fwl_settings["exclude_discrete"]),
        )
        if (not use_perf_weights) and ("Perf_GDM" in df_geo_raw.columns):
            df_geo_raw = df_geo_raw.drop(columns=["Perf_GDM"])
        df_prod = _clean_prod_cached(mapped_prod)
        df_geo_raw["PVTNUM_GDM"] = pd.to_numeric(df_geo_raw["PVTNUM_GDM"], errors="coerce")
        pvts_bc = sorted(df_geo_raw["PVTNUM_GDM"].dropna().astype(int).unique().tolist())
        df_prod_m = df_prod
        if (
            df_prod is not None
            and not df_prod.empty
            and "PVTNUM_GDM" in df_prod.columns
            and pvts_bc
        ):
            df_prod_m = _apply_prod_pvt_mapping_ui(
                df_prod,
                pvts_bc,
                key_prefix="shared",
                prod_path=str(prod_path) if prod_path else "",
                title="Соответствие кода объекта в добыче → PVTNUM (БК, веса)",
            )
        df_geo = prepare_brooks_training_data(df_geo_raw, df_prod_m)
    except Exception as e:
        st.error(f"Ошибка чтения геологической модели: {e}")
        return
    st.caption(
        f"Выбранный файл скважин: `{Path(wells_path).name}`"
        + (f" | Выбранный файл добычи: `{Path(prod_path).name}`" if prod_path else " | Выбранный файл добычи: не выбран")
    )

    required = {"PORO_GDM", "PC", "Кнг_W", "PVTNUM_GDM"}
    missing = [c for c in required if c not in df_geo.columns]
    if missing:
        st.error(f"В файле геомодели отсутствуют колонки: {missing}")
        return

    if "weight" not in df_geo.columns:
        df_geo["weight"] = 1.0

    st.subheader("Сопоставление колонок лабораторной таблицы Брукса-Кори")
    st.caption("Предпросмотр исходной лабораторной таблицы (первые строки):")
    cols = list(bc_lab_df.columns)
    saved_lab_map = st.session_state.get("bc_lab_col_map", {})
    defaults_lab = {
        "Код горизонта": saved_lab_map.get("Код горизонта") if saved_lab_map.get("Код горизонта") in cols else _safe_guess_col(cols, ["ГОРИЗ", "HORIZ", "КОД"]),
        "Poro (лаборатория)": saved_lab_map.get("Poro (лаборатория)") if saved_lab_map.get("Poro (лаборатория)") in cols else _safe_guess_col(cols, ["КП", "PORO", "ПОРИСТ"]),
        "Swi/Swl (лаборатория)": saved_lab_map.get("Swi/Swl (лаборатория)") if saved_lab_map.get("Swi/Swl (лаборатория)") in cols else _safe_guess_col(cols, ["КВО", "SWL", "SWI"]),
        "Perm (лаборатория)": saved_lab_map.get("Perm (лаборатория)") if saved_lab_map.get("Perm (лаборатория)") in cols else _safe_guess_col(cols, ["КПР", "ПРОНИЦАЕМОСТ", "PERM"]),
        "Pvit (лаборатория)": saved_lab_map.get("Pvit (лаборатория)") if saved_lab_map.get("Pvit (лаборатория)") in cols else _safe_guess_col(cols, ["РВЫТ", "PVIT"]),
        "n (лаборатория)": (
            saved_lab_map.get("n (лаборатория)")
            if saved_lab_map.get("n (лаборатория)") in cols
            else _guess_bc_lab_n_column(cols)
        ),
    }
    lab_map_df = pd.DataFrame([defaults_lab])
    lab_cfg = {
        k: st.column_config.SelectboxColumn(label=k, options=cols, required=True) for k in defaults_lab.keys()
    }
    lab_edit = st.data_editor(
        lab_map_df,
        hide_index=True,
        num_rows="fixed",
        use_container_width=True,
        key="bc_lab_mapping_table",
        column_config=lab_cfg,
    )
    st.dataframe(_round_df(bc_lab_df.head(12)), use_container_width=True)
    horizon_col = str(lab_edit.iloc[0]["Код горизонта"])
    poro_col = str(lab_edit.iloc[0]["Poro (лаборатория)"])
    swl_col = str(lab_edit.iloc[0]["Swi/Swl (лаборатория)"])
    perm_col = str(lab_edit.iloc[0]["Perm (лаборатория)"])
    pvit_col = str(lab_edit.iloc[0]["Pvit (лаборатория)"])
    n_col = str(lab_edit.iloc[0]["n (лаборатория)"])
    st.session_state["bc_lab_col_map"] = {
        "Код горизонта": horizon_col,
        "Poro (лаборатория)": poro_col,
        "Swi/Swl (лаборатория)": swl_col,
        "Perm (лаборатория)": perm_col,
        "Pvit (лаборатория)": pvit_col,
        "n (лаборатория)": n_col,
    }
    st.caption(
        f"В расчетах БК используются выбранные столбцы: горизонт=`{horizon_col}`, poro=`{poro_col}`, "
        f"swl=`{swl_col}`, perm=`{perm_col}`, pvit=`{pvit_col}`, n=`{n_col}`."
    )
    swl_dbg = pd.to_numeric(bc_lab_df[swl_col], errors="coerce")
    swl_zero_share = float((swl_dbg == 0).mean()) if swl_dbg.notna().sum() > 0 else np.nan
    n_sw = int(swl_dbg.notna().sum())
    if n_sw:
        bad_gt1 = int((swl_dbg > 1.0).sum())
        bad_nonpos = int((swl_dbg <= 0).sum())
        if bad_gt1 or bad_nonpos:
            st.warning(
                "Столбец **Swi/Swl (лаборатория)** должен содержать доли связной воды в интервале **(0; 1]**. "
                f"Сейчас: значений **> 1**: {bad_gt1}, **≤ 0** (включая нули): {bad_nonpos} из {n_sw} числовых ячеек. "
                "Проверьте сопоставление столбца и единицы (не проценты), при необходимости пересчитайте в доли."
            )
    d1, d2, d3 = st.columns(3)
    d1.metric("swl min", f"{swl_dbg.min():.4g}" if swl_dbg.notna().sum() else "nan")
    d2.metric("swl max", f"{swl_dbg.max():.4g}" if swl_dbg.notna().sum() else "nan")
    d3.metric("доля нулей swl", f"{100*swl_zero_share:.2f}%" if np.isfinite(swl_zero_share) else "nan")
    if np.isfinite(swl_zero_share) and swl_zero_share > 0.25:
        st.warning(
            "У выбранного swl-столбца высокая доля нулей. Проверьте, что выбран корректный столбец "
            "(часто путают `Sw_min` с фактической водонасыщенностью образца)."
        )
    if not use_lab_meta_filter:
        st.caption("Для вашего файла БК используются только его данные; фильтры — по колонкам этого файла.")
        _bc_urev = int(st.session_state.get("bc_user_upload_rev", 0))
        horizons_bc = sorted(pd.Series(bc_lab_df[horizon_col]).dropna().astype(str).str.strip().unique().tolist())
        sel_h_bc = st.multiselect(
            "Горизонты для БК (из загруженного файла)",
            options=horizons_bc,
            default=horizons_bc,
            key=f"bc_self_horizons_{_bc_urev}",
        )
        # Фильтр по площади только из колонки с точным именем «Площадь» (как в шаблоне файла)
        area_col_name = "Площадь"
        area_ok = area_col_name in bc_lab_df.columns and horizon_col != area_col_name
        sel_a_bc: list[str] = []
        if area_ok:
            areas_bc = sorted(
                pd.Series(bc_lab_df[area_col_name]).dropna().astype(str).str.strip().unique().tolist()
            )
            sel_a_bc = st.multiselect(
                "Площади для БК (колонка «Площадь» в вашем файле; опционально)",
                options=areas_bc,
                default=areas_bc,
                key=f"bc_self_areas_{_bc_urev}",
            )
        elif "Площадь" not in bc_lab_df.columns:
            st.caption("В файле нет колонки «Площадь» — фильтр только по горизонтам.")
        bc_lab_df[horizon_col] = bc_lab_df[horizon_col].astype(str).str.strip()
        hmask_self = bc_lab_df[horizon_col].isin([str(x).strip() for x in sel_h_bc]) if sel_h_bc else pd.Series(False, index=bc_lab_df.index)
        if area_ok and sel_a_bc:
            bc_lab_df[area_col_name] = bc_lab_df[area_col_name].astype(str).str.strip()
            amask_self = bc_lab_df[area_col_name].isin([str(x).strip() for x in sel_a_bc])
            bc_lab_df = bc_lab_df[hmask_self & amask_self].copy()
        else:
            bc_lab_df = bc_lab_df[hmask_self].copy()
        if bc_lab_df.empty:
            st.error("После фильтрации вашего файла БК не осталось данных.")
            return

    lab = bc_lab_df.copy()
    lab = lab.rename(
        columns={
            horizon_col: "HORIZON",
            poro_col: "PORO_LAB",
            swl_col: "SWL_LAB",
            perm_col: "PERM_LAB",
            pvit_col: "PVIT_LAB",
            n_col: "N_LAB",
        }
    )
    for c in ["PORO_LAB", "SWL_LAB", "PERM_LAB", "PVIT_LAB", "N_LAB"]:
        lab[c] = pd.to_numeric(lab[c], errors="coerce")
    # Пористость в долях (<1) для всех формул
    lab["PORO_LAB_FRAC"] = np.where(lab["PORO_LAB"] > 1, lab["PORO_LAB"] / 100.0, lab["PORO_LAB"])
    perm_cap = float(st.session_state.get("bc_perm_cap", 5000.0))
    if np.isfinite(perm_cap) and perm_cap > 0:
        lab["PERM_LAB"] = np.clip(lab["PERM_LAB"], 0.0, perm_cap)
    lab["perm_poro"] = np.sqrt(lab["PERM_LAB"] / lab["PORO_LAB_FRAC"].replace(0, np.nan))
    lab = lab.dropna(subset=["PORO_LAB_FRAC", "SWL_LAB", "PERM_LAB", "PVIT_LAB", "N_LAB", "perm_poro"])
    lab = lab[
        (lab["PORO_LAB_FRAC"] > 0)
        & (lab["SWL_LAB"] > 0)
        & (lab["PERM_LAB"] > 0)
        & (lab["PVIT_LAB"] > 0)
        & (lab["N_LAB"] > 0)
        & (lab["perm_poro"] > 0)
    ]
    # Для БК используем только одну строку на каждый образец каждой скважины
    dedup_keys = [c for c in ["Скважина", "Номер образца", "Порядковый номер образца"] if c in lab.columns]
    if dedup_keys:
        lab = lab.sort_values(dedup_keys).drop_duplicates(subset=dedup_keys, keep="first").reset_index(drop=True)
    lab["HORIZON"] = lab["HORIZON"].astype(str).str.strip()
    preview_cols = [c for c in ["Скважина", "Номер образца", "Порядковый номер образца", "HORIZON", "PORO_LAB_FRAC", "SWL_LAB", "PERM_LAB", "perm_poro", "PVIT_LAB", "N_LAB"] if c in lab.columns]
    with st.expander("Подготовленная таблица для БК (первые 20 строк)", expanded=False):
        st.dataframe(_round_df(lab[preview_cols].head(20)), use_container_width=True)

    pvts = sorted(pd.to_numeric(df_geo["PVTNUM_GDM"], errors="coerce").dropna().astype(int).unique().tolist())
    _sync_pvt_session_state(str(wells_path), pvts)
    hors_universe = sorted(lab["HORIZON"].astype(str).unique().tolist())
    st.session_state["bc_horizon_map_ctx"] = {"pvts": pvts, "hors_universe": hors_universe}
    _bc_pvt_horizon_mapping_run()
    pvt_horizon_map = _pvt_horizon_map_from_session(pvts)

    maxiter = int(st.session_state.get("bc_maxiter", 200))
    popsize = int(st.session_state.get("bc_popsize", 20))
    bc_optimizer_method = str(st.session_state.get("bc_optimizer_method", "differential_evolution"))
    use_perf_weights = bool(st.session_state.get("bc_use_perf_weights", False))

    use_manual_bc = st.checkbox("Использовать свои коэффициенты для 4 зависимостей (без оптимизации)", value=False, key="bc_manual_mode")
    manual_params_by_pvt: dict[int, dict[str, float]] = {}
    if use_manual_bc:
        st.subheader("Ввод своих коэффициентов (по каждому региону)")
        for p in pvts:
            with st.expander(f"PVTNUM {p}: коэффициенты", expanded=False):
                c1, c2 = st.columns(2)
                with c1:
                    a_swl = st.number_input(
                        f"a (swl=a·exp(b·Кп)) | PVT {p}",
                        value=0.15,
                        min_value=1e-6,
                        max_value=25.0,
                        format="%.6f",
                        key=f"man_a_swl_{p}",
                    )
                    a_perm = st.number_input(f"a_perm | PVT {p}", value=1.0, format="%.6f", key=f"man_a_perm_{p}")
                    a_pvit = st.number_input(f"a_pvit | PVT {p}", value=1.0, format="%.6f", key=f"man_a_pvit_{p}")
                    a_n = st.number_input(f"a_n | PVT {p}", value=1.0, format="%.6f", key=f"man_a_n_{p}")
                with c2:
                    b_swl = st.number_input(
                        f"b (swl=a·exp(b·Кп)) | PVT {p}",
                        value=-0.5,
                        min_value=-80.0,
                        max_value=80.0,
                        format="%.6f",
                        key=f"man_b_swl_{p}",
                    )
                    b_perm = st.number_input(f"b_perm | PVT {p}", value=-0.5, format="%.6f", key=f"man_b_perm_{p}")
                    b_pvit = st.number_input(f"b_pvit | PVT {p}", value=-0.5, format="%.6f", key=f"man_b_pvit_{p}")
                    b_n = st.number_input(f"b_n | PVT {p}", value=-0.5, format="%.6f", key=f"man_b_n_{p}")
                manual_params_by_pvt[p] = {
                    "a_swl": float(a_swl),
                    "b_swl": float(b_swl),
                    "a_perm": float(a_perm),
                    "b_perm": float(b_perm),
                    "a_pvit": float(a_pvit),
                    "b_pvit": float(b_pvit),
                    "a_n": float(a_n),
                    "b_n": float(b_n),
                }

        st.subheader("Предпросмотр зависимостей по введенным коэффициентам")
        if st.button("Показать / обновить предпросмотр", key="bc_manual_preview_btn"):
            st.session_state["bc_manual_preview_on"] = True
        if st.session_state.get("bc_manual_preview_on"):
            p_preview = st.selectbox("Регион для предпросмотра ручных коэффициентов", options=pvts, key="bc_manual_preview_pvt")
            hs = pvt_horizon_map.get(p_preview, [])
            if not hs:
                st.info(f"Для PVTNUM {p_preview} не выбраны горизонты.")
            else:
                lsub = lab[lab["HORIZON"].isin(hs)].copy()
                if lsub.empty:
                    st.info(f"Для PVTNUM {p_preview} нет лабораторных точек после фильтра.")
                else:
                    prm = manual_params_by_pvt[p_preview]
                    fig1 = _plot_bc_cloud(
                        lsub,
                        "PORO_LAB_FRAC",
                        "SWL_LAB",
                        f"PVT {p_preview}: swl=a·exp(b·Кп) — ручные коэффициенты",
                        opt_ab=(prm["a_swl"], prm["b_swl"]),
                        curve_kind="swl_exp_ab",
                    )
                    fig2 = _plot_bc_cloud(
                        lsub,
                        "SWL_LAB",
                        "PERM_LAB",
                        f"PVT {p_preview}: Кпр(Кво) — ручные коэффициенты",
                        opt_ab=(prm["a_perm"], prm["b_perm"]),
                    )
                    fig3 = _plot_bc_cloud(
                        lsub,
                        "perm_poro",
                        "PVIT_LAB",
                        f"PVT {p_preview}: pvit(perm_poro) — ручные коэффициенты",
                        opt_ab=(prm["a_pvit"], prm["b_pvit"]),
                    )
                    fig4 = _plot_bc_cloud(
                        lsub,
                        "perm_poro",
                        "N_LAB",
                        f"PVT {p_preview}: n(perm_poro) — ручные коэффициенты",
                        opt_ab=(prm["a_n"], prm["b_n"]),
                    )
                    c1, c2 = st.columns(2)
                    c1.plotly_chart(fig1, use_container_width=True)
                    c2.plotly_chart(fig2, use_container_width=True)
                    c3, c4 = st.columns(2)
                    c3.plotly_chart(fig3, use_container_width=True)
                    c4.plotly_chart(fig4, use_container_width=True)

    bc_busy = bool(st.session_state.get("bc_busy", False))
    if bc_busy:
        st.warning("Идет расчет Брукса-Кори... Пожалуйста, дождитесь завершения.")
    if st.button("Рассчитать Брукса-Кори", type="primary", disabled=bc_busy):
        pvt_horizon_map = _pvt_horizon_map_from_session(pvts)
        if any(not (pvt_horizon_map.get(p) or []) for p in pvts):
            st.error("Выберите горизонты для каждого PVTNUM.")
            return

        st.session_state["bc_busy"] = True
        _ui_lock(True, "ui_lock_bc")
        perm_cap_run = float(st.session_state.get("bc_perm_cap", 5000.0))

        results = []
        params_rows = []
        bc_meta = {}
        timing_rows = []
        t0_total = time.perf_counter()
        for p in pvts:
            t0_pvt = time.perf_counter()
            g = df_geo[pd.to_numeric(df_geo["PVTNUM_GDM"], errors="coerce") == float(p)].copy()
            h = pvt_horizon_map[p]
            lsub = lab[lab["HORIZON"].astype(str).isin([str(x) for x in h])].copy()
            if g.empty or lsub.empty:
                continue
            swl_exp_info = auto_exp_bounds_swl_poro(lsub["PORO_LAB_FRAC"].to_numpy(), lsub["SWL_LAB"].to_numpy())
            perm_info = auto_power_bounds(lsub["SWL_LAB"].to_numpy(), lsub["PERM_LAB"].to_numpy())
            pvit_info = auto_power_bounds(lsub["perm_poro"].to_numpy(), lsub["PVIT_LAB"].to_numpy())
            n_info = auto_power_bounds(lsub["perm_poro"].to_numpy(), lsub["N_LAB"].to_numpy())
            b_swl = swl_exp_info["bounds"]; b_perm = perm_info["bounds"]; b_pvit = pvit_info["bounds"]; b_n = n_info["bounds"]
            bounds = {"swl": b_swl, "perm": b_perm, "pvit": b_pvit, "n": b_n}
            envelopes = {
                "swl": {"kind": "exp_ab", "lower": swl_exp_info["lower"], "upper": swl_exp_info["upper"], "x": lsub["PORO_LAB_FRAC"].to_numpy(dtype=float)},
                "perm": {"lower": perm_info["lower"], "upper": perm_info["upper"], "x": lsub["SWL_LAB"].to_numpy(dtype=float)},
                "pvit": {"lower": pvit_info["lower"], "upper": pvit_info["upper"], "x": lsub["perm_poro"].to_numpy(dtype=float)},
                "n": {"lower": n_info["lower"], "upper": n_info["upper"], "x": lsub["perm_poro"].to_numpy(dtype=float)},
            }
            if use_manual_bc:
                params = {**manual_params_by_pvt.get(p, {}), "perm_max_md": perm_cap_run}
            else:
                baseline = {"a_swl": 0.15, "b_swl": -0.5, "a_perm": 1.0, "b_perm": -0.5, "a_pvit": 1.0, "b_pvit": -0.5, "a_n": 1.0, "b_n": -0.5, "perm_max_md": perm_cap_run}
                corr = {"a_swl": float(swl_exp_info.get("center")[0]), "b_swl": float(swl_exp_info.get("center")[1]), "a_perm": float(perm_info.get("center")[0]), "b_perm": float(perm_info.get("center")[1]), "a_pvit": float(pvit_info.get("center")[0]), "b_pvit": float(pvit_info.get("center")[1]), "a_n": float(n_info.get("center")[0]), "b_n": float(n_info.get("center")[1]), "perm_max_md": perm_cap_run}
                params = optimize_brooks_corey_for_region(
                    g,
                    bounds=bounds,
                    envelopes=envelopes,
                    maxiter=maxiter,
                    popsize=popsize,
                    initial_guess={
                        "swl": swl_exp_info.get("center"),
                        "perm": perm_info.get("center"),
                        "pvit": pvit_info.get("center"),
                        "n": n_info.get("center"),
                    },
                    baseline_params=baseline,
                    perm_max_md=perm_cap_run,
                    optimizer_method=bc_optimizer_method,
                )
                best_params, best_score = None, -np.inf
                for _, cp in [("auto", params), ("corr", corr), ("default", baseline)]:
                    cp = {**cp, "perm_max_md": perm_cap_run}
                    if envelope_max_violation(cp, envelopes) > 1e-9:
                        continue
                    score = evaluate_brooks_score(g, cp)
                    if score > best_score:
                        best_params, best_score = cp, score
                if best_params is not None:
                    params = best_params
            elapsed = float(time.perf_counter() - t0_pvt)
            if not params:
                continue
            g["Kng_BC_model"] = compute_soil_from_params(g, params)
            results.append(g)
            params_rows.append({"PVTNUM_GDM": p, **params})
            timing_rows.append({"PVTNUM_GDM": int(p), "rows_geo": int(len(g)), "rows_lab": int(len(lsub)), "elapsed_sec": elapsed})
            bc_meta[p] = {"lab": lsub.copy(), "bounds": bounds, "perm_max_md": perm_cap_run, "centers": {"swl": swl_exp_info.get("center"), "perm": perm_info.get("center"), "pvit": pvit_info.get("center"), "n": n_info.get("center")}, "envelopes": envelopes}
        if not results:
            st.session_state["bc_busy"] = False
            _ui_lock(False, "ui_lock_bc")
            st.error("Не удалось рассчитать параметры Брукса-Кори.")
            return
        bc_res = pd.concat(results, ignore_index=True)
        bc_params = pd.DataFrame(params_rows)
        bc_qa = _compute_qa_metrics(bc_res, "Кнг_W", "Kng_BC_model")
        total_elapsed = float(time.perf_counter() - t0_total)
        bc_timing = _timing_table_with_total(timing_rows, total_elapsed=total_elapsed)
        st.session_state["bc_result_df"] = bc_res
        st.session_state["bc_params_df"] = bc_params
        st.session_state["bc_qa_df"] = bc_qa
        st.session_state["bc_timing_df"] = bc_timing
        st.session_state["bc_total_elapsed_sec"] = total_elapsed
        st.session_state["bc_meta"] = bc_meta
        st.session_state["bc_busy"] = False
        _ui_lock(False, "ui_lock_bc")
        st.success("Расчет Брукса-Кори завершен.")

    bc_res = st.session_state.get("bc_result_df")
    bc_params = st.session_state.get("bc_params_df")
    if not isinstance(bc_res, pd.DataFrame) or not isinstance(bc_params, pd.DataFrame) or bc_res.empty:
        return

    _bc_results_dashboard_run()
    _bc_well_preview_run()
    if not st.session_state.get("_scroll_to_top_pending"):
        restore_bc_scroll = bool(st.session_state.get("_bc_scroll_restore", True))
        _preserve_scroll_position(restore=restore_bc_scroll)
        if not restore_bc_scroll:
            st.session_state["_bc_scroll_restore"] = True


def compare_methods_tab() -> None:
    st.title("Сравнение методов")
    _scroll_to_top_if_pending()
    st.caption(
        "Сначала сохраните результаты по скважинам в вкладках J-функции и Брукса-Кори "
        "(кнопка «Запомнить»). Снимки можно записать в CSV и загрузить после перезапуска приложения."
    )
    _render_snapshot_disk_panel(key_prefix="compare_tab_snap")
    st.divider()
    _render_methods_comparison_block(block_key="compare_tab")


def main() -> None:
    page = st.sidebar.radio("Раздел", options=["Лаборатория", "Подбор J функции Леверетта", "Брукса-Кори", "Сравнение методов"])
    prev_page = st.session_state.get("_active_page")
    if prev_page != page:
        _mark_scroll_to_top_pending()
        _clear_preserved_scroll(SCROLL_STORAGE_BC)
        st.session_state["_bc_scroll_restore"] = False
        _scroll_page_top()
        # Привязки горизонт→PVTNUM не сбрасываем при смене вкладки (в т.ч. «Сравнение методов»).
    st.session_state["_active_page"] = page
    if page == "Лаборатория":
        laboratory_tab()
    elif page == "Подбор J функции Леверетта":
        leverett_tab()
    elif page == "Брукса-Кори":
        brooks_corey_tab()
    else:
        compare_methods_tab()

    _scroll_to_top_if_pending(finish=True)


if __name__ == "__main__":
    main()
