from __future__ import annotations

import io
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
from optimize import run_pipeline

st.set_page_config(page_title="J-функция Леверетта", layout="wide")
st.markdown(
    """
    <style>
    /* Стабилизирует прокрутку при первом rerun после изменения виджетов */
    html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
        overflow-anchor: none;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

DATA_DIR = Path(__file__).resolve().parent / "data"
UPLOAD_DIR = DATA_DIR / "uploads"


@st.cache_data(show_spinner=False)
def _read_table_from_bytes(file_bytes: bytes, filename: str) -> pd.DataFrame:
    name = filename.lower()
    if name.endswith(".csv"):
        return pd.read_csv(io.BytesIO(file_bytes), low_memory=False)
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
        return pd.read_csv(p, low_memory=False)
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


def _clean_wells_df(df: pd.DataFrame) -> pd.DataFrame:
    df = _normalize_columns(df.copy())
    if df.empty:
        return df
    numeric_candidates = [col for col in df.columns if col != df.columns[0]]
    for col in numeric_candidates:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if numeric_candidates:
        mask_bad = (df[numeric_candidates] <= -1).any(axis=1)
        df = df.loc[~mask_bad].dropna()
    if "ACTNUM_GDM" in df.columns:
        df = df.loc[df["ACTNUM_GDM"] != 0]
    if "FWL_GDM" in df.columns:
        df = df.loc[df["FWL_GDM"] != 0]
        df = df.loc[df["FWL_GDM"] >= 3]
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
def _clean_wells_cached(df: pd.DataFrame) -> pd.DataFrame:
    return _clean_wells_df(df)


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


def _well_convergence_percent_weighted(df: pd.DataFrame) -> float:
    """Средневзвешенный процент сходимости: веса из колонки weight (если есть)."""
    eps = 1e-6
    true_vals = pd.to_numeric(df["Кнг_W"], errors="coerce")
    pred_vals = pd.to_numeric(df["Kng_model"], errors="coerce")
    w = pd.to_numeric(df.get("weight", 1.0), errors="coerce").fillna(1.0).to_numpy(dtype=float)
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


def _well_weighted_crossplot_df(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for well, g in df.groupby("WELL_NAME"):
        y_true = pd.to_numeric(g["Кнг_W"], errors="coerce")
        y_pred = pd.to_numeric(g["Kng_model"], errors="coerce")
        w = pd.to_numeric(g.get("weight", 1.0), errors="coerce").fillna(1.0)
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


def _compute_qa_metrics(df: pd.DataFrame, true_col: str, pred_col: str) -> pd.DataFrame:
    rows = []
    for pvt_raw, g in df.groupby("PVTNUM_GDM"):
        y_true = pd.to_numeric(g[true_col], errors="coerce").to_numpy()
        y_pred = pd.to_numeric(g[pred_col], errors="coerce").to_numpy()
        w = pd.to_numeric(g.get("weight", 1.0), errors="coerce").fillna(1.0).to_numpy()
        m = np.isfinite(y_true) & np.isfinite(y_pred) & np.isfinite(w)
        if m.sum() == 0:
            continue
        yt = y_true[m]
        yp = y_pred[m]
        ww = w[m]
        err = yp - yt
        mae = float(np.average(np.abs(err), weights=ww))
        rmse = float(np.sqrt(np.average(err**2, weights=ww)))
        bias = float(np.average(err, weights=ww))
        score = float(1 - (np.sum(ww * np.abs(err)) / np.sum(ww)))
        # локально, чтобы не тащить импорт сверху второй раз
        from sklearn.metrics import r2_score

        r2 = float(r2_score(yt, yp)) if len(yt) > 1 else np.nan
        rows.append(
            {
                "PVTNUM_GDM": int(float(pvt_raw)),
                "MAE": mae,
                "RMSE": rmse,
                "BIAS": bias,
                "R2": r2,
                "SCORE": score,
            }
        )
    qa = pd.DataFrame(rows)
    if not qa.empty:
        qa = qa.sort_values("PVTNUM_GDM").reset_index(drop=True)
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


def _render_methods_comparison_block(block_key: str = "compare_methods") -> None:
    cat_j = _snapshot_catalog("J")
    cat_bc = _snapshot_catalog("BC")
    if cat_j.empty or cat_bc.empty:
        st.info("Сохраните результаты обоих методов кнопкой «Запомнить результаты по скважинам».")
        return

    def _calc_metrics(df_snap: pd.DataFrame) -> dict[str, float]:
        y = pd.to_numeric(df_snap["Кнг_hist"], errors="coerce").to_numpy(dtype=float)
        p = pd.to_numeric(df_snap["Кнг_model"], errors="coerce").to_numpy(dtype=float)
        m = np.isfinite(y) & np.isfinite(p) & (np.abs(y) > 1e-15)
        if m.sum() == 0:
            return {"MAE": np.nan, "RMSE": np.nan, "BIAS": np.nan, "CORR": np.nan, "SCORE": np.nan}
        e = p[m] - y[m]
        mae = float(np.mean(np.abs(e)))
        rmse = float(np.sqrt(np.mean(e**2)))
        bias = float(np.mean(e))
        corr = float(np.corrcoef(y[m], p[m])[0, 1]) if m.sum() > 2 else np.nan
        score = float(1 - np.mean(np.abs(e)))
        return {"MAE": mae, "RMSE": rmse, "BIAS": bias, "CORR": corr, "SCORE": score}

    def _calc_metrics_by_horizon(df_snap: pd.DataFrame) -> pd.DataFrame:
        if "PVTNUM_GDM" not in df_snap.columns or df_snap["PVTNUM_GDM"].isna().all():
            row = _calc_metrics(df_snap)
            row["PVTNUM_GDM"] = "—"
            row["N_points"] = int(
                (
                    np.isfinite(pd.to_numeric(df_snap["Кнг_hist"], errors="coerce"))
                    & np.isfinite(pd.to_numeric(df_snap["Кнг_model"], errors="coerce"))
                    & (np.abs(pd.to_numeric(df_snap["Кнг_hist"], errors="coerce")) > 1e-15)
                ).sum()
            )
            return pd.DataFrame([row])
        out_rows = []
        for pvt, g in df_snap.groupby("PVTNUM_GDM", dropna=False):
            m = _calc_metrics(g)
            m["PVTNUM_GDM"] = pvt
            y = pd.to_numeric(g["Кнг_hist"], errors="coerce")
            p = pd.to_numeric(g["Кнг_model"], errors="coerce")
            m["N_points"] = int(
                (y.notna() & p.notna() & (y.abs() > 1e-15)).sum()
            )
            out_rows.append(m)
        tab = pd.DataFrame(out_rows)
        tab["_ord"] = pd.to_numeric(tab["PVTNUM_GDM"], errors="coerce")
        tab = tab.sort_values("_ord", na_position="last").drop(columns="_ord").reset_index(drop=True)
        return tab

    def _crossplot_df(df_snap: pd.DataFrame) -> pd.DataFrame:
        if df_snap.empty:
            return df_snap
        y = pd.to_numeric(df_snap["Кнг_hist"], errors="coerce")
        p = pd.to_numeric(df_snap["Кнг_model"], errors="coerce")
        mask = y.notna() & p.notna() & (y.abs() > 1e-15)
        return df_snap.loc[mask].copy()

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
        region_col = "PVTNUM_GDM" if "PVTNUM_GDM" in work.columns else None
        if region_col is None:
            work["_REGION"] = "Все"
            region_col = "_REGION"
        rows = []
        for region, g in work.groupby(region_col, dropna=False):
            w = g["weight"].to_numpy(dtype=float)
            sw = float(np.sum(w))
            if sw <= 0:
                continue
            hist_w = float(np.sum(w * g["Кнг_hist"].to_numpy(dtype=float)) / sw)
            model_w = float(np.sum(w * g["Кнг_model"].to_numpy(dtype=float)) / sw)
            rows.append(
                {
                    "Регион": region,
                    "Средневзвешенное Кнг (история)": hist_w,
                    "Средневзвешенное Кнг (модель)": model_w,
                    "Дельта": model_w - hist_w,
                    "Точек": int(len(g)),
                }
            )
        if not rows:
            return pd.DataFrame()
        tab = pd.DataFrame(rows)
        tab["_ord"] = pd.to_numeric(tab["Регион"], errors="coerce")
        tab = tab.sort_values("_ord", na_position="last").drop(columns="_ord").reset_index(drop=True)
        return tab

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
                "История": [
                    float(np.nanmin(hist)),
                    float(np.nanmax(hist)),
                    float(np.nanmean(hist)),
                    float(np.nanmedian(hist)),
                    float(np.nanstd(hist)),
                ],
                "Модель": [
                    float(np.nanmin(model)),
                    float(np.nanmax(model)),
                    float(np.nanmean(model)),
                    float(np.nanmedian(model)),
                    float(np.nanstd(model)),
                ],
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
    st.caption("Точки с Кнг_hist = 0 не участвуют в метриках и на кроссплотах.")
    tab_j = _calc_metrics_by_horizon(sj)
    tab_b = _calc_metrics_by_horizon(sb)
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
            hover_data=[c for c in ["WELL_NAME", "_AXIS", "Регион"] if c in sj_plot.columns],
            title="J: расчетное(историческое)",
            opacity=0.65,
        )
        fig_j.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(color="red", dash="dash"))
        fig_j.update_traces(hovertemplate="Кнг_hist=%{x:.3f}<br>Кнг_model=%{y:.3f}<extra></extra>")
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
            hover_data=[c for c in ["WELL_NAME", "_AXIS", "Регион"] if c in sb_plot.columns],
            title="БК: расчетное(историческое)",
            opacity=0.65,
        )
        fig_bc.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(color="red", dash="dash"))
        fig_bc.update_traces(hovertemplate="Кнг_hist=%{x:.3f}<br>Кнг_model=%{y:.3f}<extra></extra>")
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
            reg_opts = ["Все регионы"] + [str(x) for x in tab_reg_j["Регион"].tolist()]
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
            reg_opts = ["Все регионы"] + [str(x) for x in tab_reg_b["Регион"].tolist()]
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
                    "WELL_NAME": True,
                    "Регион": True,
                    "points": ":.3f",
                    "avg_weight": ":.3f",
                    "convergence_percent": ":.3f",
                },
                title="J: кроссплот по скважинам",
                opacity=0.85,
            )
            fig_jw.update_traces(hovertemplate="Кнг_hist_wmean=%{x:.3f}<br>Кнг_J_wmean=%{y:.3f}<extra></extra>")
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
                    "WELL_NAME": True,
                    "Регион": True,
                    "points": ":.3f",
                    "avg_weight": ":.3f",
                    "convergence_percent": ":.3f",
                },
                title="БК: кроссплот по скважинам",
                opacity=0.85,
            )
            fig_bw.update_traces(hovertemplate="Кнг_hist_wmean=%{x:.3f}<br>Кнг_БК_wmean=%{y:.3f}<extra></extra>")
            fig_bw.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(color="red", dash="dash"))
            fig_bw.update_layout(
                xaxis_title="Кнг_hist (средневзвеш.)",
                yaxis_title="Кнг_БК (средневзвеш.)",
                legend_title_text="Регион",
            )
            cw2.plotly_chart(fig_bw, use_container_width=True)
            cw2.caption(f"Скважин (точек кроссплота): {len(sb_cross)}")

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
        hover_data=["WELL_NAME", "corr_j_hist", "corr_bc_hist", "points_interp"],
        title="Кластеры: согласованность J-функции и Брукса–Кори (корреляция и тренд)",
    )
    fig_cluster.update_traces(hovertemplate="corr_j_bc=%{x:.3f}<br>trend_j_bc=%{y:.3f}<extra></extra>")
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


def _default_bounds_for_pvts(pvts: list[int]) -> dict[int, dict[str, tuple[float, float]]]:
    return {pvt: {"a": (0.05, 0.30), "b": (-3.0, -0.5), "sigma": (25.0, 35.0)} for pvt in pvts}


def _bounds_ui_manual(pvts: list[int]) -> dict[int, dict[str, tuple[float, float]]]:
    st.subheader("Ограничения коэффициентов по регионам (PVTNUM)")
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
    components.html(
        """
        <script>
        (function() {
          const doc = window.parent.document;
          if (!doc) return;

          // Снимаем фокус с активного элемента, чтобы не "держался" низ страницы
          if (doc.activeElement && typeof doc.activeElement.blur === 'function') {
            doc.activeElement.blur();
          }

          // Ищем реальные scroll-контейнеры Streamlit и принудительно ставим в начало
          const candidates = [
            doc.scrollingElement,
            doc.documentElement,
            doc.body,
            doc.querySelector('section.main'),
            doc.querySelector('[data-testid="stAppViewContainer"]'),
            doc.querySelector('[data-testid="stMain"]')
          ].filter(Boolean);

          candidates.forEach((el) => {
            try {
              el.scrollTop = 0;
              if (typeof el.scrollTo === 'function') {
                el.scrollTo({ top: 0, left: 0, behavior: 'auto' });
              }
            } catch (e) {}
          });

          // Якорь-фокус вверху для стабильного старта "активной" области
          let topAnchor = doc.getElementById('__cursor_top_anchor__');
          if (!topAnchor) {
            topAnchor = doc.createElement('div');
            topAnchor.id = '__cursor_top_anchor__';
            topAnchor.setAttribute('tabindex', '-1');
            topAnchor.style.position = 'absolute';
            topAnchor.style.top = '0';
            topAnchor.style.left = '0';
            topAnchor.style.width = '1px';
            topAnchor.style.height = '1px';
            topAnchor.style.opacity = '0';
            doc.body.prepend(topAnchor);
          }
          try {
            topAnchor.focus({ preventScroll: true });
            topAnchor.scrollIntoView({ block: 'start', inline: 'nearest', behavior: 'auto' });
          } catch (e) {}
        })();
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


def _map_uploaded_wells_df(raw_wells: pd.DataFrame, *, key_prefix: str, title: str) -> pd.DataFrame:
    df = _normalize_columns(raw_wells.copy())
    cols = list(df.columns)
    st.subheader(title)
    st.caption("Выберите соответствия колонок сверху, затем проверьте предпросмотр.")
    if not cols:
        return df
    mapping_rules: list[tuple[str, str, list[str]]] = [
        ("WELL_NAME", "Скважина", ["WELL_NAME", "СКВАЖ", "WELL", "STVOL"]),
        ("PVTNUM_GDM", "Регион", ["PVTNUM", "PVT", "ЭКСПЛ", "OBJECT", "OBJ"]),
        ("PORO_GDM", "Пористость", ["PORO", "ПОР", "PHI"]),
        ("PERM_GDM", "Проницаемость", ["PERM", "ПРОНИ", "KPR", "K_PR"]),
        ("PC", "Капиллярное давление", ["PC", "КАПИЛ", "P_CAP"]),
        ("SWL_GDM", "Кво", ["SWL", "SWI", "SW_MIN", "КВО"]),
        ("Кнг_W", "Нефтенасыщенность", ["КНГ", "KNG", "НЕФТЕНАС", "RIGIS"]),
    ]
    saved_map = st.session_state.get(f"{key_prefix}_wells_map_saved", {})
    defaults = {}
    for target, _, hints in mapping_rules:
        cand = saved_map.get(target)
        defaults[target] = cand if isinstance(cand, str) and cand in cols else _safe_guess_col(cols, hints)
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
            key=f"{key_prefix}_map_wells_{target}",
        )

    row2 = st.columns(4)
    for i, (target, ui_label, _) in enumerate(row2_rules):
        default = defaults[target]
        idx = cols.index(default) if default in cols else 0
        pick[target] = row2[i].selectbox(
            ui_label,
            options=cols,
            index=idx,
            key=f"{key_prefix}_map_wells_{target}",
        )

    perf_saved = saved_map.get("Perf_GDM")
    perf_default = perf_saved if isinstance(perf_saved, str) and perf_saved in cols else _safe_guess_col(cols, ["PERF_GDM", "PERF", "ПЕРФ", "ПЕРФОРА"])
    perf_options = ["<не использовать>"] + cols
    perf_idx = perf_options.index(perf_default) if perf_default in perf_options else 0
    perf_pick = row2[3].selectbox("Перфорация", options=perf_options, index=perf_idx, key=f"{key_prefix}_map_wells_Perf_GDM")
    if perf_pick != "<не использовать>":
        pick["Perf_GDM"] = perf_pick
    if len(set(pick.values())) != len(pick):
        st.error("Для обязательных полей скважин выбраны повторяющиеся колонки. Выберите уникальные соответствия.")
        return pd.DataFrame()
    st.session_state[f"{key_prefix}_wells_map_saved"] = pick.copy()
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
    rules: list[tuple[str, str, list[str]]] = [
        ("WELL_NAME", "Скважина", ["WELL_NAME", "СТВОЛ", "СКВАЖ", "WELL"]),
        ("PVTNUM_GDM", "Регион", ["PVTNUM", "ЭКСПЛ", "ОБЪЕКТ", "OBJECT"]),
    ]
    saved_map = st.session_state.get(f"{key_prefix}_prod_map_saved", {})
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
        picks[target] = host.selectbox(ui_label, options=cols, index=idx, key=f"{key_prefix}_map_prod_{target}")
    st.caption("Предпросмотр загруженного файла добычи:")
    st.dataframe(_round_df(df.head(12)), use_container_width=True)
    if len(set(picks.values())) != len(picks):
        st.error("Для файла добычи выбраны повторяющиеся колонки сопоставления.")
        return pd.DataFrame()
    st.session_state[f"{key_prefix}_prod_map_saved"] = picks.copy()
    rename_map = {src: dst for dst, src in picks.items()}
    return df.rename(columns=rename_map)


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

    ann_text = []
    if trend_fit and np.isfinite(trend_fit.get("a", np.nan)) and np.isfinite(trend_fit.get("b", np.nan)):
        a, b = trend_fit["a"], trend_fit["b"]
        fig.add_trace(
            go.Scatter(
                x=grid,
                y=a * (grid**b),
                mode="lines",
                name=f"Тренд лаб.: J = {a:.4g}·Swn^{b:.4g}",
                line=dict(width=2, color="black"),
            )
        )
        ann_text.append(f"Лаб. тренд: J = {a:.4g}·Swn<sup>{b:.4g}</sup>")

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
                        y=aa * (grid**bb),
                        mode="lines",
                        name=name,
                        line=dict(width=2, dash=dash, color=col),
                    )
                )

    if optimal and np.isfinite(optimal.get("a", np.nan)) and np.isfinite(optimal.get("b", np.nan)):
        a, b = optimal["a"], optimal["b"]
        fig.add_trace(
            go.Scatter(
                x=grid,
                y=a * (grid**b),
                mode="lines",
                name=f"Модель (опт.): J = {a:.4g}·Swn^{b:.4g}",
                line=dict(width=3, color="crimson"),
            )
        )
        ann_text.append(f"Опт. модель: J = {a:.4g}·Swn<sup>{b:.4g}</sup>")

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

    lab_src = st.radio(
        "Источник лабораторных точек для J(Swn)",
        options=("kkd", "matrix"),
        format_func=lambda x: "База ККД (Excel → SQLite)" if x == "kkd" else "Excel: матрица ступеней (Sw + J)",
        horizontal=True,
        key="lab_cloud_source_mode",
    )

    df: pd.DataFrame | None = None

    if lab_src == "kkd":
        if excel_path().is_file() and not sqlite_path().is_file():
            try:
                build_kkd_sqlite()
            except Exception:
                pass
        up = st.file_uploader("Загрузить Excel ККД (сохранится как data/ККД_БД.xlsx)", type=["xlsx", "xls"], key="kkd_upload")
        if up is not None:
            target = DATA_DIR / "ККД_БД.xlsx"
            target.write_bytes(up.getbuffer())
            st.success(f"Файл сохранён: {target}")

        c1, c2 = st.columns(2)
        with c1:
            if st.button("Пересобрать SQLite из Excel", type="primary"):
                try:
                    dbp = build_kkd_sqlite()
                    st.success(f"База обновлена: {dbp}")
                except FileNotFoundError as e:
                    st.error(str(e))
        with c2:
            st.caption(f"Excel: `{excel_path()}`  |  SQLite: `{sqlite_path()}`")

        try:
            db_file = sqlite_path()
            df = _load_kkd_cached(str(db_file), db_file.stat().st_mtime)
        except FileNotFoundError:
            st.info(
                "База ещё не создана. Положите `ККД_БД.xlsx` в папку `data` или загрузите файл выше, "
                "затем нажмите «Пересобрать SQLite из Excel». Либо выберите источник «матрица ступеней»."
            )
            return
    else:
        st.markdown(
            "Загрузите `.xlsx` / `.xls`, где **первая строка** задаёт объединённые заголовки "
            "(блок водонасыщенности и блок J), **вторая** — подписи ступеней («1 ст.», …). "
            "Таблица разворачивается: для каждой строки считаются `Sw_min`/`Sw_max` по всем ступеням, "
            "`Swn = (Sw − Sw_min)/(Sw_max − Sw_min)`; если Swn получается 0 или 1 — в ячейку записывается пропуск."
        )
        mat_up = st.file_uploader("Файл матрицы ступеней", type=["xlsx", "xls"], key="lab_matrix_stairs_upload")
        if mat_up is None:
            st.info("Выберите файл Excel с матрицей ступеней.")
            return
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
        with st.expander("Предпросмотр (первые 20 строк)", expanded=False):
            st.dataframe(df.head(20), use_container_width=True)

    cols = list(df.columns)

    def _idx_or_zero(name: str | None) -> int:
        return cols.index(name) if name and name in cols else 0

    area_guess = guess_area_column(cols)
    hor_guess = guess_horizon_column(cols)
    j_guess = guess_j_column(cols)
    _, swn_list = pick_second_swn_column(cols)

    st.subheader("Сопоставление колонок")
    c1, c2, c3, c4 = st.columns(4)
    area_col = c1.selectbox("Колонка площади", options=cols, index=_idx_or_zero(area_guess))
    hor_col = c2.selectbox("Колонка кода горизонта", options=cols, index=_idx_or_zero(hor_guess))
    j_col = c3.selectbox("J (функция Леверетта)", options=cols, index=_idx_or_zero(j_guess))
    default_swn = "Swn" if "Swn" in cols else cols[0]
    swn_idx = cols.index(default_swn) if default_swn in cols else 0
    swn_col = c4.selectbox(
        "Swn",
        options=cols,
        index=swn_idx,
        help=(f"Найденные кандидаты SWn: {swn_list}" if swn_list else None),
    )

    areas = sorted(pd.Series(df[area_col]).dropna().astype(str).str.strip().unique().tolist())
    horizons = sorted(pd.Series(df[hor_col]).dropna().astype(str).str.strip().unique().tolist())

    st.subheader("Фильтр данных")
    st.caption("Площади и горизонты сохраняются в рамках сессии.")
    if "lab_sel_areas" not in st.session_state:
        st.session_state["lab_sel_areas"] = areas.copy()
    else:
        st.session_state["lab_sel_areas"] = [a for a in st.session_state["lab_sel_areas"] if a in areas]
    if "lab_sel_hors" not in st.session_state:
        st.session_state["lab_sel_hors"] = []
    else:
        st.session_state["lab_sel_hors"] = [h for h in st.session_state["lab_sel_hors"] if h in horizons]
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
        }
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

    st.divider()
    st.subheader("Лаборатория: загрузка данных для метода Брукса-Кори")
    bc_file = st.file_uploader(
        "Загрузить таблицу лабораторных точек (файл зависимости_облако_точек)",
        type=["csv", "xlsx", "xls"],
        key="bc_lab_upload",
    )
    if bc_file is not None:
        try:
            bc_df = _read_table(bc_file)
            bc_df = _add_pvit_n_if_missing(_normalize_columns(bc_df))
            st.session_state["bc_user_upload_df"] = bc_df
            # Сброс виджетов фильтра БК (площади/горизонты), чтобы списки брались из нового файла
            st.session_state["bc_user_upload_rev"] = int(time.time() * 1000) % 1_000_000_000
            st.success(f"Данные Брукса-Кори загружены: {len(bc_df)} строк (используются только они, без ККД из БД).")
        except Exception as e:
            st.error(f"Ошибка загрузки данных Брукса-Кори: {e}")


def leverett_tab() -> None:
    st.title("Подбор J функции Леверетта")
    st.write("Загрузите данные скважин, задайте ограничения или автограницы, выполните подбор.")

    lab_ready = bool(st.session_state.get("lab_cloud_ready"))
    cloud: pd.DataFrame | None = st.session_state.get("lab_cloud")

    with st.sidebar:
        st.header("Загрузка данных")
        wells_file = st.file_uploader("Файл скважин", type=["csv", "xlsx", "xls", "txt"], key="wells_file")
        prod_file = st.file_uploader("Файл добычи (опционально для весов)", type=["csv", "xlsx", "xls"], key="prod_file")
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

    if wells_file is not None:
        p = _persist_uploaded_file(wells_file, "wells")
        st.session_state["wells_file_path"] = p
        st.session_state["shared_wells_file_path"] = p
    if prod_file is not None:
        p = _persist_uploaded_file(prod_file, "prod")
        st.session_state["prod_file_path"] = p
        st.session_state["shared_prod_file_path"] = p

    wells_path = _coalesce_path("shared_wells_file_path", "wells_file_path", "bc_wells_path")
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
        df_wells = _clean_wells_cached(mapped_wells)
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

    df_wells["PVTNUM_GDM"] = pd.to_numeric(df_wells["PVTNUM_GDM"], errors="coerce")
    pvts = sorted(df_wells["PVTNUM_GDM"].dropna().astype(int).unique().tolist())
    if not pvts:
        st.error("Не удалось определить регионы PVTNUM_GDM.")
        return

    st.subheader("Предпросмотр данных")
    st.dataframe(_round_df(df_wells.head(30)), use_container_width=True)
    st.caption(f"Строк после очистки: {len(df_wells)}")

    pvt_horizon_map: dict[int, list[str]] = st.session_state.get("pvt_horizon_map", {})
    if lab_ready and cloud is not None and not cloud.empty:
        st.subheader("Связь код горизонта -> PVTNUM")
        hors_universe = sorted(cloud["lab_horizon"].astype(str).unique().tolist())
        for pvt in pvts:
            existing = pvt_horizon_map.get(pvt, [])
            pvt_horizon_map[pvt] = st.multiselect(
                f"Горизонты для PVTNUM {pvt}",
                options=hors_universe,
                default=existing,
                key=f"pvt_hor_map_{pvt}",
            )
        st.session_state["pvt_horizon_map"] = pvt_horizon_map

    st.subheader("Режим границ a, b")
    if lab_ready and cloud is not None and not cloud.empty:
        bounds_mode = st.radio(
            "Как задавать amin/amax/bmin/bmax",
            options=("Ручной ввод", "Автоподбор по лабораторному облаку"),
            horizontal=True,
        )
    else:
        bounds_mode = "Ручной ввод"
        st.caption("Автоподбор границ доступен после выбора данных во вкладке «Лаборатория».")

    bounds_by_pvt: dict[int, dict[str, tuple[float, float]]] = {}
    auto_preview: dict[int, dict] = {}

    if bounds_mode == "Ручной ввод":
        bounds_by_pvt = _bounds_ui_manual(pvts)
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

        stored_bounds = st.session_state.get("auto_bounds_by_pvt") or {}
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
            bounds_by_pvt = stored_bounds

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
        merged.update(partial)
        return merged

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
                    result_df, params_df, qa_df = run_pipeline(
                        df_wells=df_wells,
                        df_prod=df_prod,
                        bounds_by_pvt=_merge_bounds(bounds_by_pvt),
                        maxiter=maxiter,
                        popsize=popsize,
                        fixed_params=fixed_params,
                        optimizer_method=optimizer_method,
                    )
                except Exception as e:
                    st.error(f"Ошибка расчета: {e}")
                else:
                    st.session_state["leverett_result_df"] = result_df
                    st.session_state["leverett_params_df"] = params_df
                    st.session_state["leverett_qa_df"] = qa_df
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
                    result_df, params_df, qa_df = run_pipeline(
                        df_wells=df_wells,
                        df_prod=df_prod,
                        bounds_by_pvt=_merge_bounds(bounds_by_pvt),
                        maxiter=maxiter,
                        popsize=popsize,
                        fixed_params=None,
                        optimizer_method=optimizer_method,
                    )
                except Exception as e:
                    st.error(f"Ошибка расчета: {e}")
                else:
                    st.session_state["leverett_result_df"] = result_df
                    st.session_state["leverett_params_df"] = params_df
                    st.session_state["leverett_qa_df"] = qa_df
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

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Параметры")
        st.dataframe(_round_df(params_df), use_container_width=True)
    with c2:
        st.subheader("Метрики")
        st.dataframe(_round_df(qa_df), use_container_width=True)
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

    color_mode = st.radio(
        "Палитра точек на графике",
        options=("По весу", "По толщине"),
        horizontal=True,
        key="scatter_color_mode",
    )

    depth_for_thickness = _pick_depth_column(region_df)
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

    hover_cols = [
        c
        for c in ["WELL_NAME", "PC", "PORO_GDM", "PERM_GDM", "SWL_GDM", "Кнг_W", "Kng_model", "weight", "thickness"]
        if c in region_df.columns
    ]
    hover_fmt = {c: (True if c == "WELL_NAME" else ":.3f") for c in hover_cols}
    color_col = "weight"
    if color_mode == "По толщине" and "thickness" in region_df.columns:
        color_col = "thickness"

    fig_scatter = px.scatter(
        region_df,
        x="Кнг_W",
        y="Kng_model",
        color=color_col if color_col in region_df.columns else None,
        color_continuous_scale="Viridis",
        hover_data=hover_fmt,
        title=f"PVT {region}: предсказанное Кнг(историческое) ({'вес' if color_col == 'weight' else 'толщина'})",
        opacity=0.7,
    )
    fig_scatter.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(color="red", dash="dash"))
    fig_scatter.update_layout(xaxis_title="Кнг историческое (ГИС)", yaxis_title="Кнг предсказанное")
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.subheader("Кроссплот по скважинам (средневзвешенные значения)")
    if "WELL_NAME" in region_df.columns:
        cross_df = _well_weighted_crossplot_df(region_df)
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
                    "WELL_NAME": True,
                    "points": ":.3f",
                    "avg_weight": ":.3f",
                    "convergence_percent": ":.3f",
                },
                title=f"PVT {region}: кроссплот по скважинам (средневзвешенно)",
                opacity=0.85,
            )
            fig_well_cross.update_traces(hovertemplate="Кнг_W_wmean=%{x:.3f}<br>Kng_model_wmean=%{y:.3f}<extra></extra>")
            fig_well_cross.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(color="red", dash="dash"))
            fig_well_cross.update_layout(
                xaxis_title="Кнг_W (средневзвеш.)",
                yaxis_title="Kng_model (средневзвеш.)",
            )
            st.plotly_chart(fig_well_cross, use_container_width=True)

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
        well = st.selectbox("Скважина для детального графика", options=wells)
        well_df = region_df[region_df["WELL_NAME"].astype(str) == well].copy()
        depth_col = _pick_depth_column(well_df)
        if depth_col is None:
            st.warning("Не найдена колонка глубины (например DEPTH/DEPT).")
        elif "ACTNUM_GDM" not in well_df.columns:
            st.warning("В данных отсутствует ACTNUM_GDM для детального графика.")
        else:
            well_df[depth_col] = pd.to_numeric(well_df[depth_col], errors="coerce")
            well_df = well_df.dropna(subset=[depth_col]).sort_values(depth_col).reset_index(drop=True)
            curve_df = well_df[[depth_col, "ACTNUM_GDM", "Кнг_W", "Kng_model"]].copy()
            curve_df = curve_df.rename(columns={"Кнг_W": "Кн РИГИС", "Kng_model": "Кн J-функция"})
            chart_df = curve_df.melt(
                id_vars=[depth_col],
                value_vars=["ACTNUM_GDM", "Кн РИГИС", "Кн J-функция"],
                var_name="Кривая",
                value_name="Значение",
            )
            fig_well = px.line(
                chart_df,
                x="Значение",
                y=depth_col,
                color="Кривая",
                title=f"Скважина {well}: вертикальный профиль",
            )
            fig_well.update_traces(mode="lines")
            fig_well.update_traces(hovertemplate="Значение=%{x:.3f}<br>Глубина=%{y:.3f}<br>Кривая=%{fullData.name}<extra></extra>")
            fig_well.update_yaxes(autorange="reversed")
            fig_well.update_layout(xaxis_title="Значение", yaxis_title="Глубина")
            st.plotly_chart(fig_well, use_container_width=True)

            conv_percent = _well_convergence_percent_weighted(well_df)
            if np.isfinite(conv_percent):
                st.metric("Сходимость для скважины (средневзвеш.), %", f"{conv_percent:.2f}")

def brooks_corey_tab() -> None:
    st.title("Метод Брукса-Кори")

    bc_lab_df, bc_source = _get_bc_source_df()
    if bc_lab_df is None or bc_lab_df.empty:
        st.info(
            "Нет данных для Брукса-Кори: загрузите свой файл во вкладке «Лаборатория» "
            "(блок Брукса-Кори) или убедитесь, что доступна БД ККД."
        )
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
        wells_file = st.file_uploader("Файл скважин (для БК)", type=["csv", "xlsx", "xls", "txt"], key="bc_wells_file")
        prod_file = st.file_uploader("Файл добычи (для весов БК, опционально)", type=["csv", "xlsx", "xls"], key="bc_prod_file")
        maxiter = st.slider("Итерации БК", min_value=50, max_value=350, value=180, step=10, key="bc_maxiter")
        popsize = st.slider("Популяция БК", min_value=10, max_value=40, value=18, step=1, key="bc_popsize")
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
        df_geo_raw = _clean_wells_cached(mapped_geo)
        if (not use_perf_weights) and ("Perf_GDM" in df_geo_raw.columns):
            df_geo_raw = df_geo_raw.drop(columns=["Perf_GDM"])
        df_prod = _clean_prod_cached(mapped_prod)
        df_geo = prepare_brooks_training_data(df_geo_raw, df_prod)
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
        "n (лаборатория)": saved_lab_map.get("n (лаборатория)") if saved_lab_map.get("n (лаборатория)") in cols else _safe_guess_col(cols, [" N", "N_"]),
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
    st.caption("Подготовленная таблица для БК (с новыми столбцами и фильтром 1 строка на образец/скважину):")
    preview_cols = [c for c in ["Скважина", "Номер образца", "Порядковый номер образца", "HORIZON", "PORO_LAB_FRAC", "SWL_LAB", "PERM_LAB", "perm_poro", "PVIT_LAB", "N_LAB"] if c in lab.columns]
    st.dataframe(_round_df(lab[preview_cols].head(20)), use_container_width=True)

    pvts = sorted(pd.to_numeric(df_geo["PVTNUM_GDM"], errors="coerce").dropna().astype(int).unique().tolist())
    horizons = sorted(lab["HORIZON"].unique().tolist())
    st.subheader("Связь горизонт -> PVTNUM (Брукса-Кори)")
    pvt_h_map = st.session_state.get("bc_pvt_h_map", {})
    # Синхронизация: если горизонт убран из фильтра "Горизонты для БК",
    # удаляем его из ранее сохраненных связей горизонт -> PVTNUM.
    allowed_h = set(horizons)
    for p in list(pvt_h_map.keys()):
        prev = pvt_h_map.get(p, []) or []
        pvt_h_map[p] = [h for h in prev if str(h) in allowed_h]
    for p in pvts:
        prev_sel = [h for h in (pvt_h_map.get(p, []) or []) if str(h) in allowed_h]
        pvt_h_map[p] = st.multiselect(
            f"Горизонты для PVTNUM {p}",
            options=horizons,
            default=prev_sel,
            key=f"bc_map_{p}",
        )
    st.session_state["bc_pvt_h_map"] = pvt_h_map

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
        p_preview = st.selectbox("Регион для предпросмотра ручных коэффициентов", options=pvts, key="bc_manual_preview_pvt")
        hs = pvt_h_map.get(p_preview, [])
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
    _ui_lock(bc_busy, "ui_lock_bc")
    if bc_busy:
        st.warning("Идет расчет Брукса-Кори... Пожалуйста, дождитесь завершения.")
    if st.button("Рассчитать Брукса-Кори", type="primary", disabled=bc_busy):
        if any(not pvt_h_map.get(p) for p in pvts):
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
            h = pvt_h_map[p]
            lsub = lab[lab["HORIZON"].isin(h)].copy()
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
        bc_timing = pd.DataFrame(timing_rows).sort_values("PVTNUM_GDM").reset_index(drop=True) if timing_rows else pd.DataFrame()
        total_elapsed = float(time.perf_counter() - t0_total)
        st.session_state["bc_result_df"] = bc_res
        st.session_state["bc_params_df"] = bc_params
        st.session_state["bc_qa_df"] = bc_qa
        st.session_state["bc_timing_df"] = bc_timing
        st.session_state["bc_total_elapsed_sec"] = total_elapsed
        st.session_state["bc_meta"] = bc_meta
        st.session_state["bc_busy"] = False
        _ui_lock(False, "ui_lock_bc")
        st.success("Расчет Брукса-Кори завершен.")

    bc_res: pd.DataFrame | None = st.session_state.get("bc_result_df")
    bc_params: pd.DataFrame | None = st.session_state.get("bc_params_df")
    bc_qa: pd.DataFrame | None = st.session_state.get("bc_qa_df")
    bc_timing: pd.DataFrame | None = st.session_state.get("bc_timing_df")
    bc_total_elapsed = st.session_state.get("bc_total_elapsed_sec")
    if bc_res is None or bc_params is None or bc_res.empty:
        return

    cpar, cqa = st.columns(2)
    with cpar:
        st.subheader("Параметры Брукса-Кори по регионам")
        st.dataframe(_round_df(bc_params), use_container_width=True)
        if bc_timing is not None and not bc_timing.empty:
            st.subheader("Статистика времени расчета БК")
            bc_timing_ru = bc_timing.rename(
                columns={
                    "PVTNUM_GDM": "Регион (PVTNUM)",
                    "rows_geo": "Строк геологии",
                    "rows_lab": "Лабораторных точек",
                    "elapsed_sec": "Время расчета, сек",
                }
            )
            st.dataframe(
                _round_df(bc_timing_ru),
                use_container_width=True,
                hide_index=True,
                height=min(600, 35 * (len(bc_timing_ru) + 1)),
            )
            if bc_total_elapsed is not None:
                st.metric("Общее время расчета БК, сек", f"{float(bc_total_elapsed):.2f}")
    with cqa:
        st.subheader("Метрики качества Брукса-Кори")
        if bc_qa is not None and not bc_qa.empty:
            st.dataframe(_round_df(bc_qa), use_container_width=True)
            st.metric("GLOBAL SCORE (БК)", f"{bc_qa['SCORE'].mean():.3f}")
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
    g_conv = _filter_convergence_points(g.rename(columns={"Kng_BC_model": "Kng_model"})).rename(columns={"Kng_model": "Kng_BC_model"})
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
            c: (True if c == "WELL_NAME" else ":.3f")
            for c in ["WELL_NAME", "PC", "PORO_GDM", "Kng_BC_model", "Кнг_W", "thickness", "weight"]
            if c in g_conv.columns
        },
        title=f"PVT {psel}: предсказанное(историческое) (Брукса-Кори)",
        opacity=0.75,
    )
    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(color="red", dash="dash"))
    st.plotly_chart(fig, use_container_width=True)

    if "WELL_NAME" in g_conv.columns:
        cross = _well_weighted_crossplot_df(g_conv.rename(columns={"Kng_BC_model": "Kng_model"})).rename(columns={"Kng_model_wmean": "Kng_BC_wmean"})
        if not cross.empty:
            figw = px.scatter(
                cross,
                x="Кнг_W_wmean",
                y="Kng_BC_wmean",
                color="convergence_percent",
                color_continuous_scale="Turbo",
                hover_data={
                    "WELL_NAME": True,
                    "points": ":.3f",
                    "avg_weight": ":.3f",
                    "convergence_percent": ":.3f",
                },
                title=f"PVT {psel}: кроссплот по скважинам (БК, средневзвешенно)",
            )
            figw.update_traces(hovertemplate="Кнг_W_wmean=%{x:.3f}<br>Kng_BC_wmean=%{y:.3f}<extra></extra>")
            figw.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(color="red", dash="dash"))
            st.plotly_chart(figw, use_container_width=True)

        st.subheader("Просмотр скважины (Брукса-Кори)")
        wells = sorted(g_conv["WELL_NAME"].astype(str).unique().tolist())
        well = st.selectbox("Скважина", wells, key="bc_well_sel")
        wd = g_conv[g_conv["WELL_NAME"].astype(str) == well].copy()
        dcol = _pick_depth_column(wd)
        if dcol is None:
            st.warning("Не найдена колонка глубины для скважины.")
        elif "ACTNUM_GDM" not in wd.columns:
            st.warning("В данных отсутствует ACTNUM_GDM.")
        else:
            wd[dcol] = pd.to_numeric(wd[dcol], errors="coerce")
            wd = wd.dropna(subset=[dcol]).sort_values(dcol).reset_index(drop=True)
            curve = wd[[dcol, "ACTNUM_GDM", "Кнг_W", "Kng_BC_model"]].rename(columns={"Кнг_W": "Кн РИГИС", "Kng_BC_model": "Кн Брукса-Кори"})
            melt = curve.melt(id_vars=[dcol], value_vars=["ACTNUM_GDM", "Кн РИГИС", "Кн Брукса-Кори"], var_name="Кривая", value_name="Значение")
            fig_prof = px.line(melt, x="Значение", y=dcol, color="Кривая", title=f"Скважина {well}: вертикальный профиль (БК)")
            fig_prof.update_traces(hovertemplate="Значение=%{x:.3f}<br>Глубина=%{y:.3f}<br>Кривая=%{fullData.name}<extra></extra>")
            fig_prof.update_yaxes(autorange="reversed")
            st.plotly_chart(fig_prof, use_container_width=True)

def compare_methods_tab() -> None:
    st.title("Сравнение методов")
    st.caption("Сначала сохраните результаты по скважинам в вкладках J-функции и Брукса-Кори.")
    _render_methods_comparison_block(block_key="compare_tab")


def main() -> None:
    page = st.sidebar.radio("Раздел", options=["Лаборатория", "Подбор J функции Леверетта", "Брукса-Кори", "Сравнение методов"])
    prev_page = st.session_state.get("_active_page")
    if prev_page != page:
        # Не запоминаем связь код горизонта -> PVTNUM при переключении разделов
        st.session_state.pop("pvt_horizon_map", None)
        st.session_state.pop("bc_pvt_h_map", None)
        for k in list(st.session_state.keys()):
            ks = str(k)
            if ks.startswith("pvt_hor_map_") or ks.startswith("bc_map_"):
                st.session_state.pop(k, None)
        _scroll_page_top()
    st.session_state["_active_page"] = page
    if page == "Лаборатория":
        laboratory_tab()
    elif page == "Подбор J функции Леверетта":
        leverett_tab()
    elif page == "Брукса-Кори":
        brooks_corey_tab()
    else:
        compare_methods_tab()


if __name__ == "__main__":
    main()
