from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

from optimize import huber_loss, prepare_input_df, prepare_weights


@dataclass
class PowerBounds:
    a: tuple[float, float]
    b: tuple[float, float]


def _fit_power(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    m = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if m.sum() < 3:
        return np.nan, np.nan
    u = np.log(x[m])
    v = np.log(y[m])
    b, ln_a = np.polyfit(u, v, 1)
    return float(np.exp(ln_a)), float(b)


def _power(x: np.ndarray, a: float, b: float) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        y = a * (x**b)
    return np.nan_to_num(y, nan=np.nan, posinf=np.nan, neginf=np.nan)


def _envelopes(x: np.ndarray, y: np.ndarray) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    """
    Возвращает (lower_ab, upper_ab, center_ab) для степенной зависимости y=a*x^b
    с ограничением a>0 и b<0.
    """
    m = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if m.sum() < 5:
        return (1e-4, -1.0), (1.0, -0.1), (1e-2, -0.5)
    xx, yy = x[m].astype(float), y[m].astype(float)
    a0, b0 = _fit_power(xx, yy)
    if not np.isfinite(a0):
        a0 = float(np.median(yy))
    if not np.isfinite(b0) or b0 >= -1e-4:
        b0 = -0.35
    b0 = float(np.clip(b0, -12.0, -0.02))
    r = yy / (xx**b0)
    r = r[np.isfinite(r) & (r > 0)]
    if len(r) == 0:
        r = np.array([a0], dtype=float)
    alo = float(np.quantile(r, 0.05))
    ahi = float(np.quantile(r, 0.95))
    if alo <= 0:
        alo = max(1e-8, float(np.min(r[r > 0])) if np.any(r > 0) else 1e-6)
    if ahi <= alo:
        ahi = alo * 1.05
    lower = (alo, b0)
    upper = (ahi, b0)
    center = (float(a0), b0)
    return lower, upper, center


def auto_power_bounds(x: np.ndarray, y: np.ndarray, pad_a: float = 0.08, pad_b: float = 0.15) -> dict[str, Any]:
    low, up, center = _envelopes(x, y)
    amin = max(1e-8, low[0] * (1 - pad_a))
    amax = up[0] * (1 + pad_a)
    bmin = min(low[1], up[1], center[1]) - pad_b
    bmax = max(low[1], up[1], center[1]) + pad_b
    # Принудительно b<0
    bmax = min(bmax, -0.01)
    bmin = max(bmin, -12.0)
    if bmin >= bmax:
        bmin, bmax = center[1] - 0.3, min(center[1] + 0.1, -0.01)
    return {
        "bounds": PowerBounds((amin, amax), (bmin, bmax)),
        "center": center,
        "lower": low,
        "upper": up,
    }


def compute_soil_from_params(df: pd.DataFrame, params: dict[str, float]) -> np.ndarray:
    poro_col = "PORO_FRAC" if "PORO_FRAC" in df.columns else "PORO_GDM"
    poro = pd.to_numeric(df[poro_col], errors="coerce").to_numpy()
    pc = pd.to_numeric(df["PC"], errors="coerce").to_numpy()
    valid = np.isfinite(poro) & np.isfinite(pc) & (poro > 0) & (pc > 0)
    out = np.full(len(df), np.nan)
    if valid.sum() == 0:
        return out
    poro_v = poro[valid]
    pc_v = pc[valid]

    swl = np.clip(_power(poro_v, params["a_swl"], params["b_swl"]), 0.0, 1.0)
    perm_pred = np.clip(_power(np.clip(swl, 1e-8, None), params["a_perm"], params["b_perm"]), 1e-12, None)
    ratio = np.clip(perm_pred / np.clip(poro_v, 1e-8, None), 1e-12, None)
    pvit = np.clip(_power(ratio, params["a_pvit"], params["b_pvit"]), 1e-8, None)
    n_val = np.clip(_power(ratio, params["a_n"], params["b_n"]), 1e-3, None)

    with np.errstate(divide="ignore", invalid="ignore"):
        soil = swl + (1 - swl) * (pvit / pc_v) ** (1 / n_val)
    out[valid] = np.clip(soil, 0, 1)
    return out


def prepare_brooks_training_data(df_wells: pd.DataFrame, df_prod: pd.DataFrame | None) -> pd.DataFrame:
    """
    Такая же предобработка как для J-функции: prepare_input_df + prepare_weights.
    """
    data = prepare_input_df(df_wells)
    return prepare_weights(data, df_prod)


def _filter_target_like_j(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Кнг_W"] = pd.to_numeric(out["Кнг_W"], errors="coerce")
    out = out[out["Кнг_W"].notna() & (out["Кнг_W"] != 0)]
    if "WELL_NAME" not in out.columns or out.empty:
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
    return out.loc[sorted(set(keep_idx))]


def optimize_brooks_corey_for_region(
    df_region: pd.DataFrame,
    bounds: dict[str, PowerBounds],
    envelopes: dict[str, dict[str, tuple[float, float]]] | None = None,
    maxiter: int = 180,
    popsize: int = 18,
) -> dict[str, float]:
    """
    Подбор параметров методом differential_evolution
    с целевой функцией как у J-функции (взвешенный huber),
    и штрафом за выход оптимальной кривой за лабораторные огибающие.
    """
    train = _filter_target_like_j(df_region)
    if train.empty:
        return {}

    param_names = ["a_swl", "b_swl", "a_perm", "b_perm", "a_pvit", "b_pvit", "a_n", "b_n"]
    de_bounds = [
        bounds["swl"].a,
        bounds["swl"].b,
        bounds["perm"].a,
        bounds["perm"].b,
        bounds["pvit"].a,
        bounds["pvit"].b,
        bounds["n"].a,
        bounds["n"].b,
    ]
    y_true = pd.to_numeric(train["Кнг_W"], errors="coerce").to_numpy()
    w = pd.to_numeric(train.get("weight", 1.0), errors="coerce").fillna(1.0).to_numpy()

    def _penalty_for_envelope(p: dict[str, float]) -> float:
        if not envelopes:
            return 0.0
        pen = 0.0
        hard_max = 0.0

        def _viol(y: np.ndarray, ylo: np.ndarray, yhi: np.ndarray) -> tuple[float, float]:
            v = np.maximum(0.0, ylo - y) + np.maximum(0.0, y - yhi)
            return float(np.nanmean(v)), float(np.nanmax(v))

        for key, a_name, b_name in [
            ("swl", "a_swl", "b_swl"),
            ("perm", "a_perm", "b_perm"),
            ("pvit", "a_pvit", "b_pvit"),
            ("n", "a_n", "b_n"),
        ]:
            if key not in envelopes:
                continue
            env = envelopes[key]
            x = np.asarray(env.get("x"), dtype=float)
            x = x[np.isfinite(x) & (x > 0)]
            if len(x) == 0:
                continue
            lo_a, lo_b = env["lower"]
            up_a, up_b = env["upper"]
            y = _power(x, p[a_name], p[b_name])
            ylo = _power(x, lo_a, lo_b)
            yhi = _power(x, up_a, up_b)
            avg_v, max_v = _viol(y, np.minimum(ylo, yhi), np.maximum(ylo, yhi))
            pen += avg_v
            hard_max = max(hard_max, max_v)

        # Жесткий барьер: если где-то вышли за огибающие — решение недопустимо.
        if hard_max > 1e-9:
            return float(1e9 + 1e9 * hard_max)
        return float(1e6 * pen)

    def loss(vec: np.ndarray) -> float:
        p = {k: float(v) for k, v in zip(param_names, vec)}
        # Жесткие физические ограничения
        if any(p[k] <= 0 for k in ["a_swl", "a_perm", "a_pvit", "a_n"]):
            return 1e12
        if any(p[k] >= 0 for k in ["b_swl", "b_perm", "b_pvit", "b_n"]):
            return 1e12
        pred = compute_soil_from_params(train, p)
        m = np.isfinite(pred) & np.isfinite(y_true)
        if m.sum() == 0:
            return 1e12
        r = pred[m] - y_true[m]
        base = float(np.sum(w[m] * huber_loss(r)))
        return base + _penalty_for_envelope(p)

    res = differential_evolution(loss, bounds=de_bounds, maxiter=maxiter, popsize=popsize, seed=42)
    return {k: float(v) for k, v in zip(param_names, res.x)}
