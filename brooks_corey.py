from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, dual_annealing

from optimize import huber_loss, prepare_input_df, prepare_weights

DEFAULT_PERM_MAX_MD = 5000.0


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


def swl_a_exp_b_poro(poro: np.ndarray, a: float, b: float) -> np.ndarray:
    """swl = a * exp(b * poro), poro — пористость (доля); результат обрезается до (0, 1]."""
    p = np.asarray(poro, dtype=float)
    z = np.clip(p * float(b), -60.0, 60.0)
    with np.errstate(over="ignore"):
        y = float(a) * np.exp(z)
    return np.clip(y, 1e-12, 1.0)


def _sanitize_swl_exp_envelope_ab(a: float, b: float) -> tuple[float, float]:
    """
    Экспоненциальная swl(Кп) = a·exp(b·Кп): **a > 0**, **b < 0** (как физ. ветка для коридора/оптимизации).
    """
    aa = float(a)
    if not np.isfinite(aa) or aa <= 0:
        aa = 0.15
    else:
        aa = float(np.clip(aa, 1e-4, 25.0))
    bb = float(b)
    if not np.isfinite(bb) or bb >= -1e-12:
        bb = -0.5
    bb = float(np.clip(bb, -80.0, -1e-4))
    return aa, bb


def _ensure_swl_exp_upper_a_above_lower(
    lower: tuple[float, float],
    upper: tuple[float, float],
) -> tuple[float, float]:
    """Верхняя огибающая по a выше нижней; обе пары с a>0, b<0."""
    lo_a, _lo_b = lower
    up_a, up_b = upper
    if up_a <= lo_a:
        up_a = lo_a * (1.0 + 1e-6) if lo_a > 0 else 1.05e-4
    up_a, up_b = _sanitize_swl_exp_envelope_ab(up_a, up_b)
    return (up_a, up_b)


def _ensure_swl_exp_upper_b_not_flatter_than_lower(
    lower: tuple[float, float],
    upper: tuple[float, float],
    poro_ref: float,
    min_abs_b_ratio: float = 0.72,
) -> tuple[float, float]:
    """
    У верхней эксп. огибающей модуль наклона по Кп (|b| в y=a·exp(b·Кп)) не сильно меньше, чем у нижней:
    |b_up| >= min_abs_b_ratio * |b_lo|. При увеличении крутизны b подбирается a так, чтобы значение при Кп=poro_ref
    сохранилось (коридор не «схлопывается» в почти горизонтальную верхнюю дугу).
    """
    _lo_a, lo_b = lower
    up_a, up_b = upper
    mlo = abs(float(lo_b))
    mup = abs(float(up_b))
    if mlo < 1e-10 or mup >= mlo * float(min_abs_b_ratio):
        return _sanitize_swl_exp_envelope_ab(up_a, up_b)
    b_new = float(-mlo * float(min_abs_b_ratio))
    b_new = float(np.clip(b_new, -80.0, -1e-4))
    phi = float(poro_ref) if np.isfinite(poro_ref) and poro_ref > 0 else 0.15
    phi = float(np.clip(phi, 1e-6, 1.0))
    z_old = float(np.clip(phi * float(up_b), -60.0, 60.0))
    z_new = float(np.clip(phi * b_new, -60.0, 60.0))
    with np.errstate(over="ignore"):
        y_ref = float(max(float(up_a) * np.exp(z_old), 1e-18))
        a_new = float(y_ref / max(np.exp(z_new), 1e-300))
    return _sanitize_swl_exp_envelope_ab(a_new, b_new)


def _fit_swl_exp_ab_center(xx: np.ndarray, yy: np.ndarray) -> tuple[float, float]:
    """Оценка (a, b) по ln(swl) ≈ ln(a) + b*poro."""
    m = np.isfinite(xx) & np.isfinite(yy) & (xx > 0) & (yy > 1e-12)
    if m.sum() < 3:
        return _sanitize_swl_exp_envelope_ab(0.15, -0.5)
    xs = xx[m].astype(float)
    ys = np.clip(yy[m].astype(float), 1e-12, 1.0)
    try:
        b0, ln_a = np.polyfit(xs, np.log(ys), 1)
        a0 = float(np.exp(ln_a))
        return _sanitize_swl_exp_envelope_ab(a0, float(b0))
    except Exception:
        return _sanitize_swl_exp_envelope_ab(0.15, -0.5)


def auto_exp_bounds_swl_poro(
    x: np.ndarray,
    y: np.ndarray,
    pad_a: float = 0.2,
    pad_b: float = 3.0,
    upper_b_min_ratio: float = 0.72,
) -> dict[str, Any]:
    """
    Границы и огибающие для swl(poro) = a*exp(b*poro).
    lower/upper — пары (a, b): **a > 0**, **b < 0**; у верхней **a** не ниже, чем у нижней.
    У верхней огибающей **|b|** не сильно меньше, чем у нижней (не «положе» по наклону): не ниже
    ``upper_b_min_ratio * |b_lo|``; при подгонке **b** масштабируется **a** по медиане Кп.
    """
    a0, b0 = _fit_swl_exp_ab_center(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    alo = max(1e-4, a0 * (1.0 - pad_a))
    ahi = min(25.0, a0 * (1.0 + pad_a))
    if ahi <= alo:
        ahi = min(25.0, alo * 1.1)
    # сырые b вокруг центра — затем принудительно b<0 у огибающих
    blo = float(np.clip(b0 - pad_b, -80.0, -1e-4))
    bhi = float(np.clip(b0 + pad_b, -80.0, -1e-4))
    if blo >= bhi:
        blo, bhi = float(np.clip(b0 - 1.0, -80.0, -1e-4)), float(np.clip(b0 + 1.0, -80.0, -1e-4))
    if blo >= bhi:
        blo, bhi = -5.0, -0.01
    lower = _sanitize_swl_exp_envelope_ab(alo, blo)
    upper = _sanitize_swl_exp_envelope_ab(ahi, bhi)
    center = _sanitize_swl_exp_envelope_ab(a0, b0)
    upper = _ensure_swl_exp_upper_a_above_lower(lower, upper)
    xv = np.asarray(x, dtype=float)
    poro_ref = float(np.nanmedian(xv[np.isfinite(xv) & (xv > 0)])) if np.any(np.isfinite(xv) & (xv > 0)) else 0.15
    upper = _ensure_swl_exp_upper_b_not_flatter_than_lower(
        lower, upper, poro_ref, min_abs_b_ratio=float(upper_b_min_ratio)
    )
    upper = _ensure_swl_exp_upper_a_above_lower(lower, upper)
    lo_a, lo_b = lower
    up_a, up_b = upper
    ce_a, ce_b = center
    b_de_lo = float(min(lo_b, up_b, ce_b) - 2.0)
    b_de_hi = float(max(lo_b, up_b, ce_b) + 2.0)
    b_de_lo = max(b_de_lo, -80.0)
    b_de_hi = min(b_de_hi, -1e-4)
    if b_de_lo >= b_de_hi:
        b_de_lo, b_de_hi = -5.0, -0.01

    a_de_lo = max(1e-4, min(lo_a, up_a, ce_a) * 0.45)
    a_de_hi = min(25.0, max(lo_a, up_a, ce_a) * 1.25)
    if a_de_lo >= a_de_hi:
        a_de_lo, a_de_hi = 1e-4, min(25.0, max(lo_a, up_a) * 1.2)

    return {
        "bounds": PowerBounds((a_de_lo, a_de_hi), (b_de_lo, b_de_hi)),
        "center": center,
        "lower": lower,
        "upper": upper,
    }


def _sanitize_power_envelope_ab(a: float, b: float) -> tuple[float, float]:
    """
    Степенная огибающая y = a·x^b в цепочке БК (perm, pvit, n): **a > 0**, **b < 0**.
    """
    aa = float(a)
    if not np.isfinite(aa) or aa <= 0:
        aa = 1e-4
    else:
        aa = max(aa, 1e-12)
    bb = float(b)
    if not np.isfinite(bb) or bb >= -1e-12:
        bb = -0.35
    bb = float(np.clip(bb, -12.0, -0.02))
    return aa, bb


def _ensure_upper_a_above_lower(
    lower: tuple[float, float],
    upper: tuple[float, float],
) -> tuple[float, float]:
    """Верхняя граница по a не ниже нижней (после санитизации), b уже < 0."""
    lo_a, lo_b = lower
    up_a, up_b = upper
    if up_a <= lo_a:
        up_a = lo_a * (1.0 + 1e-6) if lo_a > 0 else 1.05e-4
    up_a, up_b = _sanitize_power_envelope_ab(up_a, up_b)
    return (up_a, up_b)


def _envelopes(x: np.ndarray, y: np.ndarray) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    """
    Возвращает (lower_ab, upper_ab, center_ab) для степенной зависимости y=a*x^b.

    Нижняя и **верхняя** огибающие всегда с **a > 0** и **b < 0** (как для ветвей perm / pvit / n в БК).
    Для верхней дополнительно гарантируется up_a > lo_a, чтобы коридор был ненулевым по амплитуде.
    """
    m = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if m.sum() < 5:
        lo = _sanitize_power_envelope_ab(1e-4, -1.0)
        up = _sanitize_power_envelope_ab(1.0, -0.1)
        ce = _sanitize_power_envelope_ab(1e-2, -0.5)
        up = _ensure_upper_a_above_lower(lo, up)
        return lo, up, ce
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
    lower = _sanitize_power_envelope_ab(alo, b0)
    upper = _sanitize_power_envelope_ab(ahi, b0)
    center = _sanitize_power_envelope_ab(float(a0), b0)
    upper = _ensure_upper_a_above_lower(lower, upper)
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
    """
    Цепочка БК:
    1) swl(Кп) = a*exp(b*Кп), обрезка до (0,1].
    2) Кпр(Кво) — степень.
    3) Pvit(√(Кпр/Кп)), n(√(Кпр/Кп)) — степени; Кпр ограничена сверху (мД).
    4) Swat по Corey; если Swat>1 или Pc=0 → Swat=1; если Swat<0 → Swat=0.
    5) Soil = 1 - Swat (сопоставление с Кнг нефти).
    """
    poro_col = "PORO_FRAC" if "PORO_FRAC" in df.columns else "PORO_GDM"
    poro = pd.to_numeric(df[poro_col], errors="coerce").to_numpy()
    pc = pd.to_numeric(df["PC"], errors="coerce").to_numpy()
    valid = np.isfinite(poro) & np.isfinite(pc) & (poro > 0)
    out = np.full(len(df), np.nan)
    if valid.sum() == 0:
        return out
    poro_v = poro[valid]
    pc_v = pc[valid]

    perm_max = float(params.get("perm_max_md", DEFAULT_PERM_MAX_MD))
    if not np.isfinite(perm_max) or perm_max <= 0:
        perm_max = DEFAULT_PERM_MAX_MD

    # 1) swl = a*exp(b*poro)
    kvo = swl_a_exp_b_poro(poro_v, params["a_swl"], params["b_swl"])
    # 2) Кпр(Кво)
    perm_pred = np.clip(
        _power(np.clip(kvo, 1e-8, 1.0 - 1e-8), params["a_perm"], params["b_perm"]),
        1e-12,
        perm_max,
    )
    # 3) √(Кпр/Кп)
    ratio = np.sqrt(np.clip(perm_pred / np.clip(poro_v, 1e-8, None), 1e-12, None))
    pvit = np.clip(_power(ratio, params["a_pvit"], params["b_pvit"]), 1e-8, None)
    n_val = np.clip(_power(ratio, params["a_n"], params["b_n"]), 1e-3, None)

    with np.errstate(divide="ignore", invalid="ignore"):
        swat = kvo + (1.0 - kvo) * (pvit / np.where(pc_v > 0, pc_v, np.nan)) ** (1.0 / n_val)
    swat = np.where(~np.isfinite(swat) | (pc_v == 0) | (swat > 1.0), 1.0, swat)
    swat = np.where(swat < 0.0, 0.0, swat)
    swat = np.clip(swat, 0.0, 1.0)
    soil = 1.0 - swat
    out[valid] = np.clip(soil, 0.0, 1.0)
    return out


def evaluate_brooks_score(df: pd.DataFrame, params: dict[str, float]) -> float:
    y_true = pd.to_numeric(df["Кнг_W"], errors="coerce").to_numpy()
    w = pd.to_numeric(df.get("weight", 1.0), errors="coerce").fillna(1.0).to_numpy()
    y_pred = compute_soil_from_params(df, params)
    m = np.isfinite(y_true) & np.isfinite(y_pred) & np.isfinite(w)
    if m.sum() == 0:
        return -np.inf
    denom = float(np.sum(w[m]))
    if denom <= 0:
        return -np.inf
    return float(1 - (np.sum(w[m] * np.abs(y_pred[m] - y_true[m])) / denom))


def envelope_max_violation(
    params: dict[str, float],
    envelopes: dict[str, dict[str, Any]] | None,
) -> float:
    """
    Максимальное нарушение огибающих для 4 зависимостей.
    0.0 -> внутри огибающих.
    """
    if not envelopes:
        return 0.0
    vmax = 0.0
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
        if key == "swl" and env.get("kind") == "exp_ab":
            y = swl_a_exp_b_poro(x, float(params[a_name]), float(params[b_name]))
            ylo = swl_a_exp_b_poro(x, float(lo_a), float(lo_b))
            yhi = swl_a_exp_b_poro(x, float(up_a), float(up_b))
        else:
            y = _power(x, float(params[a_name]), float(params[b_name]))
            ylo = _power(x, lo_a, lo_b)
            yhi = _power(x, up_a, up_b)
        v = np.maximum(0.0, np.minimum(ylo, yhi) - y) + np.maximum(0.0, y - np.maximum(ylo, yhi))
        if np.isfinite(v).any():
            vmax = max(vmax, float(np.nanmax(v)))
    return vmax


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


def _clip_bc_vec(vec: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
    out = np.clip(np.asarray(vec, dtype=float), lb, ub)
    for idx in (3, 5, 7):
        out[idx] = min(float(out[idx]), -1e-3)
    return out


def _bc_pso_optimize(
    loss: Callable[[np.ndarray], float],
    de_bounds: list[tuple[float, float]],
    *,
    maxiter: int,
    popsize: int,
    seed: int = 42,
) -> np.ndarray:
    """Рой частиц по тем же границам и loss, что и для DE (8 параметров БК)."""
    lb = np.array([b[0] for b in de_bounds], dtype=float)
    ub = np.array([b[1] for b in de_bounds], dtype=float)
    dim = len(de_bounds)
    rng = np.random.default_rng(seed)
    particles = int(max(popsize, 12, 2 * dim))
    x = rng.uniform(lb, ub, size=(particles, dim))
    for i in range(particles):
        x[i] = _clip_bc_vec(x[i], lb, ub)
    v = np.zeros_like(x)
    pbest = x.copy()
    pbest_val = np.array([loss(xx) for xx in pbest], dtype=float)
    gidx = int(np.argmin(pbest_val))
    gbest = pbest[gidx].copy()
    gbest_val = float(pbest_val[gidx])
    w_pso, c1, c2 = 0.72, 1.49, 1.49
    for _ in range(maxiter):
        r1 = rng.random((particles, dim))
        r2 = rng.random((particles, dim))
        v = w_pso * v + c1 * r1 * (pbest - x) + c2 * r2 * (gbest - x)
        x = x + v
        for i in range(particles):
            x[i] = _clip_bc_vec(x[i], lb, ub)
        vals = np.array([loss(xx) for xx in x], dtype=float)
        better = vals < pbest_val
        pbest[better] = x[better]
        pbest_val[better] = vals[better]
        cur_idx = int(np.argmin(pbest_val))
        cur_val = float(pbest_val[cur_idx])
        if cur_val < gbest_val:
            gbest_val = cur_val
            gbest = pbest[cur_idx].copy()
    return gbest


def optimize_brooks_corey_for_region(
    df_region: pd.DataFrame,
    bounds: dict[str, PowerBounds],
    envelopes: dict[str, dict[str, Any]] | None = None,
    maxiter: int = 180,
    popsize: int = 18,
    initial_guess: dict[str, tuple[float, float]] | None = None,
    baseline_params: dict[str, float] | None = None,
    perm_max_md: float = DEFAULT_PERM_MAX_MD,
    optimizer_method: str = "differential_evolution",
) -> dict[str, float]:
    """
    Подбор параметров глобальной оптимизацией (differential_evolution / dual_annealing / pso):
    взвешенный Huber по невязке Кнг + штраф за огибающие лаборатории.
    """
    train = _filter_target_like_j(df_region)
    if train.empty:
        return {}

    poro_col = "PORO_FRAC" if "PORO_FRAC" in train.columns else "PORO_GDM"
    poro_all = pd.to_numeric(train[poro_col], errors="coerce").to_numpy(dtype=float)
    poro_pos = poro_all[np.isfinite(poro_all) & (poro_all > 0)]
    if len(poro_pos) == 0:
        return {}
    resolved_perm_max = float(perm_max_md) if np.isfinite(perm_max_md) and perm_max_md > 0 else DEFAULT_PERM_MAX_MD

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
            if key == "swl" and env.get("kind") == "exp_ab":
                y = swl_a_exp_b_poro(x, p[a_name], p[b_name])
                ylo = swl_a_exp_b_poro(x, lo_a, lo_b)
                yhi = swl_a_exp_b_poro(x, up_a, up_b)
            else:
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
        p["perm_max_md"] = resolved_perm_max
        # Жесткие физические ограничения
        if not (1e-9 < p["a_swl"] <= 100.0):
            return 1e12
        if not np.isfinite(p["b_swl"]) or p["b_swl"] >= 0 or p["b_swl"] < -120.0:
            return 1e12
        if any(p[k] <= 0 for k in ["a_perm", "a_pvit", "a_n"]):
            return 1e12
        if any(p[k] >= 0 for k in ["b_perm", "b_pvit", "b_n"]):
            return 1e12
        pred = compute_soil_from_params(train, p)
        m = np.isfinite(pred) & np.isfinite(y_true)
        if m.sum() == 0:
            return 1e12
        r = pred[m] - y_true[m]
        base = float(np.sum(w[m] * huber_loss(r)))
        return base + _penalty_for_envelope(p)

    # Первое приближение из корреляционных зависимостей (по облакам лабораторных точек)
    init_guess = initial_guess or {}
    x0 = np.array(
        [
            init_guess.get("swl", (np.mean(bounds["swl"].a), np.mean(bounds["swl"].b)))[0],
            init_guess.get("swl", (np.mean(bounds["swl"].a), np.mean(bounds["swl"].b)))[1],
            init_guess.get("perm", (np.mean(bounds["perm"].a), np.mean(bounds["perm"].b)))[0],
            init_guess.get("perm", (np.mean(bounds["perm"].a), np.mean(bounds["perm"].b)))[1],
            init_guess.get("pvit", (np.mean(bounds["pvit"].a), np.mean(bounds["pvit"].b)))[0],
            init_guess.get("pvit", (np.mean(bounds["pvit"].a), np.mean(bounds["pvit"].b)))[1],
            init_guess.get("n", (np.mean(bounds["n"].a), np.mean(bounds["n"].b)))[0],
            init_guess.get("n", (np.mean(bounds["n"].a), np.mean(bounds["n"].b)))[1],
        ],
        dtype=float,
    )
    lb = np.array([b[0] for b in de_bounds], dtype=float)
    ub = np.array([b[1] for b in de_bounds], dtype=float)
    x0 = _clip_bc_vec(x0, lb, ub)

    method = (optimizer_method or "differential_evolution").lower()
    if method == "pso":
        res_x = _bc_pso_optimize(loss, de_bounds, maxiter=maxiter, popsize=popsize, seed=42)
    elif method == "dual_annealing":
        da_res = dual_annealing(loss, bounds=de_bounds, maxiter=maxiter, seed=42)
        res_x = _clip_bc_vec(np.asarray(da_res.x, dtype=float), lb, ub)
    else:
        rng = np.random.default_rng(42)
        n_dim = len(de_bounds)
        n_pop = max(popsize * n_dim, 24)
        init_pop = rng.uniform(lb, ub, size=(n_pop, n_dim))
        init_pop[0] = x0
        for i in range(1, min(6, n_pop)):
            jitter = rng.normal(0.0, 0.05, size=n_dim) * (ub - lb)
            init_pop[i] = _clip_bc_vec(x0 + jitter, lb, ub)
        res = differential_evolution(
            loss,
            bounds=de_bounds,
            maxiter=maxiter,
            popsize=popsize,
            seed=42,
            init=init_pop,
        )
        res_x = np.asarray(res.x, dtype=float)

    best = {k: float(v) for k, v in zip(param_names, res_x)}
    best["perm_max_md"] = resolved_perm_max
    best_val = float(loss(np.array([best[k] for k in param_names], dtype=float)))

    # Fallback: сравниваем с базовыми наборами коэффициентов и берём лучший
    candidates: list[dict[str, float]] = []
    if baseline_params:
        bp = {k: float(baseline_params[k]) for k in param_names if k in baseline_params}
        bp["perm_max_md"] = resolved_perm_max
        candidates.append(bp)
    if initial_guess:
        ig = {
            "a_swl": float(initial_guess.get("swl", (np.mean(bounds["swl"].a), np.mean(bounds["swl"].b)))[0]),
            "b_swl": float(initial_guess.get("swl", (np.mean(bounds["swl"].a), np.mean(bounds["swl"].b)))[1]),
            "a_perm": float(initial_guess.get("perm", (np.mean(bounds["perm"].a), np.mean(bounds["perm"].b)))[0]),
            "b_perm": float(initial_guess.get("perm", (np.mean(bounds["perm"].a), np.mean(bounds["perm"].b)))[1]),
            "a_pvit": float(initial_guess.get("pvit", (np.mean(bounds["pvit"].a), np.mean(bounds["pvit"].b)))[0]),
            "b_pvit": float(initial_guess.get("pvit", (np.mean(bounds["pvit"].a), np.mean(bounds["pvit"].b)))[1]),
            "a_n": float(initial_guess.get("n", (np.mean(bounds["n"].a), np.mean(bounds["n"].b)))[0]),
            "b_n": float(initial_guess.get("n", (np.mean(bounds["n"].a), np.mean(bounds["n"].b)))[1]),
            "perm_max_md": resolved_perm_max,
        }
        candidates.append(ig)

    lb = np.array([b[0] for b in de_bounds], dtype=float)
    ub = np.array([b[1] for b in de_bounds], dtype=float)
    for cand in candidates:
        if len(cand) < len(param_names):
            continue
        vec = np.array([cand[k] for k in param_names], dtype=float)
        vec = np.clip(vec, lb, ub)
        for idx in [3, 5, 7]:
            vec[idx] = min(vec[idx], -1e-3)
        val = float(loss(vec))
        if val < best_val:
            best_val = val
            best = {k: float(v) for k, v in zip(param_names, vec)}
            best["perm_max_md"] = resolved_perm_max

    return best
