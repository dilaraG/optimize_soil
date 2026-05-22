from functools import partial
from typing import Any

import time

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, dual_annealing
from sklearn.metrics import r2_score

REQUIRED_WELL_COLUMNS = [
    "WELL_NAME",
    "PVTNUM_GDM",
    "PORO_GDM",
    "PERM_GDM",
    "PC",
    "SWL_GDM",
    "Кнг_W",
]

DEFAULT_LOW_SWN_THRESHOLD = 0.01
DEFAULT_J_CAP_AT_LOW_SWN = 20.0


def prepare_weights(df: pd.DataFrame, df_j: pd.DataFrame | None) -> pd.DataFrame:
    df = df.copy()
    df["weight"] = 1.0

    if "Perf_GDM" in df.columns:
        perf = pd.to_numeric(df["Perf_GDM"], errors="coerce").fillna(0)
        df["weight"] *= np.where(perf == 1, 2.0, 1.0)

    if df_j is None or df_j.empty:
        return df

    prod = df_j.copy()
    prod = prod.rename(
        columns={
            "Ствол скважины": "WELL_NAME",
            "Экспл. объект": "PVTNUM_GDM",
            "Нак. добыча нефти, ст.м³": "CUM_OIL",
            "Нак. добыча нефти, т": "CUM_OIL",
        }
    )

    if not {"WELL_NAME", "PVTNUM_GDM", "CUM_OIL"}.issubset(prod.columns):
        return df

    prod["CUM_OIL"] = pd.to_numeric(prod["CUM_OIL"], errors="coerce")
    prod = prod.dropna(subset=["CUM_OIL"])
    if prod.empty:
        return df

    prod["rank"] = prod.groupby("PVTNUM_GDM")["CUM_OIL"].rank(ascending=False)
    max_rank = prod["rank"].max()
    if max_rank > 1:
        prod["rank_norm"] = 1 + 9 * (prod["rank"] - 1) / (max_rank - 1)
    else:
        prod["rank_norm"] = 1.0

    rank_map = prod[["WELL_NAME", "PVTNUM_GDM", "rank_norm"]].drop_duplicates()
    df = df.merge(rank_map, on=["WELL_NAME", "PVTNUM_GDM"], how="left")
    df["rank_norm"] = df["rank_norm"].fillna(1.0)
    df["weight"] *= df["rank_norm"]
    return df


def _safe_series(df: pd.DataFrame, col: str) -> np.ndarray:
    return pd.to_numeric(df[col], errors="coerce").to_numpy()


def calc_kng_vector(df: pd.DataFrame, a: float, b: float, sigma: float) -> np.ndarray:
    poro_col = "PORO_FRAC" if "PORO_FRAC" in df.columns else "PORO_GDM"
    poro = _safe_series(df, poro_col)
    perm = _safe_series(df, "PERM_GDM")
    pc = _safe_series(df, "PC")
    swl = _safe_series(df, "SWL_GDM")

    valid = (poro > 0) & (perm > 0) & np.isfinite(pc) & np.isfinite(swl)
    kng = np.full(len(df), np.nan)
    if valid.sum() == 0:
        return kng

    poro = poro[valid]
    perm = perm[valid]
    pc = pc[valid]
    swl = swl[valid]

    j = np.pi * pc / sigma * np.sqrt(perm / poro)
    kvn = (j / a) ** (1 / b)
    kv = swl + (1 - swl) * kvn
    kng_valid = 1 - kv
    kng_valid = np.clip(kng_valid, 0, 1)
    kng[valid] = kng_valid
    return kng


def huber_loss(r: np.ndarray, delta: float = 0.1) -> np.ndarray:
    abs_r = np.abs(r)
    mask = abs_r <= delta
    return np.where(mask, 0.5 * r**2, delta * (abs_r - 0.5 * delta))


def j_power_from_swn(swn: np.ndarray, a: float, b: float) -> np.ndarray:
    """Степенная J(Swn) = a·Swn^b."""
    with np.errstate(over="ignore", invalid="ignore"):
        return a * (np.asarray(swn, dtype=float) ** b)


def apply_low_swn_j_cap(
    swn: np.ndarray,
    j: np.ndarray,
    *,
    swn_threshold: float = DEFAULT_LOW_SWN_THRESHOLD,
    j_cap: float = DEFAULT_J_CAP_AT_LOW_SWN,
) -> np.ndarray:
    """При Swn ≤ порога ограничивает J сверху (по умолчанию J ≤ 20 при Swn ≤ 0.01)."""
    s = np.asarray(swn, dtype=float)
    y = np.asarray(j, dtype=float).copy()
    thr = float(swn_threshold)
    cap = float(j_cap)
    if not (np.isfinite(thr) and thr > 0 and np.isfinite(cap) and cap > 0):
        return y
    low = s <= thr
    if np.any(low):
        y[low] = np.minimum(y[low], cap)
    return y


def _low_swn_j_cap_penalty(
    a: float,
    b: float,
    *,
    swn_threshold: float = DEFAULT_LOW_SWN_THRESHOLD,
    j_cap: float = DEFAULT_J_CAP_AT_LOW_SWN,
    n_grid: int = 24,
    scale: float = 2e5,
) -> float:
    """Штраф, если некапированная J = a·Swn^b превышает j_cap при Swn ≤ порога."""
    thr = float(swn_threshold)
    cap = float(j_cap)
    if not (np.isfinite(a) and np.isfinite(b) and np.isfinite(thr) and thr > 0 and np.isfinite(cap) and cap > 0):
        return 0.0
    s0 = max(1e-9, 1e-4)
    if s0 >= thr:
        return 0.0
    grid = np.logspace(np.log10(s0), np.log10(thr), int(max(6, min(n_grid, 80))))
    raw = j_power_from_swn(grid, a, b)
    if not np.any(np.isfinite(raw)):
        return 0.0
    viol = np.clip(raw - cap, 0.0, np.inf)
    viol = viol[np.isfinite(viol)]
    if viol.size == 0:
        return 0.0
    return float(scale * np.mean(viol**2))


def _j_power_envelope_penalty(
    a: float,
    b: float,
    env: dict[str, Any] | None,
    *,
    n_grid: int = 48,
    scale: float = 2e5,
    swn_threshold: float = DEFAULT_LOW_SWN_THRESHOLD,
    j_cap: float = DEFAULT_J_CAP_AT_LOW_SWN,
) -> float:
    """
    Штраф, если степенная J_lab(Swn) = a·Swn^b выходит за лабораторные нижнюю/верхнюю огибающие
    на отрезке [s_min, s_max] облака (коридор по вертикали в каждой точке сетки).
    """
    if not env:
        return 0.0
    lo = env.get("lower") or {}
    up = env.get("upper") or {}
    try:
        alo, blo = float(lo["a"]), float(lo["b"])
        ahi, bhi = float(up["a"]), float(up["b"])
    except (KeyError, TypeError, ValueError):
        return 0.0
    if not all(np.isfinite([a, b, alo, blo, ahi, bhi])):
        return 0.0
    s0 = float(env.get("s_min", 1e-4))
    s1 = float(env.get("s_max", 1.0))
    if not (np.isfinite(s0) and np.isfinite(s1)) or s1 <= 0:
        return 0.0
    s0 = max(s0, 1e-9)
    s1 = max(s1, s0 * 1.01)
    grid = np.logspace(np.log10(s0), np.log10(s1), int(max(8, min(n_grid, 120))))
    thr = float(swn_threshold)
    cap = float(j_cap)
    if np.isfinite(thr) and thr > 0 and s0 <= thr <= s1:
        grid = np.unique(np.sort(np.concatenate([grid, [thr]])))
    with np.errstate(over="ignore", invalid="ignore"):
        y_lo_b = alo * (grid**blo)
        y_hi_b = ahi * (grid**bhi)
        j_lo = np.minimum(y_lo_b, y_hi_b)
        j_hi = np.maximum(y_lo_b, y_hi_b)
        j_opt = apply_low_swn_j_cap(grid, j_power_from_swn(grid, a, b), swn_threshold=thr, j_cap=cap)
    viol_lo = np.clip(j_lo - j_opt, 0.0, np.inf)
    viol_hi = np.clip(j_opt - j_hi, 0.0, np.inf)
    return float(scale * np.mean(viol_lo**2 + viol_hi**2))


def loss_function(
    params: tuple[float, float, float],
    df: pd.DataFrame,
    j_envelope: dict[str, Any] | None = None,
    *,
    low_swn_threshold: float = DEFAULT_LOW_SWN_THRESHOLD,
    j_cap_at_low_swn: float = DEFAULT_J_CAP_AT_LOW_SWN,
) -> float:
    a, b, sigma = params
    kng_model = calc_kng_vector(df, a, b, sigma)
    kng_true = _safe_series(df, "Кнг_W")
    weights = pd.to_numeric(df.get("weight", 1.0), errors="coerce").fillna(1.0).to_numpy()
    valid = ~np.isnan(kng_model) & np.isfinite(kng_true)
    if not np.any(valid):
        return 1e9
    r = kng_model[valid] - kng_true[valid]
    base = float(np.sum(weights[valid] * huber_loss(r)))
    env_pen = _j_power_envelope_penalty(
        a, b, j_envelope, swn_threshold=low_swn_threshold, j_cap=j_cap_at_low_swn
    )
    cap_pen = _low_swn_j_cap_penalty(a, b, swn_threshold=low_swn_threshold, j_cap=j_cap_at_low_swn)
    return base + env_pen + cap_pen


def _make_loss_fn(
    g: pd.DataFrame,
    j_envelope: dict[str, Any] | None,
    *,
    low_swn_threshold: float,
    j_cap_at_low_swn: float,
):
    return partial(
        loss_function,
        df=g,
        j_envelope=j_envelope,
        low_swn_threshold=low_swn_threshold,
        j_cap_at_low_swn=j_cap_at_low_swn,
    )


def _pso_optimize(
    g: pd.DataFrame,
    bounds: list[tuple[float, float]],
    maxiter: int = 120,
    particles: int = 28,
    j_envelope: dict[str, Any] | None = None,
    *,
    low_swn_threshold: float = DEFAULT_LOW_SWN_THRESHOLD,
    j_cap_at_low_swn: float = DEFAULT_J_CAP_AT_LOW_SWN,
) -> tuple[float, float, float]:
    dim = 3
    lb = np.array([b[0] for b in bounds], dtype=float)
    ub = np.array([b[1] for b in bounds], dtype=float)
    rng = np.random.default_rng(42)
    x = rng.uniform(lb, ub, size=(particles, dim))
    v = np.zeros_like(x)
    pbest = x.copy()
    loss_fn = _make_loss_fn(
        g, j_envelope, low_swn_threshold=low_swn_threshold, j_cap_at_low_swn=j_cap_at_low_swn
    )
    pbest_val = np.array([loss_fn(tuple(xx)) for xx in x], dtype=float)
    gidx = int(np.argmin(pbest_val))
    gbest = pbest[gidx].copy()
    gbest_val = float(pbest_val[gidx])

    w, c1, c2 = 0.72, 1.49, 1.49
    for _ in range(maxiter):
        r1 = rng.random((particles, dim))
        r2 = rng.random((particles, dim))
        v = w * v + c1 * r1 * (pbest - x) + c2 * r2 * (gbest - x)
        x = np.clip(x + v, lb, ub)
        vals = np.array([loss_fn(tuple(xx)) for xx in x], dtype=float)
        better = vals < pbest_val
        pbest[better] = x[better]
        pbest_val[better] = vals[better]
        cur_idx = int(np.argmin(pbest_val))
        cur_val = float(pbest_val[cur_idx])
        if cur_val < gbest_val:
            gbest_val = cur_val
            gbest = pbest[cur_idx].copy()
    return float(gbest[0]), float(gbest[1]), float(gbest[2])


def optimize_pvt(
    df: pd.DataFrame,
    bounds_by_pvt: dict[int, dict[str, tuple[float, float]]],
    maxiter: int = 200,
    popsize: int = 20,
    fixed_params: dict[int, tuple[float, float, float]] | None = None,
    optimizer_method: str = "differential_evolution",
    j_envelope_by_pvt: dict[int, dict[str, Any]] | None = None,
    timing_rows: list[dict[str, Any]] | None = None,
    lab_counts_by_pvt: dict[int, int] | None = None,
    *,
    low_swn_threshold: float = DEFAULT_LOW_SWN_THRESHOLD,
    j_cap_at_low_swn: float = DEFAULT_J_CAP_AT_LOW_SWN,
) -> dict[int, tuple[float, float, float]]:
    params: dict[int, tuple[float, float, float]] = {}
    fixed = fixed_params or {}
    lab_n = lab_counts_by_pvt or {}
    for pvt_raw, g in df.groupby("PVTNUM_GDM"):
        pvt = int(float(pvt_raw))
        n_lab = int(lab_n.get(pvt, 0))
        t0 = time.perf_counter()
        if pvt in fixed:
            params[pvt] = fixed[pvt]
        elif pvt not in bounds_by_pvt:
            if timing_rows is not None:
                timing_rows.append(
                    {
                        "PVTNUM_GDM": pvt,
                        "rows_geo": int(len(g)),
                        "rows_lab": n_lab,
                        "elapsed_sec": float(time.perf_counter() - t0),
                    }
                )
            continue
        else:
            bounds = [
                bounds_by_pvt[pvt]["a"],
                bounds_by_pvt[pvt]["b"],
                bounds_by_pvt[pvt]["sigma"],
            ]
            env = j_envelope_by_pvt.get(pvt) if j_envelope_by_pvt else None
            loss_fn = _make_loss_fn(
                g,
                env,
                low_swn_threshold=low_swn_threshold,
                j_cap_at_low_swn=j_cap_at_low_swn,
            )
            method = (optimizer_method or "differential_evolution").lower()
            if method == "pso":
                params[pvt] = _pso_optimize(
                    g,
                    bounds=bounds,
                    maxiter=maxiter,
                    particles=max(12, popsize),
                    j_envelope=env,
                    low_swn_threshold=low_swn_threshold,
                    j_cap_at_low_swn=j_cap_at_low_swn,
                )
            elif method == "dual_annealing":
                result = dual_annealing(loss_fn, bounds=bounds, maxiter=maxiter, seed=42)
                params[pvt] = (float(result.x[0]), float(result.x[1]), float(result.x[2]))
            else:
                result = differential_evolution(
                    loss_fn, bounds=bounds, maxiter=maxiter, popsize=popsize
                )
                params[pvt] = (float(result.x[0]), float(result.x[1]), float(result.x[2]))
        if timing_rows is not None:
            timing_rows.append(
                {
                    "PVTNUM_GDM": pvt,
                    "rows_geo": int(len(g)),
                    "rows_lab": n_lab,
                    "elapsed_sec": float(time.perf_counter() - t0),
                }
            )
    return params


def apply_model(df: pd.DataFrame, params: dict[int, tuple[float, float, float]]) -> pd.DataFrame:
    df = df.copy()
    kng_model = np.zeros(len(df))

    for i, row in df.iterrows():
        pvt = int(float(row["PVTNUM_GDM"]))
        if pvt not in params:
            continue
        a, b, sigma = params[pvt]
        val = calc_kng_vector(row.to_frame().T, a, b, sigma)[0]
        kng_model[i] = val

    df["Kng_model"] = kng_model
    return df


def _as_weight_array(weights: Any) -> np.ndarray:
    """Приводит веса к float-массиву (1.0 вместо NaN)."""
    w = pd.to_numeric(pd.Series(np.asarray(weights).ravel()), errors="coerce").fillna(1.0).to_numpy(dtype=float)
    return w


def _qa_metrics_row(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    weights: Any,
    pvt_label: int | str,
) -> dict[str, Any]:
    w = _as_weight_array(weights)
    m = np.isfinite(y_true) & np.isfinite(y_pred) & np.isfinite(w)
    if not np.any(m):
        return {
            "PVTNUM_GDM": pvt_label,
            "MAE": np.nan,
            "RMSE": np.nan,
            "BIAS": np.nan,
            "R2": np.nan,
            "SCORE": np.nan,
        }
    yt = y_true[m]
    yp = y_pred[m]
    ww = w[m]
    err = yp - yt
    sw = float(np.sum(ww))
    mae = float(np.average(np.abs(err), weights=ww)) if sw > 0 else np.nan
    rmse = float(np.sqrt(np.average(err**2, weights=ww))) if sw > 0 else np.nan
    bias = float(np.average(err, weights=ww)) if sw > 0 else np.nan
    score = float(1.0 - (float(np.sum(ww * np.abs(err))) / sw)) if sw > 0 else np.nan
    r2 = float(r2_score(yt, yp)) if len(yt) > 1 else float("nan")
    return {
        "PVTNUM_GDM": pvt_label,
        "MAE": mae,
        "RMSE": rmse,
        "BIAS": bias,
        "R2": r2,
        "SCORE": score,
    }


def compute_qa(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pvt_raw, g in df.groupby("PVTNUM_GDM"):
        rows.append(
            _qa_metrics_row(
                _safe_series(g, "Кнг_W"),
                _safe_series(g, "Kng_model"),
                g.get("weight", 1.0),
                int(float(pvt_raw)),
            )
        )
    regional = pd.DataFrame(rows)
    if not regional.empty:
        regional = regional.sort_values(
            by="PVTNUM_GDM",
            key=lambda s: pd.to_numeric(s, errors="coerce"),
        ).reset_index(drop=True)
    global_row = _qa_metrics_row(
        _safe_series(df, "Кнг_W"),
        _safe_series(df, "Kng_model"),
        df.get("weight", 1.0),
        "Все регионы",
    )
    qa = pd.concat([pd.DataFrame([global_row]), regional], ignore_index=True)
    return qa


def prepare_input_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    numeric_cols = [
        "PORO_GDM",
        "PERM_GDM",
        "PC",
        "SWL_GDM",
        "Кнг_W",
        "PVTNUM_GDM",
        "Perf_GDM",
        "ACTNUM_GDM",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Нормализованная пористость в долях (<1)
    if "PORO_GDM" in df.columns:
        poro = pd.to_numeric(df["PORO_GDM"], errors="coerce")
        df["PORO_FRAC"] = np.where(poro > 1, poro / 100.0, poro)

    if "Кнг_W" in df.columns:
        mask = df["Кнг_W"] > 1
        df.loc[mask, "Кнг_W"] = df.loc[mask, "Кнг_W"] / 100

    required_for_model = ["PORO_FRAC", "PERM_GDM", "PC", "SWL_GDM", "Кнг_W", "PVTNUM_GDM"]
    df = df.dropna(subset=required_for_model)

    if "ACTNUM_GDM" in df.columns:
        df = df[df["ACTNUM_GDM"] == 1]

    return df.reset_index(drop=True)


def _filter_training_kng(df: pd.DataFrame) -> pd.DataFrame:
    """
    Для обучения исключаем нулевые и выбросные Кнг_W по каждой скважине.
    Визуализация при этом выполняется на полном датафрейме.
    """
    out = df.copy()
    out["Кнг_W"] = pd.to_numeric(out["Кнг_W"], errors="coerce")
    mask = out["Кнг_W"].notna() & (out["Кнг_W"] != 0)
    out = out.loc[mask].copy()
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
        # Если фильтр слишком агрессивный, оставляем почти все кроме экстремумов по 2/98
        if len(keep) < max(5, int(0.5 * len(g))):
            ql, qh = np.quantile(x, [0.02, 0.98])
            keep = g[(g["Кнг_W"] >= ql) & (g["Кнг_W"] <= qh)]
        keep_idx.extend(keep.index.tolist())
    return out.loc[sorted(set(keep_idx))].copy()


def run_pipeline(
    df_wells: pd.DataFrame,
    df_prod: pd.DataFrame | None,
    bounds_by_pvt: dict[int, dict[str, tuple[float, float]]],
    maxiter: int = 200,
    popsize: int = 20,
    fixed_params: dict[int, tuple[float, float, float]] | None = None,
    optimizer_method: str = "differential_evolution",
    j_envelope_by_pvt: dict[int, dict[str, Any]] | None = None,
    lab_counts_by_pvt: dict[int, int] | None = None,
    *,
    low_swn_threshold: float = DEFAULT_LOW_SWN_THRESHOLD,
    j_cap_at_low_swn: float = DEFAULT_J_CAP_AT_LOW_SWN,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    missing = [c for c in REQUIRED_WELL_COLUMNS if c not in df_wells.columns]
    if missing:
        raise ValueError(f"В файле скважин отсутствуют колонки: {missing}")

    data = prepare_input_df(df_wells)
    data = prepare_weights(data, df_prod)
    train_data = _filter_training_kng(data)
    timing_rows: list[dict[str, Any]] = []
    params = optimize_pvt(
        train_data,
        bounds_by_pvt,
        maxiter=maxiter,
        popsize=popsize,
        fixed_params=fixed_params,
        optimizer_method=optimizer_method,
        j_envelope_by_pvt=j_envelope_by_pvt,
        timing_rows=timing_rows,
        lab_counts_by_pvt=lab_counts_by_pvt,
        low_swn_threshold=low_swn_threshold,
        j_cap_at_low_swn=j_cap_at_low_swn,
    )
    result = apply_model(data, params)
    qa = compute_qa(result)
    timing_df = _timing_table_with_total(timing_rows)

    params_df = pd.DataFrame(
        [
            {"PVTNUM_GDM": pvt, "a": vals[0], "b": vals[1], "sigma": vals[2]}
            for pvt, vals in sorted(params.items())
        ]
    )
    return result, params_df, qa, timing_df


def _timing_table_with_total(
    timing_rows: list[dict[str, Any]],
    total_elapsed: float | None = None,
) -> pd.DataFrame:
    if not timing_rows:
        return pd.DataFrame(columns=["PVTNUM_GDM", "rows_geo", "rows_lab", "elapsed_sec"])
    regional = pd.DataFrame(timing_rows).sort_values(
        by="PVTNUM_GDM",
        key=lambda s: pd.to_numeric(s, errors="coerce"),
    ).reset_index(drop=True)
    elapsed = float(total_elapsed) if total_elapsed is not None else float(regional["elapsed_sec"].sum())
    total_row = {
        "PVTNUM_GDM": "Все регионы",
        "rows_geo": int(regional["rows_geo"].sum()),
        "rows_lab": int(regional["rows_lab"].sum()),
        "elapsed_sec": elapsed,
    }
    return pd.concat([pd.DataFrame([total_row]), regional], ignore_index=True)