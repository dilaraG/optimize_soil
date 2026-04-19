import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
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
    poro = _safe_series(df, "PORO_GDM")
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


def loss_function(params: tuple[float, float, float], df: pd.DataFrame) -> float:
    a, b, sigma = params
    kng_model = calc_kng_vector(df, a, b, sigma)
    kng_true = _safe_series(df, "Кнг_W")
    weights = pd.to_numeric(df.get("weight", 1.0), errors="coerce").fillna(1.0).to_numpy()
    valid = ~np.isnan(kng_model) & np.isfinite(kng_true)
    if not np.any(valid):
        return 1e9
    r = kng_model[valid] - kng_true[valid]
    return float(np.sum(weights[valid] * huber_loss(r)))


def optimize_pvt(
    df: pd.DataFrame,
    bounds_by_pvt: dict[int, dict[str, tuple[float, float]]],
    maxiter: int = 200,
    popsize: int = 20,
    fixed_params: dict[int, tuple[float, float, float]] | None = None,
) -> dict[int, tuple[float, float, float]]:
    params: dict[int, tuple[float, float, float]] = {}
    fixed = fixed_params or {}
    for pvt_raw, g in df.groupby("PVTNUM_GDM"):
        pvt = int(float(pvt_raw))
        if pvt in fixed:
            params[pvt] = fixed[pvt]
            continue
        if pvt not in bounds_by_pvt:
            continue
        bounds = [
            bounds_by_pvt[pvt]["a"],
            bounds_by_pvt[pvt]["b"],
            bounds_by_pvt[pvt]["sigma"],
        ]
        result = differential_evolution(loss_function, bounds=bounds, args=(g,), maxiter=maxiter, popsize=popsize)
        params[pvt] = (float(result.x[0]), float(result.x[1]), float(result.x[2]))
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


def compute_qa(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pvt_raw, g in df.groupby("PVTNUM_GDM"):
        y_true = _safe_series(g, "Кнг_W")
        y_pred = _safe_series(g, "Kng_model")
        w = pd.to_numeric(g.get("weight", 1.0), errors="coerce").fillna(1.0).to_numpy()
        err = y_pred - y_true
        mae = float(np.average(np.abs(err), weights=w))
        rmse = float(np.sqrt(np.average(err**2, weights=w)))
        bias = float(np.average(err, weights=w))
        r2 = float(r2_score(y_true, y_pred))
        score = float(1 - (np.sum(w * np.abs(err)) / np.sum(w)))
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

    if "Кнг_W" in df.columns:
        mask = df["Кнг_W"] > 1
        df.loc[mask, "Кнг_W"] = df.loc[mask, "Кнг_W"] / 100

    required_for_model = ["PORO_GDM", "PERM_GDM", "PC", "SWL_GDM", "Кнг_W", "PVTNUM_GDM"]
    df = df.dropna(subset=required_for_model)

    if "ACTNUM_GDM" in df.columns:
        df = df[df["ACTNUM_GDM"] == 1]

    return df.reset_index(drop=True)


def run_pipeline(
    df_wells: pd.DataFrame,
    df_prod: pd.DataFrame | None,
    bounds_by_pvt: dict[int, dict[str, tuple[float, float]]],
    maxiter: int = 200,
    popsize: int = 20,
    fixed_params: dict[int, tuple[float, float, float]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    missing = [c for c in REQUIRED_WELL_COLUMNS if c not in df_wells.columns]
    if missing:
        raise ValueError(f"В файле скважин отсутствуют колонки: {missing}")

    data = prepare_input_df(df_wells)
    data = prepare_weights(data, df_prod)
    params = optimize_pvt(
        data,
        bounds_by_pvt,
        maxiter=maxiter,
        popsize=popsize,
        fixed_params=fixed_params,
    )
    result = apply_model(data, params)
    qa = compute_qa(result)

    params_df = pd.DataFrame(
        [
            {"PVTNUM_GDM": pvt, "a": vals[0], "b": vals[1], "sigma": vals[2]}
            for pvt, vals in sorted(params.items())
        ]
    )
    return result, params_df, qa