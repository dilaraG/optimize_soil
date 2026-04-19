"""
Анализ лабораторных точек J vs Swn: подбор степенной модели и автограниц a, b.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())


def guess_area_column(columns: list[str]) -> str | None:
    keys = ("площад", "площ", "area", "местор", "куст", "field")
    for c in columns:
        if any(k in _norm(c) for k in keys):
            return c
    return None


def guess_horizon_column(columns: list[str]) -> str | None:
    keys = ("горизонт", "код", "stratum", "zone", "пласт")
    for c in columns:
        t = _norm(c)
        if "горизонт" in t or "код горизонта" in t or ("код" in t and "площ" not in t):
            return c
    for c in columns:
        if any(k in _norm(c) for k in keys):
            return c
    return None


def guess_j_column(columns: list[str]) -> str | None:
    keys = ("леверет", "leverett", "j_func", " j", "функция j")
    for c in columns:
        t = _norm(c)
        if "леверет" in t or "leverett" in t:
            return c
    for c in columns:
        if "функция" in _norm(c) and "j" in _norm(c):
            return c
    return None


def guess_swn_columns(columns: list[str]) -> list[str]:
    swn_cols = [c for c in columns if re.search(r"swn|sw_n|насыщ", _norm(c), re.I)]
    if len(swn_cols) >= 2:
        return swn_cols
    # второй по смыслу: колонки со словом вода / нефть / газ в названии — иначе все float-кандидаты
    alt = [c for c in columns if re.search(r"sw|насыщ|водн", _norm(c), re.I)]
    out = []
    for c in alt:
        if c not in out:
            out.append(c)
    return out


def pick_second_swn_column(columns: list[str]) -> tuple[str | None, list[str]]:
    swn_list = guess_swn_columns(columns)
    if len(swn_list) >= 2:
        return swn_list[1], swn_list
    if len(swn_list) == 1:
        return swn_list[0], swn_list
    return None, []


def fit_power_j_swn(swn: np.ndarray, j: np.ndarray) -> dict[str, Any]:
    """J ≈ a * Swn^b в лог-лог масштабе."""
    mask = np.isfinite(swn) & np.isfinite(j) & (swn > 0) & (j > 0)
    if mask.sum() < 3:
        return {"a": np.nan, "b": np.nan, "r2": np.nan, "n": int(mask.sum())}
    u = np.log(swn[mask])
    v = np.log(j[mask])
    coef = np.polyfit(u, v, 1)
    b = float(coef[0])
    ln_a = float(coef[1])
    a = float(np.exp(ln_a))
    v_pred = ln_a + b * u
    ss_res = float(np.sum((v - v_pred) ** 2))
    ss_tot = float(np.sum((v - np.mean(v)) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan
    return {"a": a, "b": b, "r2": r2, "n": int(mask.sum())}


def _median_pairwise_slope(u: np.ndarray, v: np.ndarray) -> float:
    """Наклон dv/du по соседним точкам на сортировке по u (устойчиво к выбросам)."""
    if len(u) < 2:
        return -0.35
    order = np.argsort(u)
    us, vs = u[order], v[order]
    du = np.diff(us)
    dv = np.diff(vs)
    m = np.isfinite(du) & (np.abs(du) > 1e-12)
    if not np.any(m):
        return -0.35
    slopes = dv[m] / du[m]
    slopes = slopes[np.isfinite(slopes)]
    if slopes.size == 0:
        return -0.35
    return float(np.median(slopes))


def _binned_extreme_slope(u: np.ndarray, v: np.ndarray, mode: str, n_bins: int = 14) -> float:
    """
    Наклон в log-log по «верхней» или «нижней» границе облака (макс./мин. v в корзинах по u).
    mode: 'upper' | 'lower'
    """
    if len(u) < 4:
        return -0.35
    qs = np.unique(np.quantile(u, np.linspace(0, 1, max(5, min(n_bins, len(u) // 2)) + 1)))
    if qs.size < 3:
        return -0.35
    cx, cy = [], []
    for k in range(len(qs) - 1):
        m = (u >= qs[k]) & (u <= qs[k + 1])
        if m.sum() < 1:
            continue
        cx.append(float(np.mean(u[m])))
        if mode == "upper":
            cy.append(float(np.max(v[m])))
        else:
            cy.append(float(np.min(v[m])))
    cx_a = np.asarray(cx, dtype=float)
    cy_a = np.asarray(cy, dtype=float)
    if cx_a.size < 3:
        return -0.35
    b_hat, intercept = np.polyfit(cx_a, cy_a, 1)
    b_hat = float(b_hat)
    if not np.isfinite(b_hat) or b_hat >= -1e-4:
        b_hat = -0.25
    b_hat = float(np.clip(b_hat, -12.0, -0.06))
    return b_hat


def _tight_a_from_b(s: np.ndarray, j: np.ndarray, b: float, side: str) -> float:
    """side='upper': a = max(j/s^b); side='lower': a = min(j/s^b). Требуется s>0, j>0, b<0."""
    with np.errstate(divide="ignore", invalid="ignore"):
        r = j / (s**b)
    r = r[np.isfinite(r) & (r > 0)]
    if r.size == 0:
        return float("nan")
    if side == "upper":
        return float(np.max(r))
    return float(np.min(r))


def _clip_power_curve(s: np.ndarray, j_min: float, j_max: float, a: float, b: float) -> tuple[np.ndarray, np.ndarray]:
    """Сетка только по диапазону s; J обрезается в пределах точек."""
    s0, s1 = float(np.min(s)), float(np.max(s))
    if s1 <= s0:
        s1 = s0 + 1e-9
    grid = np.linspace(s0, s1, 200)
    with np.errstate(over="ignore", invalid="ignore"):
        y = a * (grid**b)
    y = np.clip(np.real(y), j_min, j_max)
    return grid, y


def auto_ab_bounds_from_cloud(
    swn: np.ndarray,
    j: np.ndarray,
    pad_a: float = 0.06,
    pad_b: float = 0.12,
) -> dict[str, Any]:
    """
    Огибающие — степенные J = a·Swn^b с b < 0 (больше J ↔ меньше Swn).

    Верхняя: выбирается b_up < 0 по «верхней» границе облака в (ln Swn, ln J), затем
    a_up = max_i(J_i / Swn_i^b_up) — тогда a_up·Swn_i^b_up >= J_i для всех i.

    Нижняя: b_lo < 0 по «нижней» границе, a_lo = min_i(J_i / Swn_i^b_lo) — кривая не выше точек в узлах.

    Для отрисовки: кривые строятся только на [Swn_min, Swn_max], значения J обрезаются в [J_min, J_max],
    чтобы линии не уходили за пределы данных по вертикали.
    """
    mask = np.isfinite(swn) & np.isfinite(j) & (swn > 0) & (j > 0)
    if mask.sum() < 5:
        return {
            "a_bounds": (0.05, 0.35),
            "b_bounds": (-3.0, -0.4),
            "center": {"a": np.nan, "b": np.nan},
            "lower": {"a": np.nan, "b": np.nan},
            "upper": {"a": np.nan, "b": np.nan},
            "plot": None,
            "n": int(mask.sum()),
        }
    s = swn[mask].astype(float)
    jj = j[mask].astype(float)
    j_min, j_max = float(np.min(jj)), float(np.max(jj))
    u = np.log(s)
    v = np.log(jj)
    n = len(u)

    # Центр: OLS в log-log; при b >= 0 принудительно убывающий J(Swn) (b < 0) и пересчёт ln a
    b_ols, ln_a_ols = np.polyfit(u, v, 1)
    b_ols = float(b_ols)
    ln_a_ols = float(ln_a_ols)
    if b_ols >= -1e-4:
        b_ols = float(np.clip(_median_pairwise_slope(u, v), -8.0, -0.06))
        if b_ols >= -1e-4:
            b_ols = -0.35
        ln_a_ols = float(np.mean(v - b_ols * u))
    a_mid = float(np.exp(ln_a_ols))

    # Наклоны огибающих (оба < 0): верхняя — менее крутая max(...), нижняя — более крутая min(...)
    b_up_raw = _binned_extreme_slope(u, v, mode="upper")
    b_lo_raw = _binned_extreme_slope(u, v, mode="lower")
    b_up = float(np.clip(max(b_up_raw, b_ols), -8.0, -0.055))
    b_lo = float(np.clip(min(b_lo_raw, b_ols), -12.0, -0.06))
    if b_lo >= b_up:
        b_lo = float(min(b_lo_raw, b_ols) - 0.12)
        b_lo = float(np.clip(b_lo, -12.0, b_up - 0.02))

    a_up = _tight_a_from_b(s, jj, b_up, "upper")
    a_lo = _tight_a_from_b(s, jj, b_lo, "lower")
    if not (np.isfinite(a_up) and np.isfinite(a_lo)):
        return {
            "a_bounds": (0.05, 0.35),
            "b_bounds": (-3.0, -0.4),
            "center": {"a": a_mid, "b": b_ols},
            "lower": {"a": np.nan, "b": np.nan},
            "upper": {"a": np.nan, "b": np.nan},
            "plot": None,
            "n": int(n),
        }

    # Границы для оптимизации (прямоугольник вокруг двух степенных моделей лаборатории)
    amin = float(max(1e-8, min(a_lo, a_up) * (1 - pad_a)))
    amax = float(max(a_lo, a_up) * (1 + pad_a))
    if amax <= amin:
        amax = amin * 1.05
    bmin = float(min(b_lo, b_up) - pad_b)
    bmax = float(max(b_lo, b_up) + pad_b)
    if bmin >= bmax:
        bmin, bmax = b_ols - 0.4, b_ols + 0.15
    bmax = float(min(bmax, -0.05))
    bmin = float(max(bmin, -12.0))

    gx_u, gy_u = _clip_power_curve(s, j_min, j_max, a_up, b_up)
    gx_l, gy_l = _clip_power_curve(s, j_min, j_max, a_lo, b_lo)

    return {
        "a_bounds": (amin, amax),
        "b_bounds": (bmin, bmax),
        "center": {"a": float(a_mid), "b": float(b_ols)},
        "lower": {"a": float(a_lo), "b": float(b_lo)},
        "upper": {"a": float(a_up), "b": float(b_up)},
        "plot": {
            "upper": {"x": gx_u, "y": gy_u, "a": float(a_up), "b": float(b_up)},
            "lower": {"x": gx_l, "y": gy_l, "a": float(a_lo), "b": float(b_lo)},
        },
        "n": int(n),
    }


def filter_lab_df(
    df: pd.DataFrame,
    area_col: str,
    horizon_col: str,
    areas: list[Any],
    horizons: list[Any],
) -> pd.DataFrame:
    out = df.copy()
    out[area_col] = out[area_col].astype(str).str.strip()
    out[horizon_col] = out[horizon_col].astype(str).str.strip()
    return out[out[area_col].isin(areas) & out[horizon_col].isin(horizons)]
