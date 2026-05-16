"""
Анализ лабораторных точек J vs Swn: подбор степенной модели и автограниц a, b.
"""

from __future__ import annotations

import re
from typing import Any, Sequence

import numpy as np
import pandas as pd


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())


def guess_area_column(columns: list[str]) -> str | None:
    """
    Колонка площади месторождения. Не путать с «Площадность»/пористостью (там есть подстрока «площад»).
    """
    exclude = (
        "площадност",
        "porosit",
        "порист",
        "насыщ",
        "прониц",
        "проницаем",
    )

    def _ok_name(c: str) -> bool:
        t = _norm(c)
        if any(ex in t for ex in exclude):
            return False
        if "площадност" in t:
            return False
        return True

    # Точное «Площадь» в приоритете
    for c in columns:
        if not _ok_name(c):
            continue
        if _norm(c) == "площадь":
            return c

    keys = ("площад", "площ", "area", "местор", "куст", "field")
    for c in columns:
        if not _ok_name(c):
            continue
        t = _norm(c)
        if any(k in t for k in keys):
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
    """J ≈ a * Swn^b в лог-лог: OLS по тем же точкам, что и для огибающих (без сильных выбросов)."""
    mask = np.isfinite(swn) & np.isfinite(j) & (swn > 0) & (j > 0)
    if mask.sum() < 3:
        return {"a": np.nan, "b": np.nan, "r2": np.nan, "n": int(mask.sum())}
    s = swn[mask].astype(float)
    jj = j[mask].astype(float)
    env_m = _j_cloud_inlier_mask_for_envelopes(s, jj)
    s_fit = s[env_m]
    jj_fit = jj[env_m]
    if len(s_fit) < 3:
        s_fit, jj_fit = s, jj
    u = np.log(s_fit)
    v = np.log(jj_fit)
    coef = np.polyfit(u, v, 1)
    b = float(coef[0])
    ln_a = float(coef[1])
    a = float(np.exp(ln_a))
    v_pred = ln_a + b * u
    ss_res = float(np.sum((v - v_pred) ** 2))
    ss_tot = float(np.sum((v - np.mean(v)) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan
    return {"a": a, "b": b, "r2": r2, "n": int(len(u))}


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


def _refine_lower_j_power_for_low_swn(
    s: np.ndarray,
    j: np.ndarray,
    b_lo: float,
    b_up: float,
    low_swn_quantile: float = 0.18,
    n_grid: int = 44,
) -> tuple[float, float]:
    """
    Нижняя степенная J = a·Swn^b не должна сильно провисать у малых Swn относительно точек:
    перебираем b_lo в [b_lo, min(b_up−ε, −0.055)] и для каждого b берём a = min_i(J_i/Swn_i^b);
    выбираем пару, у которой минимальный зазор J_i − a·Swn_i^b по точкам с Swn в нижнем квантиле
    минимален (стремимся к касанию хотя бы одной точки в этой зоне), не нарушая a·Swn^b ≤ J везде.
    """
    b_lo = float(b_lo)
    b_up = float(b_up)
    b_cap = float(min(-0.055, b_up - 0.02))
    if not (np.isfinite(b_lo) and np.isfinite(b_cap)) or b_lo >= b_cap:
        a0 = _tight_a_from_b(s, j, b_lo, "lower")
        return b_lo, a0

    q = float(np.clip(low_swn_quantile, 0.05, 0.45))
    s_thr = float(np.quantile(s, q))
    m_low = s <= s_thr
    if not np.any(m_low):
        a0 = _tight_a_from_b(s, j, b_lo, "lower")
        return b_lo, a0

    bs = np.linspace(b_lo, b_cap, int(max(8, min(n_grid, 80))))
    best_b = b_lo
    best_gap = float("inf")
    for b in bs:
        bb = float(b)
        a_try = _tight_a_from_b(s, j, bb, "lower")
        if not np.isfinite(a_try):
            continue
        with np.errstate(over="ignore", invalid="ignore"):
            pred = a_try * (s**bb)
        gap = j - pred
        if np.min(gap) < -1e-7:
            continue
        g_low = float(np.min(gap[m_low]))
        if g_low < best_gap:
            best_gap = g_low
            best_b = bb

    a_best = _tight_a_from_b(s, j, best_b, "lower")
    if not np.isfinite(a_best):
        return b_lo, _tight_a_from_b(s, j, b_lo, "lower")
    return float(best_b), float(a_best)


def _tight_a_from_b_robust_upper(s: np.ndarray, j: np.ndarray, b: float) -> float:
    """
    Робастная оценка a для верхней огибающей.
    Отсекаем экстремально большие r = J / Swn^b, чтобы единичные выбросы
    не завышали верхнюю границу.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        r = j / (s**b)
    r = r[np.isfinite(r) & (r > 0)]
    if r.size == 0:
        return float("nan")
    if r.size < 8:
        return float(np.max(r))

    q1, q3 = np.quantile(r, [0.25, 0.75])
    iqr = q3 - q1
    hi = q3 + 1.5 * iqr
    core = r[r <= hi]
    if core.size < max(5, int(0.3 * r.size)):
        core = r

    # Верхняя граница как высокий квантиль "ядра", а не абсолютный max.
    # Это сохраняет верхнюю огибающую близкой к облаку без улета по выбросам.
    return float(np.quantile(core, 0.98))


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


def _j_cloud_inlier_mask_for_envelopes(
    s: np.ndarray,
    j: np.ndarray,
    *,
    iqr_k: float = 2.8,
    min_keep_frac: float = 0.55,
) -> np.ndarray:
    """
    Сильные выбросы в облаке (Swn, J) не участвуют в построении нижней/верхней огибающих.

    1) По IQR в log(Swn) и log(J) — грубый отсев по осям.
    2) По робастным остаткам в log-log: наклон — медиана попарных наклонов, сдвиг — медиана(v − b·u);
       точки с |z| > порога (жёстче при малом n) отбрасываются — сильные выбросы относительно
       основного тренда в (ln Swn, ln J), в том числе при малой выборке по горизонту.
    """
    n = int(len(s))
    if n < 4:
        return np.ones(n, dtype=bool)
    s = np.asarray(s, dtype=float)
    j = np.asarray(j, dtype=float)
    u = np.log(s)
    v = np.log(j)
    base = np.isfinite(u) & np.isfinite(v) & (s > 0) & (j > 0)
    if int(base.sum()) < 4:
        return np.ones(n, dtype=bool)

    min_keep = max(5, int((0.35 if n <= 14 else min_keep_frac) * n))

    # --- IQR по осям log ---
    m_iqr = np.ones(n, dtype=bool)
    if n >= 10:
        uu, vv = u[base], v[base]
        q1, q3 = np.quantile(vv, [0.25, 0.75])
        iqr = float(q3 - q1)
        q1s, q3s = np.quantile(uu, [0.25, 0.75])
        iqrs = float(q3s - q1s)
        if iqr >= 1e-12 and iqrs >= 1e-12:
            lo_v, hi_v = q1 - iqr_k * iqr, q3 + iqr_k * iqr
            lo_u, hi_u = q1s - iqr_k * iqrs, q3s + iqr_k * iqrs
            m_iqr = np.zeros(n, dtype=bool)
            m_iqr[base] = (vv >= lo_v) & (vv <= hi_v) & (uu >= lo_u) & (uu <= hi_u)
            if int(m_iqr.sum()) < min_keep:
                m_iqr = np.ones(n, dtype=bool)

    # --- Остатки от робастной прямой в (ln Swn, ln J) ---
    def _residual_z_thresh(n_pts: int) -> float:
        if n_pts <= 10:
            return 1.9
        if n_pts <= 14:
            return 2.05
        if n_pts <= 20:
            return 2.25
        if n_pts <= 28:
            return 2.55
        return 3.15

    m_res = np.ones(n, dtype=bool)
    ub, vb = u[base], v[base]
    if len(ub) >= 4:
        b_med = float(_median_pairwise_slope(ub, vb))
        if not np.isfinite(b_med):
            b_med = -0.35
        ln_a = float(np.median(vb - b_med * ub))
        pred = ln_a + b_med * u
        res = v - pred
        med_r = float(np.median(res[base]))
        mad = float(np.median(np.abs(res[base] - med_r))) + 1e-12
        if mad < 1e-8:
            mad = float(np.std(res[base])) + 1e-12
        z = np.abs(res - med_r) / (1.4826 * mad + 1e-15)
        th = _residual_z_thresh(int(base.sum()))
        m_res = np.zeros(n, dtype=bool)
        m_res[base] = z[base] <= th
        if int(m_res.sum()) < min_keep:
            m_res = np.ones(n, dtype=bool)

    combined = m_iqr & m_res
    if int(combined.sum()) < min_keep:
        combined = m_res if int(m_res.sum()) >= min_keep else m_iqr
    if int(combined.sum()) < min_keep:
        return np.ones(n, dtype=bool)
    return combined


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

    Нижняя: b_lo < 0, затем пара (a_lo, b_lo) уточняется так, чтобы у малых Swn (нижний квантиль)
    кривая не «провисала» сильно ниже точек — стремление к касанию хотя бы одной точки в этой зоне.

    Сильные выбросы по **log(Swn)** и **log(J)** (IQR) и по **остаткам** от робастной прямой в log-log
    при расчёте **центральной степенной (OLS в log-log), нижней и верхней** огибающих не используются;
    при малом n порог по z жёстче.

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

    env_m = _j_cloud_inlier_mask_for_envelopes(s, jj)
    s_e = s[env_m]
    jj_e = jj[env_m]
    if len(s_e) < 5:
        s_e, jj_e = s, jj
    u_e = np.log(s_e)
    v_e = np.log(jj_e)

    # Центр: OLS в log-log по тем же inlier, что и огибающие; при b >= 0 — робастный наклон (b < 0)
    b_ols, ln_a_ols = np.polyfit(u_e, v_e, 1)
    b_ols = float(b_ols)
    ln_a_ols = float(ln_a_ols)
    if b_ols >= -1e-4:
        b_ols = float(np.clip(_median_pairwise_slope(u_e, v_e), -8.0, -0.06))
        if b_ols >= -1e-4:
            b_ols = -0.35
        ln_a_ols = float(np.mean(v_e - b_ols * u_e))
    a_mid = float(np.exp(ln_a_ols))

    # Наклоны огибающих по подмножеству без сильных выбросов (оба b < 0)
    b_up_raw = _binned_extreme_slope(u_e, v_e, mode="upper")
    b_lo_raw = _binned_extreme_slope(u_e, v_e, mode="lower")
    b_up = float(np.clip(max(b_up_raw, b_ols), -8.0, -0.055))
    b_lo = float(np.clip(min(b_lo_raw, b_ols), -12.0, -0.06))
    if b_lo >= b_up:
        b_lo = float(min(b_lo_raw, b_ols) - 0.12)
        b_lo = float(np.clip(b_lo, -12.0, b_up - 0.02))

    a_up = _tight_a_from_b_robust_upper(s_e, jj_e, b_up)
    b_lo, a_lo = _refine_lower_j_power_for_low_swn(s_e, jj_e, b_lo, b_up)
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


def _norm_header_token(x: object) -> str:
    return _norm(str(x))


# Подстроки для **первой** строки объединённого заголовка (уровень 0 MultiIndex).
DEFAULT_MATRIX_WATER_KEYWORDS: tuple[str, ...] = (
    "водонасыщ",
    "насыщенность вод",
    "насыщенность, д",
    "sw,",
    "кво,",
    "кво ",
)
DEFAULT_MATRIX_J_KEYWORDS: tuple[str, ...] = (
    "j функц",
    "леверет",
    "leverett",
    "функция j",
    "j-функ",
)


def parse_matrix_block_keywords(user_text: str | None, fallback: tuple[str, ...]) -> list[str]:
    """
    Разбор подстрок из поля ввода: запятая, точка с запятой, перевод строки.
    Пустой ввод → fallback.
    """
    raw = (user_text or "").strip()
    if not raw:
        return list(fallback)
    parts = re.split(r"[,;\n\r]+", raw)
    out: list[str] = []
    for p in parts:
        t = _norm_header_token(p)
        if t:
            out.append(t)
    if not out:
        return list(fallback)
    seen: set[str] = set()
    uniq: list[str] = []
    for x in out:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq


def _is_matrix_meta_second_level(l1: object) -> bool:
    """
    Вторая строка заголовка: площадь месторождения, скважина, фация, № образца, код горизонта —
    не ступени Sw/J (чтобы «Площадь» не перепутать с блоком воды; «Код горизонта» под блоком J
    остаётся метаданными, а не ступенью).
    """
    s = _norm(str(l1))
    sc = s.replace(" ", "").replace(".", "")
    if "горизонт" in sc or "stratum" in s or "пласткод" in sc:
        return True
    if "площадь" in sc and "водонасыщ" not in s and "площадност" not in s:
        return True
    if "скв" in sc and "№" in str(l1):
        return True
    if sc == "фация" or s.startswith("фация"):
        return True
    if "обр" in sc and "№" in str(l1):
        return True
    if sc.startswith("куст"):
        return True
    return False


def _looks_like_stage_subcolumn(l1: object) -> bool:
    """Подпись ступени: «1 ст.», «2 ст.» или «1 st» и т.п."""
    s = _norm(str(l1))
    if re.search(r"\d+\s*ст", s):
        return True
    if re.search(r"\d+\s*st\b", s):
        return True
    return False


def _level0_matches_any(level0_norm: str, keywords: Sequence[str]) -> bool:
    for k in keywords:
        kk = _norm_header_token(k)
        if kk and kk in level0_norm:
            return True
    return False


def classify_j_matrix_stairs_columns(
    df: pd.DataFrame,
    water_keywords: Sequence[str] | None = None,
    j_keywords: Sequence[str] | None = None,
) -> tuple[list[tuple[Any, ...]], list[tuple[Any, ...]], list[tuple[Any, ...]]]:
    """
    Делит столбцы MultiIndex на: водонасыщенность по ступеням, J по ступеням, метаданные.

    В блок Sw/J попадают только колонки, у которых вторая строка похожа на ступень («N ст.»).
    «Площадь», «№ скв.», «Фация», «№ обр.» всегда остаются в метаданных.
    """
    if not isinstance(df.columns, pd.MultiIndex):
        raise ValueError("Ожидается MultiIndex столбцов (header=[0, 1]).")

    wk = list(water_keywords) if water_keywords is not None else list(DEFAULT_MATRIX_WATER_KEYWORDS)
    jk = list(j_keywords) if j_keywords is not None else list(DEFAULT_MATRIX_J_KEYWORDS)

    sw_cols: list[tuple[Any, ...]] = []
    j_cols: list[tuple[Any, ...]] = []
    meta_cols: list[tuple[Any, ...]] = []

    for c in df.columns:
        if not isinstance(c, tuple) or len(c) != 2:
            meta_cols.append(c)  # type: ignore[arg-type]
            continue
        l0, l1 = c[0], c[1]
        if _is_matrix_meta_second_level(l1):
            meta_cols.append(c)
            continue
        t0 = _norm_header_token(l0)
        if _level0_matches_any(t0, wk) and _looks_like_stage_subcolumn(l1):
            sw_cols.append(c)
        elif _level0_matches_any(t0, jk) and _looks_like_stage_subcolumn(l1):
            j_cols.append(c)
        else:
            meta_cols.append(c)

    return sw_cols, j_cols, meta_cols


def is_likely_j_matrix_stairs_format(
    df: pd.DataFrame,
    water_keywords: Sequence[str] | None = None,
    j_keywords: Sequence[str] | None = None,
) -> bool:
    if not isinstance(df.columns, pd.MultiIndex):
        return False
    try:
        sw, j, _meta = classify_j_matrix_stairs_columns(df, water_keywords, j_keywords)
    except ValueError:
        return False
    return bool(sw and j)


def _stage_key_from_subcol(sub: object) -> tuple[int, str]:
    """Для сортировки пар ступеней: ('7 ст.',) -> (7, '7 ст.')."""
    s = str(sub).strip()
    m = re.search(r"(\d+)", s)
    n = int(m.group(1)) if m else 0
    return n, s


def transform_j_stairs_wide_to_long(
    df: pd.DataFrame,
    water_keywords: Sequence[str] | None = None,
    j_keywords: Sequence[str] | None = None,
) -> pd.DataFrame:
    """
    Разворачивает «матрицу ступеней» в длинный вид, как для облака J(Swn).

    ``water_keywords`` / ``j_keywords`` — подстроки для **первой** строки объединённого заголовка:
    колонка относится к блоку, если любая подстрока входит в нормализованный текст уровня 0
    и вторая строка похожа на ступень. Метастолбцы («Площадь», …) не попадают в блоки.
    """
    if not isinstance(df.columns, pd.MultiIndex):
        raise ValueError("Ожидается двухуровневый заголовок (header=[0, 1] при чтении Excel).")

    sw_cols, j_cols, meta_cols = classify_j_matrix_stairs_columns(df, water_keywords, j_keywords)

    if not sw_cols or not j_cols:
        raise ValueError(
            "Не найдены столбцы ступеней для водонасыщенности и/или J. "
            "Проверьте подписи блоков (первая строка) или расширьте списки ключевых слов."
        )

    # Пары ступеней по подписи второго уровня (1 ст., 2 ст., …)
    sw_by_sub: dict[str, tuple[Any, ...]] = {}
    for c in sw_cols:
        sub = str(c[1]).strip()
        sw_by_sub[sub] = c
    pairs: list[tuple[str, tuple[Any, ...], tuple[Any, ...]]] = []
    for jc in j_cols:
        sub = str(jc[1]).strip()
        if sub in sw_by_sub:
            pairs.append((sub, sw_by_sub[sub], jc))
    pairs.sort(key=lambda p: _stage_key_from_subcol(p[0]))
    if not pairs:
        raise ValueError("Не удалось сопоставить ступени между водонасыщенностью и J.")

    meta_names = [str(c[1]).strip() if isinstance(c, tuple) and len(c) == 2 else str(c) for c in meta_cols]

    rows_out: list[dict[str, Any]] = []
    eps = 1e-12

    for _, row in df.iterrows():
        sw_vals: list[float] = []
        j_vals: list[float] = []
        for _st, sw_c, j_c in pairs:
            sw_vals.append(float(pd.to_numeric(row[sw_c], errors="coerce")))
            j_vals.append(float(pd.to_numeric(row[j_c], errors="coerce")))

        finite_sw = [v for v in sw_vals if np.isfinite(v)]
        if not finite_sw:
            continue
        sw_min = float(min(finite_sw))
        sw_max = float(max(finite_sw))
        den = sw_max - sw_min

        meta_vals: dict[str, Any] = {}
        for c, name in zip(meta_cols, meta_names, strict=False):
            meta_vals[name] = row[c]

        for (_st, sw_c, j_c), sw, jv in zip(pairs, sw_vals, j_vals):
            if not np.isfinite(sw) or not np.isfinite(jv):
                continue
            if den <= eps or not np.isfinite(den):
                swn = float("nan")
            else:
                swn = (sw - sw_min) / den
                if swn <= eps or swn >= 1.0 - eps or abs(swn) < eps or abs(swn - 1.0) < eps:
                    swn = float("nan")
            rec = {
                **meta_vals,
                "ступень": _st,
                "Водонасыщенность": sw,
                "Sw_min": sw_min,
                "Sw_max": sw_max,
                "Swn": swn,
                "J (функция Леверетта)": jv,
            }
            rows_out.append(rec)

    out = pd.DataFrame(rows_out)
    if out.empty:
        return out

    # Код горизонта: если в файле уже есть столбец (часто под объединённым «J функция»), сохраняем его.
    if "Код горизонта" in out.columns:
        out["Код горизонта"] = out["Код горизонта"].astype(str).str.strip()
    elif "Фация" in out.columns:
        out["Код горизонта"] = out["Фация"].astype(str).str.strip()
    return out
