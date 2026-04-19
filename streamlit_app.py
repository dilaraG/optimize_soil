from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from kkd_database import build_kkd_sqlite, excel_path, load_kkd_dataframe, sqlite_path
from lab_analysis import (
    auto_ab_bounds_from_cloud,
    filter_lab_df,
    fit_power_j_swn,
    guess_area_column,
    guess_horizon_column,
    guess_j_column,
    pick_second_swn_column,
)
from optimize import run_pipeline

st.set_page_config(page_title="J-функция Леверетта", layout="wide")

DATA_DIR = Path(__file__).resolve().parent / "data"


def _read_table(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file, low_memory=False)
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(uploaded_file)
    if name.endswith(".txt"):
        content = uploaded_file.getvalue().decode("utf-8", errors="ignore")
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
        df = df.loc[df["Кнг_W"] != 0]
        df.loc[df["Кнг_W"] > 1, "Кнг_W"] = df["Кнг_W"] / 100
    return df.reset_index(drop=True)


def _clean_prod_df(df: pd.DataFrame | None) -> pd.DataFrame | None:
    if df is None:
        return None
    df = _normalize_columns(df.copy())
    df = df.rename(columns=lambda x: str(x).replace("\n", ""))
    return df


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
    st.caption("База данных строится из файла `data/ККД_БД.xlsx` (SQLite: `data/kkd_lab.sqlite`).")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
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
        df = load_kkd_dataframe()
    except FileNotFoundError:
        st.info("База ещё не создана. Положите `ККД_БД.xlsx` в папку `data` или загрузите файл выше, затем нажмите «Пересобрать SQLite из Excel».")
        return

    cols = list(df.columns)

    def _idx_or_zero(name: str | None) -> int:
        return cols.index(name) if name and name in cols else 0

    area_guess = guess_area_column(cols)
    hor_guess = guess_horizon_column(cols)
    j_guess = guess_j_column(cols)
    swn_pick, swn_list = pick_second_swn_column(cols)

    st.subheader("Сопоставление колонок")
    c1, c2, c3, c4 = st.columns(4)
    area_col = c1.selectbox("Колонка площади", options=cols, index=_idx_or_zero(area_guess))
    hor_col = c2.selectbox("Колонка кода горизонта", options=cols, index=_idx_or_zero(hor_guess))
    j_col = c3.selectbox("J (функция Леверетта)", options=cols, index=_idx_or_zero(j_guess))
    default_swn = swn_pick if swn_pick in cols else cols[0]
    swn_idx = cols.index(default_swn) if default_swn in cols else 0
    swn_col = c4.selectbox(
        "Swn (по умолчанию — «второй» SWn-кандидат, если есть)",
        options=cols,
        index=swn_idx,
        help=f"Найденные SWn-колонки: {swn_list}",
    )

    areas = sorted(pd.Series(df[area_col]).dropna().astype(str).str.strip().unique().tolist())
    horizons = sorted(pd.Series(df[hor_col]).dropna().astype(str).str.strip().unique().tolist())

    st.subheader("Фильтр данных")
    st.caption("Площади — отметьте галочками; горизонты — выберите несколько из списка.")
    sel_areas: list[str] = []
    if areas:
        ncols = min(4, max(1, len(areas)))
        grid = st.columns(ncols)
        for i, a in enumerate(areas):
            if grid[i % ncols].checkbox(str(a), key=f"lab_chk_area_{i}"):
                sel_areas.append(str(a))
    sel_hors = st.multiselect("Коды горизонтов (можно несколько)", options=horizons, default=[])

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
        st.success(f"Сохранено точек: {len(filt)}. Лаб. тренд: a={fit.get('a')}, b={fit.get('b')}, R²={fit.get('r2')}")

    cloud = st.session_state.get("lab_cloud")
    if cloud is None or cloud.empty:
        return

    st.subheader("График J vs Swn (лаборатория)")
    fit = st.session_state.get("lab_trend_fit") or {}
    fig = _fig_j_swn_lab(cloud, title="Лабораторные данные", trend_fit=fit, extra_lines=None, optimal=None)
    st.plotly_chart(fig, use_container_width=True)


def leverett_tab() -> None:
    st.title("Подбор J функции Леверетта")
    st.write("Загрузите данные скважин, задайте ограничения или автограницы, выполните подбор.")

    lab_ready = bool(st.session_state.get("lab_cloud_ready"))
    cloud: pd.DataFrame | None = st.session_state.get("lab_cloud")

    with st.sidebar:
        st.header("Загрузка данных")
        wells_file = st.file_uploader("Файл скважин", type=["csv", "xlsx", "xls", "txt"], key="wells_file")
        prod_file = st.file_uploader("Файл добычи (опционально для весов)", type=["csv", "xlsx", "xls"], key="prod_file")
        maxiter = st.slider("Итерации оптимизации", min_value=20, max_value=300, value=200, step=10)
        popsize = st.slider("Размер популяции", min_value=8, max_value=40, value=20, step=1)

    if wells_file is None:
        st.info("Загрузите файл скважин, чтобы продолжить.")
        return

    try:
        raw_wells = _read_table(wells_file)
        raw_prod = _read_table(prod_file) if prod_file is not None else None
        df_wells = _clean_wells_df(raw_wells)
        df_prod = _clean_prod_df(raw_prod)
    except Exception as e:
        st.error(f"Ошибка чтения файла: {e}")
        return

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
    st.dataframe(df_wells.head(30), use_container_width=True)
    st.caption(f"Строк после очистки: {len(df_wells)}")

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
        st.markdown("**Сопоставление: какой код горизонта относится к какому PVTNUM** (можно несколько)")
        hors_universe = sorted(cloud["lab_horizon"].astype(str).unique().tolist()) if cloud is not None else []
        pvt_horizon_map: dict[int, list[str]] = {}
        for pvt in pvts:
            pvt_horizon_map[pvt] = st.multiselect(
                f"Горизонты для PVTNUM {pvt}",
                options=hors_universe,
                key=f"pvt_hor_map_{pvt}",
            )
        st.session_state["pvt_horizon_map"] = pvt_horizon_map

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

    run_opt = st.button("Рассчитать оптимальные коэффициенты", type="primary")
    run_fix = st.button("Рассчитать с заданными коэффициентами", type="secondary") if use_fixed else False

    def _merge_bounds(partial: dict[int, dict[str, tuple[float, float]]]) -> dict[int, dict[str, tuple[float, float]]]:
        merged = _default_bounds_for_pvts(pvts)
        merged.update(partial)
        return merged

    if run_fix:
        if not use_fixed:
            st.error("Включите «Рассчитать с заданными коэффициентами» и задайте a, b, sigma.")
        else:
            with st.spinner("Применяются заданные коэффициенты..."):
                try:
                    result_df, params_df, qa_df = run_pipeline(
                        df_wells=df_wells,
                        df_prod=df_prod,
                        bounds_by_pvt=_merge_bounds(bounds_by_pvt),
                        maxiter=maxiter,
                        popsize=popsize,
                        fixed_params=fixed_params,
                    )
                except Exception as e:
                    st.error(f"Ошибка расчета: {e}")
                else:
                    st.session_state["leverett_result_df"] = result_df
                    st.session_state["leverett_params_df"] = params_df
                    st.session_state["leverett_qa_df"] = qa_df
                    st.success("Расчет с заданными коэффициентами завершен.")

    if run_opt:
        if bounds_mode == "Автоподбор по лабораторному облаку" and not bounds_by_pvt:
            st.error("Сначала нажмите «Подобрать границы a, b автоматически по облаку».")
        else:
            with st.spinner("Выполняется подбор коэффициентов..."):
                try:
                    result_df, params_df, qa_df = run_pipeline(
                        df_wells=df_wells,
                        df_prod=df_prod,
                        bounds_by_pvt=_merge_bounds(bounds_by_pvt),
                        maxiter=maxiter,
                        popsize=popsize,
                        fixed_params=None,
                    )
                except Exception as e:
                    st.error(f"Ошибка расчета: {e}")
                else:
                    st.session_state["leverett_result_df"] = result_df
                    st.session_state["leverett_params_df"] = params_df
                    st.session_state["leverett_qa_df"] = qa_df
                    st.success("Расчет завершен.")

    result_df = st.session_state.get("leverett_result_df")
    params_df = st.session_state.get("leverett_params_df")
    qa_df = st.session_state.get("leverett_qa_df")
    if result_df is None or params_df is None or qa_df is None:
        st.info("Запустите расчет, чтобы увидеть результаты и графики.")
        return

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Параметры")
        st.dataframe(params_df, use_container_width=True)
    with c2:
        st.subheader("Метрики")
        st.dataframe(qa_df, use_container_width=True)

    if lab_ready and cloud is not None and not cloud.empty:
        st.subheader("J vs Swn: лаборатория + степенная модель с оптимальными a, b")
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
    region_df = result_df.loc[region_mask].copy()
    region_df = region_df.dropna(subset=["Кнг_W", "Kng_model"])

    if region_df.empty:
        st.warning("Для выбранного региона нет валидных точек.")
        return

    hover_cols = [c for c in ["WELL_NAME", "PC", "PORO_GDM", "PERM_GDM", "SWL_GDM", "Кнг_W", "Kng_model", "weight"] if c in region_df.columns]
    fig_scatter = px.scatter(
        region_df,
        x="Кнг_W",
        y="Kng_model",
        color="weight" if "weight" in region_df.columns else None,
        color_continuous_scale="Viridis",
        hover_data=hover_cols,
        title=f"PVT {region}: историческое vs предсказанное Кнг",
        opacity=0.7,
    )
    fig_scatter.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(color="red", dash="dash"))
    fig_scatter.update_layout(xaxis_title="Кнг историческое (ГИС)", yaxis_title="Кнг предсказанное")
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("#### Аналитика по выбранному региону")
    region_wells_count = region_df["WELL_NAME"].astype(str).nunique() if "WELL_NAME" in region_df.columns else 0
    region_points_count = len(region_df)
    m1, m2 = st.columns(2)
    m1.metric("Уникальных скважин (PVTNUM)", int(region_wells_count))
    m2.metric("Всего точек (PVTNUM)", int(region_points_count))

    if "weight" in result_df.columns and "WELL_NAME" in result_df.columns:
        weight_summary = (
            result_df.groupby("WELL_NAME", as_index=False)
            .agg(avg_weight=("weight", "mean"), max_weight=("weight", "max"), points=("weight", "size"))
            .sort_values(["avg_weight", "max_weight"], ascending=False)
        )
        st.markdown("**Скважины с наибольшими весами (в целом по выборке)**")
        st.dataframe(weight_summary.head(15), use_container_width=True)

    if not qa_df.empty:
        qa_row = qa_df[qa_df["PVTNUM_GDM"] == int(region)]
        if not qa_row.empty:
            qa_row = qa_row.iloc[0]
            recs = []
            if qa_row["R2"] < 0.5:
                recs.append("Низкий R2: сузьте диапазоны a/b или увеличьте maxiter/popsize.")
            if abs(qa_row["BIAS"]) > 0.08:
                recs.append("Высокий BIAS: проверьте sigma и SWL_GDM.")
            if qa_row["RMSE"] > 0.15:
                recs.append("Повышенный RMSE: проверьте выбросы PC/PERM/PORO и очистку данных.")
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
            fig_well.update_yaxes(autorange="reversed")
            fig_well.update_layout(xaxis_title="Значение", yaxis_title="Глубина")
            st.plotly_chart(fig_well, use_container_width=True)

            conv_percent = _well_convergence_percent_weighted(well_df)
            if np.isfinite(conv_percent):
                st.metric("Сходимость для скважины (средневзвеш.), %", f"{conv_percent:.2f}")

    csv_result = result_df.to_csv(index=False).encode("utf-8")
    csv_params = params_df.to_csv(index=False).encode("utf-8")
    csv_qa = qa_df.to_csv(index=False).encode("utf-8")
    c3, c4, c5 = st.columns(3)
    c3.download_button("Скачать результат", data=csv_result, file_name="leverett_result.csv", mime="text/csv")
    c4.download_button("Скачать параметры", data=csv_params, file_name="leverett_params.csv", mime="text/csv")
    c5.download_button("Скачать метрики", data=csv_qa, file_name="leverett_metrics.csv", mime="text/csv")


def main() -> None:
    page = st.sidebar.radio("Раздел", options=["Подбор J функции Леверетта", "Лаборатория"])
    if page == "Подбор J функции Леверетта":
        leverett_tab()
    else:
        laboratory_tab()


if __name__ == "__main__":
    main()
