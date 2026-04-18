import io
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from optimize import run_pipeline


st.set_page_config(page_title="Подбор J-функции Леверетта", layout="wide")


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
        df.columns
        .astype(str)
        .str.strip()
        .str.replace('"', '', regex=False)
        .str.replace('\ufeff', '', regex=False)
    )
    return df


def _clean_wells_df(df: pd.DataFrame) -> pd.DataFrame:
    """Очистка загруженных данных по логике read_clean_data.py."""
    df = _normalize_columns(df.copy())

    if df.empty:
        return df

    numeric_candidates = [col for col in df.columns if col != df.columns[0]]
    for col in numeric_candidates:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Эквивалент: df = df[~(df[columns_to_convert] <= -1).any(axis=1)].dropna()
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
    # Логика из read_clean_data.py: убрать переносы в названиях колонок
    df = df.rename(columns=lambda x: str(x).replace("\n", ""))
    return df

def _default_bounds_for_pvts(pvts: list[int]) -> dict[int, dict[str, tuple[float, float]]]:
    return {pvt: {"a": (0.05, 0.30), "b": (-3.0, -0.5), "sigma": (25.0, 35.0)} for pvt in pvts}


def _bounds_ui(pvts: list[int]) -> dict[int, dict[str, tuple[float, float]]]:
    st.subheader("Ограничения коэффициентов по регионам (PVTNUM)")
    st.caption("Для каждого региона задайте диапазоны a, b и sigma.")
    bounds = _default_bounds_for_pvts(pvts)
    for pvt in pvts:
        with st.expander(f"PVTNUM {pvt}", expanded=False):
            c1, c2 = st.columns(2)
            with c1:
                a_min = st.number_input(f"a min | PVT {pvt}", value=bounds[pvt]["a"][0], key=f"a_min_{pvt}")
                b_min = st.number_input(f"b min | PVT {pvt}", value=bounds[pvt]["b"][0], key=f"b_min_{pvt}")
                s_min = st.number_input(
                    f"sigma min | PVT {pvt}", value=bounds[pvt]["sigma"][0], key=f"s_min_{pvt}"
                )
            with c2:
                a_max = st.number_input(f"a max | PVT {pvt}", value=bounds[pvt]["a"][1], key=f"a_max_{pvt}")
                b_max = st.number_input(f"b max | PVT {pvt}", value=bounds[pvt]["b"][1], key=f"b_max_{pvt}")
                s_max = st.number_input(
                    f"sigma max | PVT {pvt}", value=bounds[pvt]["sigma"][1], key=f"s_max_{pvt}"
                )

            if a_min >= a_max or b_min >= b_max or s_min >= s_max:
                st.error("Минимум должен быть меньше максимума.")
            bounds[pvt] = {"a": (a_min, a_max), "b": (b_min, b_max), "sigma": (s_min, s_max)}
    return bounds


def _validate_columns(df_wells: pd.DataFrame, df_prod: pd.DataFrame | None) -> list[str]:
    errors: list[str] = []
    required = {"WELL_NAME", "PVTNUM_GDM", "PORO_GDM", "PERM_GDM", "PC", "SWL_GDM", "Кнг_W"}
    missing = sorted(required - set(df_wells.columns))
    if missing:
        errors.append(f"В файле скважин не хватает колонок: {missing}")
    if df_prod is not None and not df_prod.empty:
        prod_ok = any(
            col in df_prod.columns
            for col in ["Ствол скважины", "WELL_NAME"]
        ) and any(col in df_prod.columns for col in ["Экспл. объект", "PVTNUM_GDM"])
        if not prod_ok:
            errors.append("Файл добычи должен содержать колонки скважины и региона (PVTNUM).")
    return errors


def leverett_tab() -> None:
    st.title("Подбор J-функции Леверетта")
    st.write("Загрузите данные, задайте ограничения коэффициентов по регионам и запустите подбор.")

    with st.sidebar:
        st.header("Загрузка данных")
        wells_file = st.file_uploader("Файл скважин", type=["csv", "xlsx", "xls", "txt"], key="wells_file")
        prod_file = st.file_uploader(
            "Файл добычи (опционально для весов)",
            type=["csv", "xlsx", "xls"],
            key="prod_file",
        )
        maxiter = st.slider("Итерации оптимизации", min_value=20, max_value=300, value=120, step=10)
        popsize = st.slider("Размер популяции", min_value=8, max_value=40, value=15, step=1)

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

    bounds_by_pvt = _bounds_ui(pvts)
    run_clicked = st.button("Рассчитать оптимальные коэффициенты", type="primary")
    if run_clicked:
        with st.spinner("Выполняется подбор коэффициентов..."):
            try:
                result_df, params_df, qa_df = run_pipeline(
                    df_wells=df_wells,
                    df_prod=df_prod,
                    bounds_by_pvt=bounds_by_pvt,
                    maxiter=maxiter,
                    popsize=popsize,
                )
            except Exception as e:
                st.error(f"Ошибка расчета: {e}")
                return

        st.session_state["leverett_result_df"] = result_df
        st.session_state["leverett_params_df"] = params_df
        st.session_state["leverett_qa_df"] = qa_df
        st.success("Расчет завершен.")

    result_df = st.session_state.get("leverett_result_df")
    params_df = st.session_state.get("leverett_params_df")
    qa_df = st.session_state.get("leverett_qa_df")
    if result_df is None or params_df is None or qa_df is None:
        st.info("Нажмите 'Рассчитать оптимальные коэффициенты', чтобы увидеть результаты и графики.")
        return

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Оптимальные параметры")
        st.dataframe(params_df, use_container_width=True)
    with c2:
        st.subheader("Таблица метрик")
        st.dataframe(qa_df, use_container_width=True)

    st.subheader("Интерактивные графики")
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

    hover_cols = []
    for col in ["WELL_NAME", "PC", "PORO_GDM", "PERM_GDM", "SWL_GDM", "Кнг_W", "Kng_model"]:
        if col in region_df.columns:
            hover_cols.append(col)

    fig_scatter = px.scatter(
        region_df,
        x="Кнг_W",
        y="Kng_model",
        hover_data=hover_cols,
        title=f"PVT {region}: историческое vs предсказанное Кнг",
        opacity=0.7,
    )
    fig_scatter.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(color="red", dash="dash"))
    fig_scatter.update_layout(xaxis_title="Кнг историческое (ГИС)", yaxis_title="Кнг предсказанное")
    st.plotly_chart(fig_scatter, use_container_width=True)

    if "WELL_NAME" in region_df.columns:
        wells = sorted(region_df["WELL_NAME"].astype(str).unique().tolist())
        well = st.selectbox("Скважина для детального графика", options=wells)
        well_df = region_df[region_df["WELL_NAME"].astype(str) == well].copy()
        well_df = well_df.reset_index(drop=True)
        well_df["point_id"] = well_df.index + 1
        chart_df = well_df.melt(
            id_vars=["point_id", "WELL_NAME"],
            value_vars=["Кнг_W", "Kng_model"],
            var_name="type",
            value_name="Kng",
        )
        fig_well = px.line(
            chart_df,
            x="point_id",
            y="Kng",
            color="type",
            markers=True,
            hover_data=["WELL_NAME", "type", "Kng"],
            title=f"Скважина {well}: сравнение кривых",
        )
        fig_well.update_layout(xaxis_title="Номер точки", yaxis_title="Кнг")
        st.plotly_chart(fig_well, use_container_width=True)

    csv_result = result_df.to_csv(index=False).encode("utf-8")
    csv_params = params_df.to_csv(index=False).encode("utf-8")
    csv_qa = qa_df.to_csv(index=False).encode("utf-8")
    c3, c4, c5 = st.columns(3)
    c3.download_button("Скачать результат", data=csv_result, file_name="leverett_result.csv", mime="text/csv")
    c4.download_button("Скачать параметры", data=csv_params, file_name="leverett_params.csv", mime="text/csv")
    c5.download_button("Скачать метрики", data=csv_qa, file_name="leverett_metrics.csv", mime="text/csv")


def main() -> None:
    page = st.sidebar.radio("Раздел", options=["Подбор J функции Леверетта"])
    if page == "Подбор J функции Леверетта":
        leverett_tab()


if __name__ == "__main__":
    main()
