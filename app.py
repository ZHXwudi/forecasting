from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from forecasting import (
    HOURS_PER_YEAR,
    MODEL_DESCRIPTIONS,
    MODEL_ENSEMBLE,
    MODEL_RIDGE,
    MODEL_SEASONAL_NAIVE,
    describe_data,
    forecast_future,
    format_metric_table,
    load_price_data,
    run_backtests,
)


st.set_page_config(page_title="国外电价预测模型", page_icon="⚡", layout="wide")


@st.cache_data(show_spinner=False)
def cached_load(source_name: str, uploaded_bytes: bytes | None) -> pd.DataFrame:
    if uploaded_bytes is None:
        return load_price_data(source_name)
    from io import BytesIO

    return load_price_data(BytesIO(uploaded_bytes))


@st.cache_data(show_spinner=False)
def cached_backtests(df: pd.DataFrame):
    return run_backtests(df)


@st.cache_data(show_spinner=False)
def cached_future(df: pd.DataFrame, hours: int):
    return forecast_future(df, hours)


st.title("国外电价多模型预测")

with st.sidebar:
    st.header("数据与预测")
    uploaded = st.file_uploader("上传电价 CSV", type=["csv"])
    default_path = Path("price_input_foreign.csv")
    source_name = str(default_path)
    uploaded_bytes = uploaded.getvalue() if uploaded is not None else None
    forecast_days = st.slider("未来预测天数", min_value=7, max_value=365, value=365, step=7)
    show_points = st.slider("图表显示小时数", min_value=168, max_value=8760, value=720, step=168)

try:
    with st.spinner("正在清洗数据并训练模型..."):
        df = cached_load(source_name, uploaded_bytes)
        backtests = cached_backtests(df)
        future, future_weights = cached_future(df, forecast_days * 24)
except Exception as exc:
    st.error(f"运行失败：{exc}")
    st.stop()

summary = describe_data(df)
latest_backtest = backtests[-1]
best_row = latest_backtest.metrics.iloc[0]

col1, col2, col3, col4 = st.columns(4)
col1.metric("数据小时数", f"{summary['rows']:,}")
col2.metric("日期范围", f"{summary['start']:%Y-%m-%d} 至 {summary['end']:%Y-%m-%d}")
col3.metric("最新验证年", str(latest_backtest.validation_year))
col4.metric("最佳模型 RMSE", f"{best_row['RMSE']:.2f}")

tabs = st.tabs(["模型对比", "回测曲线", "未来预测", "数据概览"])

with tabs[0]:
    st.subheader("各模型预测准确率")
    st.caption("下表先解释每个模型的含义，再用回测指标比较准确率。RMSE 和 MAE 越低越好，R2 越接近 1 越好。")
    st.dataframe(
        pd.DataFrame(
            [{"模型": model_name, "中文解释": description} for model_name, description in MODEL_DESCRIPTIONS.items()]
        ),
        use_container_width=True,
        hide_index=True,
    )

    selected_year = st.selectbox(
        "验证年份",
        options=[result.validation_year for result in backtests],
        index=len(backtests) - 1,
    )
    selected_result = next(result for result in backtests if result.validation_year == selected_year)

    metrics = format_metric_table(selected_result.metrics)
    st.dataframe(metrics, use_container_width=True, hide_index=True)

    metric_chart = selected_result.metrics.sort_values("RMSE")
    fig = px.bar(
        metric_chart,
        x="model",
        y="RMSE",
        color="model",
        title=f"{selected_year} 年验证集 RMSE 对比",
    )
    fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="RMSE")
    st.plotly_chart(fig, use_container_width=True)

    weight_cols = st.columns(2)
    with weight_cols[0]:
        st.caption("该验证年的集成权重")
        st.dataframe(selected_result.ensemble_weights, use_container_width=True, hide_index=True)
    with weight_cols[1]:
        st.caption("未来预测使用的集成权重")
        st.dataframe(future_weights, use_container_width=True, hide_index=True)

with tabs[1]:
    st.subheader("验证集真实值与预测值")
    model_options = [column for column in selected_result.predictions.columns if column not in {"ds", "y", "validation_year"}]
    chosen_models = st.multiselect(
        "显示模型",
        options=model_options,
        default=[MODEL_ENSEMBLE, MODEL_RIDGE, MODEL_SEASONAL_NAIVE],
    )
    plot_data = selected_result.predictions.tail(show_points)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=plot_data["ds"], y=plot_data["y"], name="真实电价", mode="lines"))
    for model_name in chosen_models:
        fig.add_trace(go.Scatter(x=plot_data["ds"], y=plot_data[model_name], name=model_name, mode="lines"))
    fig.update_layout(xaxis_title="", yaxis_title="电价", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

with tabs[2]:
    st.subheader("未来电价预测")
    forecast_model = st.selectbox(
        "预测曲线",
        options=[column for column in future.columns if column != "ds"],
        index=list(future.columns).index(MODEL_ENSEMBLE) - 1,
    )
    recent = df.tail(min(show_points, len(df))).rename(columns={"y": "真实电价"})
    future_plot = future.head(show_points)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recent["ds"], y=recent["真实电价"], name="历史电价", mode="lines"))
    fig.add_trace(go.Scatter(x=future_plot["ds"], y=future_plot[forecast_model], name=forecast_model, mode="lines"))
    fig.update_layout(xaxis_title="", yaxis_title="电价", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    st.download_button(
        "下载未来预测 CSV",
        data=future.to_csv(index=False, encoding="utf-8-sig"),
        file_name="future_forecast.csv",
        mime="text/csv",
    )
    st.dataframe(future.head(1000), use_container_width=True, hide_index=True)

with tabs[3]:
    st.subheader("数据概览")
    overview_cols = st.columns(3)
    overview_cols[0].metric("最低电价", f"{summary['min_price']:.2f}")
    overview_cols[1].metric("平均电价", f"{summary['mean_price']:.2f}")
    overview_cols[2].metric("最高电价", f"{summary['max_price']:.2f}")

    yearly = df.assign(year=df["ds"].dt.year).groupby("year")["y"].agg(["count", "mean", "min", "max"]).reset_index()
    st.dataframe(yearly, use_container_width=True, hide_index=True)

    history = df.tail(min(HOURS_PER_YEAR, len(df)))
    fig = px.line(history, x="ds", y="y", title="最近一年历史电价")
    fig.update_layout(xaxis_title="", yaxis_title="电价")
    st.plotly_chart(fig, use_container_width=True)
