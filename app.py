from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from forecasting import (
    HOURS_PER_YEAR,
    MODEL_ENSEMBLE,
    MODEL_LSTM,
    MODEL_RIDGE,
    MODEL_SEASONAL_NAIVE,
    build_model_catalog,
    describe_data,
    forecast_future,
    format_metric_table,
    load_price_data,
    run_backtests,
)


st.set_page_config(page_title="国外电价预测模型", page_icon="⚡", layout="wide")
CACHE_VERSION = "models_en_cn_intro_v4"


@st.cache_data(show_spinner=False)
def cached_load(source_name: str, uploaded_bytes: bytes | None) -> pd.DataFrame:
    if uploaded_bytes is None:
        return load_price_data(source_name)
    from io import BytesIO

    return load_price_data(BytesIO(uploaded_bytes))


@st.cache_data(show_spinner=False)
def cached_backtests(df: pd.DataFrame, cache_version: str):
    return run_backtests(df)


@st.cache_data(show_spinner=False)
def cached_future(df: pd.DataFrame, hours: int, cache_version: str):
    return forecast_future(df, hours)


def prettify_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    table = format_metric_table(metrics)
    return table.rename(
        columns={
            "model": "模型",
            "MAE": "MAE",
            "RMSE": "RMSE",
            "WMAPE": "WMAPE (%)",
            "sMAPE": "sMAPE (%)",
            "R2": "R2",
        }
    )


def render_intro_page() -> None:
    st.title("预处理与故障识别说明")
    st.caption("这一页用于说明今后模型如何从原始时序信号出发，完成滤波去噪、异常剔除、特征提取、故障分类和异常精准检测。")

    st.subheader("整体流程")
    st.markdown(
        """
        1. 原始数据接入：采集电价、电流、电压、功率、温度、振动或设备状态量，统一时间戳和采样频率。
        2. 预处理清洗：完成缺失补全、异常值剔除、滤波去噪、归一化和分段标注。
        3. 频域与时频特征提取：结合小波分解、傅里叶变换、统计量和时序上下文构造特征矩阵。
        4. 模型训练识别：用分类模型识别故障类型，用检测模型定位异常时段和异常程度。
        5. 结果输出与告警：给出故障类别、异常分数、置信度、影响区间和处置建议。
        """
    )

    st.subheader("预处理建议")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
            **滤波去噪**

            - 对高频毛刺和随机噪声，可优先使用滑动中值滤波、Savitzky-Golay 滤波或小波阈值去噪。
            - 对工频扰动或明显周期噪声，可结合带通/带阻滤波，保留主要工作频段。
            - 对非平稳信号，建议优先使用多尺度小波分解，再重构有效频段信号。

            **异常值剔除**

            - 先用物理边界阈值剔除不可能值，例如负功率、超量程突变。
            - 再用 Hampel、IQR 或 3 sigma 方法识别孤立离群点。
            - 对连续异常段，不建议直接删除，应保留异常标签供后续异常检测模型学习。
            """
        )
    with col2:
        st.markdown(
            """
            **缺失补全**

            - 短时缺失可用线性插值、样条插值或同周期历史值回填。
            - 长时缺失建议按工况分段补全，避免跨状态硬插值。

            **标准化与切片**

            - 连续特征用 Z-score 或 RobustScaler 标准化。
            - 按固定窗口切片，例如 128 点、256 点或 24/168 小时时序窗口。
            - 为每个窗口同步保留标签、设备编号、工况编号和时间位置。
            """
        )

    st.subheader("小波与傅里叶特征提取")
    st.markdown(
        """
        - 小波变换：适合非平稳故障信号。可提取各尺度能量、能量熵、峰值、偏度、峭度以及重构后分量统计量，用于捕捉突发冲击和局部异常。
        - FFT/频谱分析：适合识别稳定周期成分和谐波结构。可提取主频、倍频幅值、谱重心、谱峭度、带宽、频带能量比等指标。
        - 时频联合：把小波包能量谱、短时傅里叶谱图或连续小波时频图输入 CNN/LSTM/Transformer，可进一步提升复杂故障识别能力。
        - 多源融合：将频域特征与温度、电压、负荷、天气、节假日等上下文特征拼接，提升解释性和分类稳定性。
        """
    )

    st.subheader("故障分类与异常精准检测方案")
    st.markdown(
        """
        **故障分类识别**

        - 规则基线：先建立阈值和专家规则，快速拦截明显异常和高风险故障。
        - 传统机器学习：对结构化特征可优先尝试 Random Forest、XGBoost、LightGBM、SVM。
        - 深度学习：对长序列和复杂耦合信号，可采用 1D CNN、LSTM、CNN-LSTM、Transformer。
        - 多分类输出：建议同时输出故障类型、故障等级和置信度，便于工程应用。

        **异常精准检测**

        - 无监督检测：当异常样本少时，可采用 Isolation Forest、One-Class SVM、Autoencoder。
        - 预测残差检测：先做正常工况预测，再对残差做控制图、动态阈值或概率密度检验。
        - 在线检测：对实时流数据，建议使用滑动窗口 + 动态阈值，输出异常分数与触发原因。
        - 精准定位：将异常检测与变点检测结合，可更准确识别异常起点、持续时长和恢复时间。
        """
    )

    st.subheader("落地实施建议")
    st.info(
        "建议先把现有预测模型作为基线层，逐步增加预处理、时频特征和故障识别模块。上线时可采用“规则 + 统计模型 + 深度模型”三级结构，兼顾稳定性、精度和可解释性。"
    )

    roadmap = pd.DataFrame(
        [
            {"阶段": "第一阶段", "目标": "完成清洗、去噪、异常值标注", "产出": "稳定训练样本库"},
            {"阶段": "第二阶段", "目标": "接入小波/傅里叶特征工程", "产出": "结构化特征矩阵"},
            {"阶段": "第三阶段", "目标": "训练故障分类与异常检测模型", "产出": "分类器与检测器"},
            {"阶段": "第四阶段", "目标": "接入实时评分和告警", "产出": "在线诊断页面与告警接口"},
        ]
    )
    st.dataframe(roadmap, use_container_width=True, hide_index=True)


def render_forecast_page() -> None:
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
            backtests = cached_backtests(df, CACHE_VERSION)
            future, future_weights = cached_future(df, forecast_days * 24, CACHE_VERSION)
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
        st.caption("模型标识采用英文主名称，并在括号中保留中文简写。RMSE 和 MAE 越低越好，R2 越接近 1 越好。")
        st.dataframe(build_model_catalog(), use_container_width=True, hide_index=True)

        selected_year = st.selectbox(
            "验证年份",
            options=[result.validation_year for result in backtests],
            index=len(backtests) - 1,
        )
        selected_result = next(result for result in backtests if result.validation_year == selected_year)

        metrics = prettify_metrics(selected_result.metrics)
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
            st.dataframe(
                selected_result.ensemble_weights.rename(columns={"model": "模型", "weight": "权重"}),
                use_container_width=True,
                hide_index=True,
            )
        with weight_cols[1]:
            st.caption("未来预测使用的集成权重")
            st.dataframe(
                future_weights.rename(columns={"model": "模型", "weight": "权重"}),
                use_container_width=True,
                hide_index=True,
            )

    with tabs[1]:
        st.subheader("验证集真实值与预测值")
        model_options = [column for column in selected_result.predictions.columns if column not in {"ds", "y", "validation_year"}]
        preferred_models = [MODEL_ENSEMBLE, MODEL_LSTM, MODEL_RIDGE, MODEL_SEASONAL_NAIVE]
        default_models = [model for model in preferred_models if model in model_options]
        if not default_models:
            default_models = model_options[: min(3, len(model_options))]
        chosen_models = st.multiselect(
            "显示模型",
            options=model_options,
            default=default_models,
            key="backtest_model_select_v4",
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
        future_model_options = [column for column in future.columns if column != "ds"]
        if not future_model_options:
            st.error("未来预测结果中没有可展示的模型列，请重新运行预测。")
            st.stop()
        default_future_index = future_model_options.index(MODEL_ENSEMBLE) if MODEL_ENSEMBLE in future_model_options else 0
        forecast_model = st.selectbox(
            "预测曲线",
            options=future_model_options,
            index=default_future_index,
            key="future_model_select_v4",
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

with st.sidebar:
    st.header("页面导航")
    page = st.radio("选择页面", options=["预测模型分析", "预处理与故障识别说明"], index=0)

if page == "预测模型分析":
    render_forecast_page()
else:
    render_intro_page()
