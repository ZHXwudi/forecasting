# 国外电价多模型预测

这个项目用于对小时级国外电价数据进行多模型预测、历史回测和未来趋势输出，并提供 Streamlit 可视化页面。当前版本已经将模型标识统一为英文主名称，同时在括号中保留中文简写，便于展示、汇报和后续 GitHub 协作。

## 当前模型

- `Seasonal Naive (同小时)`：使用上一年同一日期、同一小时的电价作为预测值，适合季节性强、周期重复明显的数据。
- `Calendar Profile (画像)`：按月份、星期和小时统计历史中位数，形成日历画像后进行预测。
- `Recent Seasonal Adjustment (修正)`：在同小时季节基线的基础上，根据近几周与历史同期差异做趋势修正。
- `Ridge Autoregression (岭回归)`：使用日历周期特征与多个滞后项，通过岭回归学习时序规律。
- `Exponential Smoothing (平滑)`：使用指数平滑建模近期趋势，并叠加小时与星期季节性。
- `LSTM (深度学习)`：优先调用 TensorFlow LSTM；若环境缺少 TensorFlow，则自动回退到稳定的序列窗口近似模型。
- `Weighted Ensemble (集成)`：根据历史回测误差自动优化多模型权重，输出最终集成预测结果。

## Streamlit 页面

应用包含两个页面：

- `预测模型分析`：展示模型说明、回测指标、回测曲线、未来预测和数据概览。
- `预处理与故障识别说明`：说明后续如何开展滤波去噪、异常值剔除、缺失补全、小波/傅里叶特征提取，以及故障分类识别与异常精准检测。

说明页的目标是为下一步功能扩展提供统一方案，建议后续按以下路径推进：

1. 完成原始信号清洗、异常标签沉淀和标准化。
2. 引入小波分解、FFT、谱能量和统计量等特征工程。
3. 组合分类模型与异常检测模型，形成故障识别链路。
4. 接入实时评分、告警和诊断页面。

## 预处理与特征工程建议

- 滤波去噪：可使用滑动中值滤波、Savitzky-Golay、小波阈值去噪、带通或带阻滤波。
- 异常值剔除：建议结合物理阈值、Hampel、IQR、3 sigma 等方法识别孤立离群点。
- 缺失补全：短缺口可插值，长缺口建议按工况分段补全。
- 小波特征：可提取多尺度能量、能量熵、峰值、偏度、峭度和重构分量统计量。
- 傅里叶特征：可提取主频、倍频幅值、谱重心、谱峭度、带宽和带能量比。
- 模型路线：结构化特征可尝试 Random Forest、XGBoost、LightGBM、SVM；长序列可尝试 1D CNN、LSTM、CNN-LSTM、Transformer。

## 使用方式

安装依赖：

```bash
pip install -r requirements.txt
```

启动 Streamlit：

```bash
streamlit run app.py
```

命令行生成结果：

```bash
python run_forecast.py --data price_input_foreign.csv --output outputs --future-hours 8760
```

## 输出文件

- `outputs/backtest_metrics.csv`：各验证年份、各模型的 MAE、RMSE、WMAPE、sMAPE、R2。
- `outputs/backtest_predictions.csv`：验证集真实值与各模型预测值。
- `outputs/future_forecast.csv`：未来时段预测结果。
- `outputs/ensemble_weights.csv`：未来预测使用的集成权重。

## 数据格式

默认读取 `price_input_foreign.csv`，支持包含以下列名的 CSV：

- `日期`
- `时间`
- `电价`

程序会自动删除全空行、解析日期时间、补齐小时频率，并对缺失小时进行时间插值。
