# 国外电价多模型预测

这个项目用小时级国外电价数据训练并对比多种预测方法，支持 Streamlit 可视化展示、历史年份回测和未来电价预测导出。

## 当前模型

- 年度同小时基准模型：用上一年同一日期、同一小时的电价作为预测值，适合年度季节性很强、规律重复明显的数据。
- 日历画像中位数模型：按月份、星期几和小时统计历史电价中位数，形成典型日历画像，再用相同日历特征预测未来。
- 近期修正季节模型：先沿用上一年同小时规律，再根据最近几周与历史同期的均值差异做趋势修正。
- 岭回归自回归模型：使用小时、星期、月份、年度周期特征，以及 1 小时、24 小时、168 小时和 8760 小时滞后电价，通过岭回归学习规律。
- 最优加权集成模型：根据历史回测误差自动优化多个基础模型的权重，综合得到最终预测结果。

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

生成文件：

- `outputs/backtest_metrics.csv`：各验证年份、各模型 MAE/RMSE/WMAPE/sMAPE/R2。
- `outputs/backtest_predictions.csv`：验证集真实值和预测值。
- `outputs/future_forecast.csv`：未来电价预测。
- `outputs/ensemble_weights.csv`：未来预测采用的集成模型权重。

## 数据格式

默认读取 `price_input_foreign.csv`，支持包含以下列的 CSV：

- `日期`
- `时间`
- `电价`

程序会自动删除全空行、解析日期时间、补齐小时频率，并对缺失小时做时间插值。

## GitHub 上传流程

等你创建好 GitHub 仓库后，在本目录执行：

```bash
git init
git add .
git commit -m "Add electricity price forecasting Streamlit app"
git branch -M main
git remote add origin <你的仓库地址>
git push -u origin main
```
