# 国外电价多模型预测

这个项目用小时级国外电价数据训练并对比多种预测方法，支持 Streamlit 可视化展示、历史年份回测和未来电价预测导出。

## 当前模型

- Seasonal naive：按上一年同小时复制，适合强年度季节性电价。
- Calendar profile：按月份、星期、小时构造历史中位数画像。
- Recent adjusted seasonal：在年度季节性基础上加入近期均值漂移修正。
- Ridge autoregression：使用日内、周内、年度周期特征和 1/24/168/8760 小时滞后项。
- Optimized ensemble：根据历史回测误差最优化多模型权重。

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
