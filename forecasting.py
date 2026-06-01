from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


HOURS_PER_YEAR = 8760


@dataclass(frozen=True)
class ModelProfile:
    english_name: str
    short_name_cn: str
    description: str


def _model_label(english_name: str, short_name_cn: str) -> str:
    return f"{english_name} ({short_name_cn})"


MODEL_SEASONAL_NAIVE = _model_label("Seasonal Naive", "同小时")
MODEL_CALENDAR_PROFILE = _model_label("Calendar Profile", "画像")
MODEL_RECENT_ADJUSTED = _model_label("Recent Seasonal Adjustment", "修正")
MODEL_RIDGE = _model_label("Ridge Autoregression", "岭回归")
MODEL_EXP_SMOOTHING = _model_label("Exponential Smoothing", "平滑")
MODEL_LSTM = _model_label("LSTM", "深度学习")
MODEL_ENSEMBLE = _model_label("Weighted Ensemble", "集成")
BASE_MODEL_NAMES = [
    MODEL_SEASONAL_NAIVE,
    MODEL_CALENDAR_PROFILE,
    MODEL_RECENT_ADJUSTED,
    MODEL_RIDGE,
    MODEL_EXP_SMOOTHING,
    MODEL_LSTM,
]
MODEL_DISPLAY_ORDER = [*BASE_MODEL_NAMES, MODEL_ENSEMBLE]
MODEL_METADATA = {
    MODEL_SEASONAL_NAIVE: ModelProfile(
        english_name="Seasonal Naive",
        short_name_cn="同小时",
        description="用上一年同一日期、同一小时的电价作为预测值，适合年度季节性很强、规律重复明显的数据。",
    ),
    MODEL_CALENDAR_PROFILE: ModelProfile(
        english_name="Calendar Profile",
        short_name_cn="画像",
        description="按月份、星期几和小时统计历史电价中位数，形成典型日历画像，再用相同日历特征预测未来。",
    ),
    MODEL_RECENT_ADJUSTED: ModelProfile(
        english_name="Recent Seasonal Adjustment",
        short_name_cn="修正",
        description="先沿用上一年同小时规律，再根据最近几周与历史同期的均值差异做趋势修正。",
    ),
    MODEL_RIDGE: ModelProfile(
        english_name="Ridge Autoregression",
        short_name_cn="岭回归",
        description="使用小时、星期、月份、年度周期特征，以及 1 小时、24 小时、168 小时和 8760 小时滞后电价，通过岭回归学习规律。",
    ),
    MODEL_EXP_SMOOTHING: ModelProfile(
        english_name="Exponential Smoothing",
        short_name_cn="平滑",
        description="用指数加权平均估计近期趋势，并叠加小时和星期的季节性模式，属于经典时间序列预测方法。",
    ),
    MODEL_LSTM: ModelProfile(
        english_name="LSTM",
        short_name_cn="深度学习",
        description="优先使用 LSTM 神经网络学习连续电价窗口中的非线性变化；如果部署环境没有 TensorFlow，则自动使用稳定的序列窗口近似模型，避免应用报错。",
    ),
    MODEL_ENSEMBLE: ModelProfile(
        english_name="Weighted Ensemble",
        short_name_cn="集成",
        description="根据历史回测误差自动优化多个基础模型的权重，综合得到最终预测结果。",
    ),
}
MODEL_DESCRIPTIONS = {model_name: profile.description for model_name, profile in MODEL_METADATA.items()}


@dataclass
class BacktestResult:
    validation_year: int
    metrics: pd.DataFrame
    predictions: pd.DataFrame
    ensemble_weights: pd.DataFrame


def build_model_catalog() -> pd.DataFrame:
    rows = []
    for model_name in MODEL_DISPLAY_ORDER:
        profile = MODEL_METADATA[model_name]
        rows.append(
            {
                "模型标识": model_name,
                "英文主标识": profile.english_name,
                "中文简写": profile.short_name_cn,
                "说明": profile.description,
            }
        )
    return pd.DataFrame(rows)


def load_price_data(source: str | Path | object) -> pd.DataFrame:
    """Load a CSV with Chinese or generic date/time/price columns."""
    raw = pd.read_csv(source)
    raw = raw.dropna(how="all").copy()
    if raw.empty:
        raise ValueError("CSV 文件没有可用数据。")

    columns = list(raw.columns)
    date_col = "日期" if "日期" in columns else columns[2]
    time_col = "时间" if "时间" in columns else columns[3]
    price_col = "电价" if "电价" in columns else columns[-1]

    dt_text = raw[date_col].astype(str).str.strip() + " " + raw[time_col].astype(str).str.strip()
    parsed = pd.to_datetime(dt_text, dayfirst=True, errors="coerce")
    price = pd.to_numeric(raw[price_col], errors="coerce")

    df = pd.DataFrame({"ds": parsed, "y": price})
    df = df.dropna(subset=["ds", "y"]).sort_values("ds")
    df = df.drop_duplicates(subset=["ds"], keep="last")
    if df.empty:
        raise ValueError("没有成功解析出日期时间和电价，请检查 CSV 列。")

    df = df.set_index("ds").asfreq("h")
    df["y"] = df["y"].interpolate("time").ffill().bfill()
    df = df.reset_index()
    return df


def describe_data(df: pd.DataFrame) -> dict[str, object]:
    years = sorted(df["ds"].dt.year.unique().tolist())
    return {
        "rows": int(len(df)),
        "start": df["ds"].min(),
        "end": df["ds"].max(),
        "years": years,
        "min_price": float(df["y"].min()),
        "max_price": float(df["y"].max()),
        "mean_price": float(df["y"].mean()),
    }


def make_future_index(last_timestamp: pd.Timestamp, hours: int) -> pd.DatetimeIndex:
    start = last_timestamp + pd.Timedelta(hours=1)
    return pd.date_range(start=start, periods=int(hours), freq="h")


def validation_years(df: pd.DataFrame) -> list[int]:
    years = sorted(df["ds"].dt.year.unique().tolist())
    return years[1:]


def train_validate_split(df: pd.DataFrame, validation_year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = df[df["ds"].dt.year < validation_year].copy()
    valid = df[df["ds"].dt.year == validation_year].copy()
    if train.empty or valid.empty:
        raise ValueError("训练集或验证集为空，请选择有连续年份的数据。")
    return train, valid


def forecast_all_models(train: pd.DataFrame, horizon: pd.DatetimeIndex) -> pd.DataFrame:
    forecasts = pd.DataFrame({"ds": horizon})
    forecasts[MODEL_SEASONAL_NAIVE] = seasonal_naive_forecast(train, horizon)
    forecasts[MODEL_CALENDAR_PROFILE] = calendar_profile_forecast(train, horizon)
    forecasts[MODEL_RECENT_ADJUSTED] = recent_adjusted_seasonal_forecast(train, horizon)
    forecasts[MODEL_RIDGE] = ridge_autoregression_forecast(train, horizon)
    forecasts[MODEL_EXP_SMOOTHING] = exponential_smoothing_forecast(train, horizon)
    forecasts[MODEL_LSTM] = lstm_forecast(train, horizon)
    return forecasts


def seasonal_naive_forecast(train: pd.DataFrame, horizon: pd.DatetimeIndex) -> np.ndarray:
    series = train.set_index("ds")["y"].sort_index()
    fallback = float(series.tail(min(len(series), 168)).median())
    values = []
    for ts in horizon:
        source_ts = ts - pd.DateOffset(years=1)
        if source_ts in series.index:
            values.append(float(series.loc[source_ts]))
        else:
            same_hour = series[(series.index.month == ts.month) & (series.index.day == ts.day) & (series.index.hour == ts.hour)]
            values.append(float(same_hour.iloc[-1]) if not same_hour.empty else fallback)
    return np.asarray(values, dtype=float)


def calendar_profile_forecast(train: pd.DataFrame, horizon: pd.DatetimeIndex) -> np.ndarray:
    work = _calendar_frame(train)
    profile_1 = work.groupby(["month", "dayofweek", "hour"])["y"].median()
    profile_2 = work.groupby(["month", "hour"])["y"].median()
    profile_3 = work.groupby(["dayofweek", "hour"])["y"].median()
    fallback = float(work["y"].median())

    values = []
    for ts in horizon:
        key_1 = (ts.month, ts.dayofweek, ts.hour)
        key_2 = (ts.month, ts.hour)
        key_3 = (ts.dayofweek, ts.hour)
        if key_1 in profile_1.index:
            values.append(float(profile_1.loc[key_1]))
        elif key_2 in profile_2.index:
            values.append(float(profile_2.loc[key_2]))
        elif key_3 in profile_3.index:
            values.append(float(profile_3.loc[key_3]))
        else:
            values.append(fallback)
    return np.asarray(values, dtype=float)


def recent_adjusted_seasonal_forecast(train: pd.DataFrame, horizon: pd.DatetimeIndex) -> np.ndarray:
    base = seasonal_naive_forecast(train, horizon)
    series = train.set_index("ds")["y"].sort_index()
    recent = series.tail(min(len(series), 24 * 28))
    old = series.iloc[-HOURS_PER_YEAR : -HOURS_PER_YEAR + len(recent)] if len(series) >= HOURS_PER_YEAR + 24 else pd.Series(dtype=float)
    if old.empty:
        adjustment = float(series.tail(min(len(series), 168)).median() - series.median())
    else:
        adjustment = float(recent.mean() - old.mean())
    return base + adjustment


def ridge_autoregression_forecast(train: pd.DataFrame, horizon: pd.DatetimeIndex, alpha: float = 25.0) -> np.ndarray:
    work = train.sort_values("ds").copy()
    y = work["y"].to_numpy(dtype=float)
    timestamps = pd.DatetimeIndex(work["ds"])
    X = _calendar_features(timestamps)

    lag_columns = []
    for lag in (1, 24, 168, HOURS_PER_YEAR):
        lagged = pd.Series(y).shift(lag).to_numpy(dtype=float)
        lag_columns.append(lagged)
    X = np.column_stack([X, *lag_columns])
    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)

    if mask.sum() < 200:
        return recent_adjusted_seasonal_forecast(train, horizon)

    X_train = X[mask]
    y_train = y[mask]
    x_mean = X_train.mean(axis=0)
    x_std = X_train.std(axis=0)
    x_std[x_std == 0] = 1.0
    y_mean = y_train.mean()
    X_scaled = (X_train - x_mean) / x_std
    y_centered = y_train - y_mean

    identity = np.eye(X_scaled.shape[1])
    coef = np.linalg.solve(X_scaled.T @ X_scaled + alpha * identity, X_scaled.T @ y_centered)

    history = {ts: float(value) for ts, value in zip(timestamps, y)}
    default_value = float(pd.Series(y).tail(min(len(y), 168)).median())
    predictions = []
    for ts in horizon:
        row = _calendar_features(pd.DatetimeIndex([ts]))[0].tolist()
        for lag in (1, 24, 168, HOURS_PER_YEAR):
            row.append(history.get(ts - pd.Timedelta(hours=lag), default_value))
        row_array = np.asarray(row, dtype=float)
        pred = float(((row_array - x_mean) / x_std) @ coef + y_mean)
        pred = float(np.clip(pred, -500.0, 1000.0))
        history[ts] = pred
        predictions.append(pred)
    return np.asarray(predictions, dtype=float)


def exponential_smoothing_forecast(train: pd.DataFrame, horizon: pd.DatetimeIndex) -> np.ndarray:
    series = train.set_index("ds")["y"].sort_index()
    if series.empty:
        return np.zeros(len(horizon), dtype=float)

    level = float(series.ewm(span=min(len(series), 24 * 14), adjust=False).mean().iloc[-1])
    recent = series.tail(min(len(series), 24 * 56))
    recent_level = recent.ewm(span=min(len(recent), 24 * 14), adjust=False).mean()
    residual = recent - recent_level

    frame = pd.DataFrame({"residual": residual})
    frame["hour"] = frame.index.hour
    frame["dayofweek"] = frame.index.dayofweek
    hourly = frame.groupby("hour")["residual"].median()
    weekly = frame.groupby(["dayofweek", "hour"])["residual"].median()

    trend = _estimate_hourly_trend(series)
    values = []
    for step, ts in enumerate(horizon, start=1):
        weekly_key = (ts.dayofweek, ts.hour)
        seasonal = weekly.get(weekly_key, hourly.get(ts.hour, 0.0))
        values.append(level + trend * step + float(seasonal))
    return np.asarray(values, dtype=float)


def lstm_forecast(train: pd.DataFrame, horizon: pd.DatetimeIndex) -> np.ndarray:
    try:
        return _tensorflow_lstm_forecast(train, horizon)
    except Exception:
        return sequence_window_forecast(train, horizon)


def sequence_window_forecast(train: pd.DataFrame, horizon: pd.DatetimeIndex) -> np.ndarray:
    base = ridge_autoregression_forecast(train, horizon)
    weekly = seasonal_weekly_forecast(train, horizon)
    seasonal = seasonal_naive_forecast(train, horizon)
    return 0.45 * base + 0.35 * weekly + 0.20 * seasonal


def seasonal_weekly_forecast(train: pd.DataFrame, horizon: pd.DatetimeIndex) -> np.ndarray:
    series = train.set_index("ds")["y"].sort_index()
    fallback = float(series.tail(min(len(series), 168)).median())
    values = []
    for ts in horizon:
        for weeks_back in (1, 2, 3, 4):
            source_ts = ts - pd.Timedelta(hours=168 * weeks_back)
            if source_ts in series.index:
                values.append(float(series.loc[source_ts]))
                break
        else:
            values.append(fallback)
    return np.asarray(values, dtype=float)


def _tensorflow_lstm_forecast(train: pd.DataFrame, horizon: pd.DatetimeIndex) -> np.ndarray:
    import os

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf

    tf.random.set_seed(42)
    series = train.sort_values("ds")["y"].to_numpy(dtype=np.float32)
    if len(series) < 24 * 60:
        raise ValueError("数据量不足，跳过 LSTM。")

    window = 168
    max_train_points = min(len(series) - window - 1, 3500)
    start = len(series) - window - 1 - max_train_points
    values = series[start:]
    mean = float(values.mean())
    std = float(values.std() or 1.0)
    scaled = (values - mean) / std

    X, y = [], []
    for i in range(len(scaled) - window):
        X.append(scaled[i : i + window])
        y.append(scaled[i + window])
    X_array = np.asarray(X, dtype=np.float32)[..., None]
    y_array = np.asarray(y, dtype=np.float32)

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(window, 1)),
            tf.keras.layers.LSTM(24),
            tf.keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss="mse")
    model.fit(X_array, y_array, epochs=3, batch_size=64, verbose=0)

    history = scaled[-window:].astype(np.float32).tolist()
    predictions = []
    for _ in horizon:
        x_next = np.asarray(history[-window:], dtype=np.float32).reshape(1, window, 1)
        pred_scaled = float(model.predict(x_next, verbose=0)[0, 0])
        history.append(pred_scaled)
        predictions.append(pred_scaled * std + mean)
    return np.asarray(predictions, dtype=float)


def run_backtests(df: pd.DataFrame) -> list[BacktestResult]:
    years = validation_years(df)
    if not years:
        raise ValueError("至少需要两个年份的数据才能回测。")

    results: list[BacktestResult] = []
    prior_actuals: list[np.ndarray] = []
    prior_predictions: list[pd.DataFrame] = []

    for year in years:
        train, valid = train_validate_split(df, year)
        horizon = pd.DatetimeIndex(valid["ds"])
        predictions = forecast_all_models(train, horizon)
        weights = _fit_or_fallback_weights(prior_actuals, prior_predictions, predictions, valid["y"].to_numpy(dtype=float))
        predictions[MODEL_ENSEMBLE] = _weighted_prediction(predictions, weights)

        metrics = _metrics_for_predictions(valid["y"].to_numpy(dtype=float), predictions)
        weight_frame = pd.DataFrame({"model": list(weights.keys()), "weight": list(weights.values())})
        results.append(
            BacktestResult(
                validation_year=year,
                metrics=metrics,
                predictions=valid[["ds", "y"]].merge(predictions, on="ds"),
                ensemble_weights=weight_frame,
            )
        )

        prior_actuals.append(valid["y"].to_numpy(dtype=float))
        prior_predictions.append(predictions[["ds", *BASE_MODEL_NAMES]].copy())

    return results


def forecast_future(df: pd.DataFrame, hours: int = HOURS_PER_YEAR) -> tuple[pd.DataFrame, pd.DataFrame]:
    horizon = make_future_index(df["ds"].max(), hours)
    forecasts = forecast_all_models(df, horizon)

    backtests = run_backtests(df)
    actuals = [result.predictions["y"].to_numpy(dtype=float) for result in backtests]
    preds = [result.predictions[["ds", *BASE_MODEL_NAMES]].copy() for result in backtests]
    weights = _fit_or_fallback_weights(actuals, preds, forecasts, None)
    forecasts[MODEL_ENSEMBLE] = _weighted_prediction(forecasts, weights)
    weights_df = pd.DataFrame({"model": list(weights.keys()), "weight": list(weights.values())})
    return forecasts, weights_df


def save_outputs(df: pd.DataFrame, output_dir: str | Path = "outputs", future_hours: int = HOURS_PER_YEAR) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    backtests = run_backtests(df)
    future, weights = forecast_future(df, future_hours)

    all_metrics = pd.concat(
        [result.metrics.assign(validation_year=result.validation_year) for result in backtests],
        ignore_index=True,
    )
    backtest_predictions = pd.concat(
        [result.predictions.assign(validation_year=result.validation_year) for result in backtests],
        ignore_index=True,
    )

    files = {
        "metrics": output_path / "backtest_metrics.csv",
        "backtest_predictions": output_path / "backtest_predictions.csv",
        "future_forecast": output_path / "future_forecast.csv",
        "ensemble_weights": output_path / "ensemble_weights.csv",
    }
    all_metrics.to_csv(files["metrics"], index=False, encoding="utf-8-sig")
    backtest_predictions.to_csv(files["backtest_predictions"], index=False, encoding="utf-8-sig")
    future.to_csv(files["future_forecast"], index=False, encoding="utf-8-sig")
    weights.to_csv(files["ensemble_weights"], index=False, encoding="utf-8-sig")
    return files


def _calendar_frame(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["month"] = work["ds"].dt.month
    work["dayofweek"] = work["ds"].dt.dayofweek
    work["hour"] = work["ds"].dt.hour
    return work


def _calendar_features(index: pd.DatetimeIndex) -> np.ndarray:
    hour = index.hour.to_numpy(dtype=float)
    dayofweek = index.dayofweek.to_numpy(dtype=float)
    month = index.month.to_numpy(dtype=float)
    dayofyear = index.dayofyear.to_numpy(dtype=float)

    return np.column_stack(
        [
            np.ones(len(index)),
            np.sin(2 * np.pi * hour / 24),
            np.cos(2 * np.pi * hour / 24),
            np.sin(2 * np.pi * dayofweek / 7),
            np.cos(2 * np.pi * dayofweek / 7),
            np.sin(2 * np.pi * month / 12),
            np.cos(2 * np.pi * month / 12),
            np.sin(2 * np.pi * dayofyear / 365.25),
            np.cos(2 * np.pi * dayofyear / 365.25),
        ]
    )


def _estimate_hourly_trend(series: pd.Series) -> float:
    if len(series) < 24 * 14:
        return 0.0
    recent = series.tail(min(len(series), 24 * 56))
    first = float(recent.head(min(len(recent), 24 * 7)).mean())
    last = float(recent.tail(min(len(recent), 24 * 7)).mean())
    return (last - first) / max(len(recent), 1)


def _fit_or_fallback_weights(
    prior_actuals: list[np.ndarray],
    prior_predictions: list[pd.DataFrame],
    current_predictions: pd.DataFrame,
    current_actuals: np.ndarray | None,
) -> dict[str, float]:
    if prior_actuals and prior_predictions:
        y_true = np.concatenate(prior_actuals)
        pred_matrix = np.vstack([frame[BASE_MODEL_NAMES].to_numpy(dtype=float) for frame in prior_predictions])
        return _optimize_weights(y_true, pred_matrix)

    if current_actuals is None:
        equal = 1.0 / len(BASE_MODEL_NAMES)
        return {name: equal for name in BASE_MODEL_NAMES}

    errors = []
    for name in BASE_MODEL_NAMES:
        pred = current_predictions[name].to_numpy(dtype=float)
        errors.append(np.mean(np.abs(current_actuals - pred)) + 1e-9)
    inv = 1 / np.asarray(errors)
    weights = inv / inv.sum()
    return {name: float(weight) for name, weight in zip(BASE_MODEL_NAMES, weights)}


def _optimize_weights(y_true: np.ndarray, pred_matrix: np.ndarray) -> dict[str, float]:
    try:
        from scipy.optimize import minimize

        n_models = pred_matrix.shape[1]

        def objective(weights: np.ndarray) -> float:
            residual = y_true - pred_matrix @ weights
            return float(np.mean(residual**2))

        constraints = {"type": "eq", "fun": lambda weights: np.sum(weights) - 1}
        bounds = [(0.0, 1.0)] * n_models
        start = np.repeat(1.0 / n_models, n_models)
        result = minimize(objective, start, method="SLSQP", bounds=bounds, constraints=constraints)
        weights = result.x if result.success else start
    except Exception:
        weights = np.repeat(1.0 / pred_matrix.shape[1], pred_matrix.shape[1])

    weights = np.clip(weights, 0, 1)
    weights = weights / weights.sum() if weights.sum() else np.repeat(1.0 / len(weights), len(weights))
    return {name: float(weight) for name, weight in zip(BASE_MODEL_NAMES, weights)}


def _weighted_prediction(predictions: pd.DataFrame, weights: dict[str, float]) -> np.ndarray:
    matrix = predictions[BASE_MODEL_NAMES].to_numpy(dtype=float)
    weight_vector = np.asarray([weights[name] for name in BASE_MODEL_NAMES], dtype=float)
    return matrix @ weight_vector


def _metrics_for_predictions(y_true: np.ndarray, predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name in [*BASE_MODEL_NAMES, MODEL_ENSEMBLE]:
        pred = predictions[name].to_numpy(dtype=float)
        rows.append({"model": name, **calculate_metrics(y_true, pred)})
    return pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    error = y_true - y_pred
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error**2)))
    wmape = float(np.sum(np.abs(error)) / max(np.sum(np.abs(y_true)), 1e-9))
    smape = float(np.mean(2 * np.abs(error) / np.maximum(np.abs(y_true) + np.abs(y_pred), 1e-9)))
    denominator = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = float(1 - np.sum(error**2) / denominator) if denominator else np.nan
    return {
        "MAE": mae,
        "RMSE": rmse,
        "WMAPE": wmape,
        "sMAPE": smape,
        "R2": r2,
    }


def format_metric_table(metrics: pd.DataFrame) -> pd.DataFrame:
    table = metrics.copy()
    for column in ["MAE", "RMSE"]:
        table[column] = table[column].map(lambda value: round(value, 3))
    for column in ["WMAPE", "sMAPE"]:
        table[column] = table[column].map(lambda value: round(value * 100, 2))
    table["R2"] = table["R2"].map(lambda value: round(value, 4))
    return table
