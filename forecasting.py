# forecasting.py
import os
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataLoader import load_data
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from visualization import (
    plot_forecast,
    plot_mape_comparison,
    plot_trending_products,
    explain_model_simplified,
    plot_forecast_summary)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# Metrics

def _safe_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    y_true_arr, y_pred_arr = np.array(y_true), np.array(y_pred)
    non_zero = y_true_arr != 0
    if np.any(non_zero):
        mape = np.mean(np.abs((y_true_arr[non_zero] - y_pred_arr[non_zero]) / y_true_arr[non_zero])) * 100
    else:
        mape = np.nan
    return {"MAE": round(mae, 2), "RMSE": round(rmse, 2), "MAPE": round(mape, 2) if not np.isnan(mape) else None}



# Forecasting

def forecast_sales(series, days=30, test_days=30, explain=True):
    
    if not isinstance(series, pd.Series):
        raise ValueError("Input must be a pandas Series with datetime index.")
    if len(series.dropna()) < (test_days + 30):
        raise ValueError("Not enough data for forecasting.")

    series = series.asfreq("D").fillna(0)
    train, test = series.iloc[:-test_days], series.iloc[-test_days:]

    seasonal_periods = max(2, min(365, max(7, int(len(train) // 30))))
    try:
        hw_model = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=seasonal_periods).fit()
    except Exception:
        hw_model = ExponentialSmoothing(train, trend="add", seasonal=None).fit()
    hw_forecast = pd.Series(hw_model.forecast(len(test)), index=test.index)

    # Create lag features for ML models
    df_lags = pd.DataFrame({"Sales": series})
    for lag in [1, 2, 3, 7, 14, 30]:
        df_lags[f"lag{lag}"] = df_lags["Sales"].shift(lag)
    df_lags.dropna(inplace=True)

    X = df_lags.drop(columns=["Sales"])
    y = df_lags["Sales"]
    X_train, X_test = X.iloc[:-test_days], X.iloc[-test_days:]
    y_train, y_test = y.iloc[:-test_days], y.iloc[-test_days:]

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

    # Train model
    lr = LinearRegression().fit(X_train_scaled, y_train)
    lr_pred = pd.Series(lr.predict(X_test_scaled), index=y_test.index)

    tree = DecisionTreeRegressor(max_depth=None, min_samples_split=5, random_state=42)
    tree.fit(X_train, y_train)
    tree_pred = pd.Series(tree.predict(X_test), index=y_test.index)

    results = {
        "Holt-Winters": _safe_metrics(y_test, hw_forecast),
        "Linear Regression": _safe_metrics(y_test, lr_pred),
        "Decision Tree": _safe_metrics(y_test, tree_pred),
    }

    os.makedirs("static", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    # Plot 
    plot_forecast(test,lr_pred)
    plot_forecast_summary(train, test, hw_forecast, lr_pred, tree_pred)
    plot_mape_comparison(results)

    # Explainability 
    if explain:
        explain_model_simplified(lr, X_train_scaled, "LinearRegression", "static/explain")
        explain_model_simplified(tree, X_train, "DecisionTree", "static/explain")

    # Save future forecast
    future_raw = hw_model.forecast(days)
    future_df = pd.DataFrame({
        "Date": pd.date_range(start=series.index.max() + pd.Timedelta(days=1), periods=days),
        "Predicted_Sales": np.round(future_raw, 2)
    })
    future_df.to_csv("data/future_forecast.csv", index=False)

    return results, future_df


# Trending Products Forecasting

def forecast_trending_products(top_n=10, min_days=30, forecast_horizon=7):

    _, df = load_data()

    if not all(col in df.columns for col in ["Date", "Product", "Total_Cost"]):
        raise KeyError("DataFrame must contain 'Date', 'Product', 'Total_Cost' columns")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    trends = {}

    for product, group in df.groupby("Product"):
        daily = group.groupby(group["Date"].dt.date)["Total_Cost"].sum()
        daily.index = pd.to_datetime(daily.index)
        if len(daily) < min_days:
            continue
        try:
            model = ExponentialSmoothing(daily, trend="add", seasonal=None).fit(optimized=True)
            future = model.forecast(forecast_horizon)
            growth = float(np.mean(future) - np.mean(daily[-forecast_horizon:]))
            trends[product] = growth
        except Exception:
            continue

    trending_products = [p for p, g in sorted(trends.items(), key=lambda x: x[1], reverse=True) if g > 0]
    if df.empty:
        raise ValueError("Input data is empty — please check your dataset.")

    os.makedirs("data", exist_ok=True)
    pd.DataFrame(trending_products, columns=["Product"]).to_csv("data/trending_products.csv", index=False)

    plot_trending_products(trending_products)

    return trending_products




if __name__ == "__main__":
    daily_sales, df = load_data()
    results, future = forecast_sales(daily_sales, days=10, test_days=30)
    print("\nForecasting Results:\n", results)
    trending = forecast_trending_products(df)
    print("\nTrending Products:\n", trending)
