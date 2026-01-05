# forecasting/data_loader.py
import os
import pandas as pd

def load_data(path="data/Retail_Transactions_Dataset.csv"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")

    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df.get("Date"), errors="coerce")
    df["Total_Cost"] = pd.to_numeric(df.get("Total_Cost", 0), errors="coerce").fillna(0)
    df.dropna(subset=["Date"], inplace=True)
    df.sort_values("Date", inplace=True)

    if "Product" in df.columns:
        df["Product"] = df["Product"].astype(str).str.split(",")
        df = df.explode("Product")
        df["Product"] = (
            df["Product"]
            .astype(str)
            .str.replace(r"[\[\]\'\"]", "", regex=True)
            .str.strip()
        )

    daily_sales = df.groupby(df["Date"].dt.date)["Total_Cost"].sum()
    daily_sales.index = pd.to_datetime(daily_sales.index)
    daily_sales = daily_sales.asfreq("D").fillna(0)

    return daily_sales, df
