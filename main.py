from fastapi import FastAPI
import joblib
import pandas as pd
import numpy as np

app = FastAPI()

# Load model + data
model = joblib.load("mcr_model.joblib")
le = joblib.load("region_encoder.joblib")
df = pd.read_csv("processed_data.csv")

df['Date'] = pd.to_datetime(df['Date'])

FEATURES = [
    'region_enc', 'year', 'month', 'quarter', 'time_idx',
    'SalesVolume', 'SemiDetachedPrice', 'TerracedPrice',
    'FlatPrice', 'FTBPrice', 'NewPrice', 'OldPrice',
    'NewSalesVolume', 'OldSalesVolume',
    'price_lag_1', 'price_lag_3', 'price_lag_6', 'price_lag_12',
    'rolling_avg_12', 'yoy_change'
]

@app.get("/")
def home():
    return {"message": "API is running"}

@app.post("/predict")
def predict(data: dict):
    region = data["region"]
    date = pd.to_datetime(data["date"])

    region_df = df[df["RegionName"] == region].sort_values("Date")

    if region_df.empty:
        return {"error": "Region not found"}

    latest = region_df[region_df["Date"] < date].tail(1)

    if latest.empty:
        return {"error": "Not enough history"}

    latest = latest.copy()

    # Update time features
    latest["year"] = date.year
    latest["month"] = date.month
    latest["quarter"] = (date.month - 1)//3 + 1
    latest["time_idx"] = (date.year - 1995) * 12 + date.month

    # Encode region
    latest["region_enc"] = le.transform([region])[0]

    X = latest[FEATURES]

    prediction = model.predict(X)[0]

    return {
        "region": region,
        "date": str(date.date()),
        "predicted_price": float(prediction)
    }
