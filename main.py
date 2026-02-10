# main.py - Run this in VSCode

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import numpy as np
import requests
from io import StringIO
import uvicorn

app = FastAPI()

# CORS - CRITICAL for Flutter to connect python -m pip install fastapi uvicorn numpy pandas scikit-learn and 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and scaler
model = joblib.load("btc_model.pkl")
scaler = joblib.load("btc_scaler.pkl")

def get_features():
    url = "https://www.cryptodatadownload.com/cdd/Binance_BTCUSDT_d.csv"
    r = requests.get(url)
    df = pd.read_csv(StringIO(r.text), skiprows=1)

    df = df.rename(columns={
        "date": "Date",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "Volume USDT": "Volume"
    })

    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
    df["Date"] = pd.to_datetime(df["Date"])

    df['open-close'] = df['Open'] - df['Close']
    df['low-high'] = df['Low'] - df['High']
    df['day'] = df['Date'].dt.day
    df['month'] = df['Date'].dt.month
    df['year'] = df['Date'].dt.year
    df['is_quarter_end'] = np.where(df['month'] % 3 == 0, 1, 0)

    last = df.iloc[-1]
    features = np.array([[last['open-close'], last['low-high'], last['is_quarter_end']]])
    features_scaled = scaler.transform(features)
    return features_scaled

@app.get("/")
def root():
    return {"status": "AI Backend Running ✅", "endpoint": "/predict"}

@app.get("/predict")
def predict():
    try:
        features = get_features()
        pred = model.predict(features)[0]
        prob = model.predict_proba(features)[0]

        direction = "UP" if pred == 1 else "DOWN"
        confidence = float(max(prob))

        print(f"✅ Prediction: {direction} ({confidence*100:.1f}%)")  # Debug log

        return {"direction": direction, "confidence": confidence}
    except Exception as e:
        print(f"❌ Error: {e}")  # Debug log
        return {"direction": "ERROR", "confidence": 0.0, "error": str(e)}

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 STARTING AI BACKEND SERVER...")
    print("=" * 60)
    print("📍 Server will run on: http://0.0.0.0:8000")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8000)