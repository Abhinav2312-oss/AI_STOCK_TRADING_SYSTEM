import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model


# ---------------- INDICATORS ----------------
def add_indicators(stock):

    stock= stock.copy()

    stock["MA50"] = stock["Close"].rolling(50).mean()
    stock["MA200"] = stock["Close"].rolling(200).mean()

    return stock


# ---------------- BACKTEST ----------------
def backtest(stock):

    stock= stock.copy()

    stock["Signal"] = 0
    stock.loc[stock["MA50"] > stock["MA200"], "Signal"] = 1
    stock.loc[stock["MA50"] < stock["MA200"], "Signal"] = -1

    stock["Returns"] = stock["Close"].pct_change()
    stock["Strategy"] = stock["Signal"].shift(1) * stock["Returns"]

    cumulative_return = (1 + stock["Strategy"]).cumprod()

    return cumulative_return


# ---------------- PREDICTION ----------------
def predict(symbol):

    stock= yf.download(symbol, period="1y")

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(stock[["Close"]])

    last_60 = scaled[-60:]
    last_60 = last_60.reshape(1, 60, 1)

    # load saved model
    model = load_model("../models/lstm_model.h5")

    pred = model.predict(last_60)
    predicted_price = scaler.inverse_transform(pred)

    return float(predicted_price[0][0])




