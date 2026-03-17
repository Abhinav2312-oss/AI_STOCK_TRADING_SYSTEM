
import sys
import os
import numpy as np
import streamlit as st
import yfinance as yf
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# ---------------- PATH SETUP ----------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_PATH = os.path.join(BASE_DIR, "src")
MODELS_PATH = os.path.join(BASE_DIR, "models")

if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from stock_prediction_project import predict, backtest

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Stock Trading System",
    page_icon="📈",
    layout="wide"
)

st.title("🤖 AI Stock Trading Dashboard")
st.write("Real-Time AI Stock Prediction System")


# ---------------- SIDEBAR ----------------
st.sidebar.header("Configuration")

ticker = st.sidebar.text_input("Enter Stock Ticker", value="AAPL").upper()
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2015-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))

# ---------------- LOAD MODELS (LOAD ONCE) ----------------
@st.cache_resource
def load_ai_models():
    scaler = joblib.load(os.path.join(MODELS_PATH, "scaler.pkl"))

    # ✅ Load LSTM model (Keras)
    lstm_model = load_model(
        os.path.join(MODELS_PATH, "lstm_model.keras")
    )

    return scaler, lstm_model


# ---------------- FETCH DATA ----------------
st.write(f"### Data for {ticker}")

stock = yf.download(ticker, start=start_date, end=end_date)

if not stock.empty:

    st.dataframe(stock.tail(10))

    # -------- Price Plot --------
    st.subheader("Price Trend")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(stock["Close"], label="Close Price")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()

    st.pyplot(fig)

    # -------- Prediction --------
    if st.sidebar.button("Predict Next Day"):

        try:
            scaler, model = load_ai_models()

            # last 60 days input
            last_60_days = stock["Close"].values[-60:].reshape(-1, 1)

            scaled_data = scaler.transform(last_60_days)

            X_input = np.array([scaled_data])
            X_input = X_input.reshape(
                (X_input.shape[0], X_input.shape[1], 1)
            )

            # prediction
            prediction_scaled = model.predict(X_input)
            prediction = scaler.inverse_transform(prediction_scaled)

            st.sidebar.success(
                f"Predicted Price: ${prediction[0][0]:.2f}"
            )

        except FileNotFoundError:
            st.sidebar.error("Model files not found in models folder.")

        except Exception as e:
            st.sidebar.error(f"Error: {e}")

    # -------- Statistics --------
    st.subheader("Summary Statistics")
    st.write(stock.describe())

else:
    st.error("Invalid ticker or no data found.")
