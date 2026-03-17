import sys
import os
sys.path.append(os.path.abspath("../src"))

import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

# ---------------- IMPORT FROM SRC ----------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_PATH = os.path.join(BASE_DIR, "src")

if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

print("SRC PATH ADDED:", SRC_PATH)
from stock_prediction_project import predict, backtest


# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Stock Trading System",
    page_icon="📈",
    layout="wide"
)

# ---------------- TITLE ----------------
st.title("🤖 AI Stock Trading Dashboard")
st.write("Real-Time AI Stock Prediction System")

# ---------------- SIDEBAR ----------------
st.sidebar.header("Settings")

symbol = st.sidebar.text_input("Stock Symbol", "AAPL")

period = st.sidebar.selectbox(
    "Select Time Period",
    ["6mo", "1y", "2y", "5y"]
)

# ---------------- DATA ----------------
stock = yf.download(symbol, period=period)

# ---------------- ADD INDICATORS ----------------
stock= add_indicators(stock)

# ---------------- METRICS ----------------
current_price = float(stock["Close"].iloc[-1])
previous_price = float(stock["Close"].iloc[-2])

change = current_price - previous_price
percent_change = (change / previous_price) * 100

col1, col2, col3 = st.columns(3)

col1.metric("Current Price", f"${current_price:.2f}")
col2.metric("Daily Change", f"{change:.2f}", f"{percent_change:.2f}%")
col3.metric("Volume", int(data["Volume"].iloc[-1]))

# ---------------- CHART ----------------
st.subheader("Stock Price Chart")

chart_data = stock[["Close", "MA50", "MA200"]]
st.line_chart(chart_data)

# ---------------- CANDLESTICK ----------------
fig = go.Figure()

fig.add_trace(go.Candlestick(
    x=data.index,
    open=stock["Open"],
    high=stock["High"],
    low=stock["Low"],
    close=stock["Close"],
    name="Candlestick"
))

fig.add_trace(go.Scatter(x=stock.index, y=stock["MA50"], name="MA50"))
fig.add_trace(go.Scatter(x=stock.index, y=stock["MA200"], name="MA200"))

st.plotly_chart(fig, use_container_width=True)

# ---------------- AI PREDICTION ----------------
st.subheader("AI Prediction")

if st.button("Run AI Prediction", key="predict_btn"):

    current, predicted, signal, _ = predict(symbol)

    c1, c2 = st.columns(2)
    c1.metric("Current Price", f"${current:.2f}")
    c2.metric("Predicted Price", f"${predicted:.2f}")

    if signal == "BUY":
        st.success("🟢 AI SIGNAL: BUY")
    else:
        st.error("🔴 AI SIGNAL: SELL")

# ---------------- INDICATORS ----------------
st.subheader("RSI Indicator")
st.line_chart(stock["RSI"])

st.subheader("MACD")
st.line_chart(stock[["MACD", "Signal_Line"]])

# ---------------- BACKTEST ----------------
returns = backtest(stock)

st.subheader("Strategy Performance")
st.line_chart(returns)

# ---------------- SENTIMENT ----------------
sentiment_score = get_news_sentiment(symbol)

st.subheader("📰 News Sentiment")

st.metric("Sentiment Score", round(sentiment_score, 3))

if sentiment_score > 0:
    st.success("Market Sentiment: Positive")
else:
    st.error("Market Sentiment: Negative")

# ---------------- RAW DATA ----------------
with st.expander("View Raw Data"):
    st.dataframe(data.tail())
