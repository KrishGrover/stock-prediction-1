import streamlit as st
import yfinance as yf
import pandas as pd

from ta.momentum import RSIIndicator
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.title("SBIN.NS 30-Day Stock Movement Predictor")

ticker = "SBIN.NS"

st.write(f"Analyzing: {ticker}")

if st.button("Predict"):

    data = yf.download(
    ticker,
    period="10y",
    auto_adjust=True,
    multi_level_index=False
)

    # Make sure Close is a Series
    data["Close"] = data["Close"].squeeze()

# Features
    data["Return"] = data["Close"].pct_change()

    data["MA10"] = data["Close"].rolling(10).mean()

    data["MA50"] = data["Close"].rolling(50).mean()

# RSI
    data["RSI"] = RSIIndicator(
    close=data["Close"],
    window=14
    ).rsi()

    data["Target"] = (
        data["Close"].shift(-30)
        > data["Close"]
    ).astype(int)

    data.dropna(inplace=True)

    X = data[
        ["Return", "MA10", "MA50", "RSI"]
    ]

    y = data["Target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        shuffle=False
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        random_state=42
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    accuracy = accuracy_score(y_test, preds)

    st.subheader("Model Accuracy")

    st.write(f"{accuracy*100:.2f}%")

    latest = X.iloc[-1:]

    prediction = model.predict(latest)[0]

    probability = model.predict_proba(latest)[0]

    st.subheader("Prediction for Next 30 Days")

    if prediction == 1:

        st.success(
            "📈 SBIN.NS is predicted to go UP in the next 30 days"
        )

        st.write(
            f"Confidence: {probability[1]*100:.2f}%"
        )

    else:

        st.error(
            "SBIN.NS is predicted to go DOWN in the next 30 days"
        )

        st.write(
            f"Confidence: {probability[0]*100:.2f}%"
        )

    st.subheader("Latest Closing Price")

    st.write(f"₹ {data['Close'].iloc[-1]:.2f}")

    st.subheader("SBIN.NS Closing Price History")

    st.line_chart(data["Close"])