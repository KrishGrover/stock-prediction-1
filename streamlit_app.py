import streamlit as st
import yfinance as yf
import pandas as pd

from ta.momentum import RSIIndicator
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.title("📈 Indian Banking Stock 30-Day Predictor")

# Dropdown for banks
bank_stocks = {
    "State Bank of India": "SBIN.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "Axis Bank": "AXISBANK.NS",
    "Kotak Mahindra Bank": "KOTAKBANK.NS",
    "Punjab National Bank": "PNB.NS",
    "Bank of Baroda": "BANKBARODA.NS",
    "Canara Bank": "CANBK.NS",
    "Union Bank": "UNIONBANK.NS"
}

# Timeline dropdown
timeline_options = {
    "1 Month":"1mo",
    "3 Months":"3mo",
    "6 Months":"6mo",
    "1 Year":"1y",
    "2 Years":"2y",
    "5 Years":"5y",
    "10 Years":"10y"
}


# Display side by side
col1,col2 = st.columns(2)

with col1:

    selected_bank = st.selectbox(
        "Select Bank",
        list(bank_stocks.keys())
    )

with col2:

    selected_timeline = st.selectbox(
        "Select Timeline",
        list(timeline_options.keys()),
        index=6
    )


ticker = bank_stocks[selected_bank]

period = timeline_options[selected_timeline]

st.write(f"Analyzing: {ticker}")

if st.button("Predict"):

    data = yf.download(
        ticker,
        period=period,
        auto_adjust=True,
        multi_level_index=False
    )

    # Convert Close to Series
    data["Close"] = data["Close"].squeeze()

    # Features

    data["Return"] = data["Close"].pct_change()

    data["MA10"] = data["Close"].rolling(10).mean()

    data["MA50"] = data["Close"].rolling(50).mean()

    data["RSI"] = RSIIndicator(
        close=data["Close"],
        window=14
    ).rsi()

    # Target

    data["Target"] = (
        data["Close"].shift(-30)
        >
        data["Close"]
    ).astype(int)

    data.dropna(inplace=True)

    X = data[
        [
            "Return",
            "MA10",
            "MA50",
            "RSI"
        ]
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

    model.fit(X_train,y_train)

    preds = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        preds
    )

    latest = X.iloc[-1:]

    prediction = model.predict(latest)[0]

    probability = model.predict_proba(latest)[0]


    # Metrics

    col1,col2,col3 = st.columns(3)

    with col1:

        st.metric(
            "Current Price",
            f"₹ {data['Close'].iloc[-1]:.2f}"
        )

    with col2:

        st.metric(
            "Accuracy",
            f"{accuracy*100:.2f}%"
        )

    with col3:

        st.metric(
            "Prediction",
            "📈 UP" if prediction==1
            else "📉 DOWN"
        )


    st.write(
        f"Confidence : "
        f"{max(probability)*100:.2f}%"
    )


    if prediction==1:

        st.success(
            f"{ticker} is predicted "
            f"to go UP in next 30 days"
        )

    else:

        st.error(
            f"{ticker} is predicted "
            f"to go DOWN in next 30 days"
        )


    st.subheader("Closing Price History")

    st.line_chart(
        data["Close"]
    )


    st.subheader("Recent Data")

    st.dataframe(
        data[
            [
                "Close",
                "Return",
                "MA10",
                "MA50",
                "RSI"
            ]
        ].tail(10)
    )