# app.py

import streamlit as st
import pandas as pd
import datetime
from dateutil.relativedelta import relativedelta
from data_fetch import fetch_data
from technical_indicators import add_all_indicators
from strategies import intraday_strategy, swing_strategy
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np
import plotly.graph_objects as go
import requests
from bs4 import BeautifulSoup

from prophet import Prophet
from prophet.plot import plot_plotly

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from sentiment import (
    fetch_company_name,
    fetch_news_headlines, 
    perform_sentiment_analysis, 
    train_sentiment_model, 
    predict_future_sentiment, 
    plot_sentiment
)

#########################
# Custom Lexicon Augmentation
#########################
positive_words = 'high profit Growth Potential Opportunity Bullish Strong Valuable Success Promising Profitable Win Winner Outstanding Record Earnings Breakthrough buy bull long support undervalued underpriced cheap upward rising trend moon rocket hold breakout call beat support buying holding'
negative_words = 'resistance squeeze cover seller Risk Loss Decline Bearish Weak Declining Uncertain Troubling Downturn Struggle Unstable Volatile Slump Disaster Plunge sell bear bubble bearish short overvalued overbought overpriced expensive downward falling sold sell low put miss'

pos_list = positive_words.lower().split()
neg_list = negative_words.lower().split()

sia = SentimentIntensityAnalyzer()
# Update VADER lexicon
for word in pos_list:
    sia.lexicon[word] = 2.0  # Increase sentiment score
for word in neg_list:
    sia.lexicon[word] = -2.0 # Decrease sentiment score


#########################
# Utility / Helper Functions
#########################

def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def rsi(series, period=14):
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = pd.Series(gain).ewm(alpha=1/period, adjust=False).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi_vals = 100 - (100 / (1 + rs))
    return pd.Series(rsi_vals, index=series.index)

def apply_pine_strategy_logic(data: pd.DataFrame) -> pd.DataFrame:
    data = data.sort_values(by='datetime').copy()
    data['EMA20'] = ema(data['Close'], 20)
    data['EMA50'] = ema(data['Close'], 50)
    data['EMA35'] = ema(data['Close'], 35)
    data['RSI'] = rsi(data['Close'], 14)

    # Pine conditions (example from before)
    # Adjust RSI exit thresholds if needed or use as is
    # EnterLong = EMA20 cross above EMA50
    data['EnterLong'] = (data['EMA20'] > data['EMA50']) & (data['EMA20'].shift(1) <= data['EMA50'].shift(1))
    # ExitLong = close crosses below EMA35 and RSI > 80
    data['ExitLong'] = (data['Close'] < data['EMA35']) & (data['Close'].shift(1) >= data['EMA35'].shift(1)) & (data['RSI'] > 80)

    # EnterShort = EMA50 cross above EMA20
    data['EnterShort'] = (data['EMA50'] > data['EMA20']) & (data['EMA50'].shift(1) <= data['EMA20'].shift(1))
    # ExitShort = RSI < 20
    data['ExitShort'] = (data['RSI'] < 20)

    # Time filter
    start = pd.Timestamp('2007-01-01')
    end = pd.Timestamp('2021-06-01')
    in_range = (data['datetime'] >= start) & (data['datetime'] <= end)

    data['signal'] = 0
    data.loc[in_range & data['EnterLong'], 'signal'] = 1
    data.loc[in_range & data['EnterShort'], 'signal'] = -1
    # Exits are just conditions, actual trade close logic implemented in evaluate_strategy
    return data

def prepare_features_for_model(data: pd.DataFrame, horizon_days: int) -> (pd.DataFrame, pd.Series):
    df = data.copy()
    df = df.sort_values("datetime")
    df.set_index("datetime", inplace=True)

    if len(df) < horizon_days:
        return pd.DataFrame(), pd.Series()

    df["target"] = df["Close"].shift(-horizon_days)
    df.dropna(inplace=True)

    feature_cols = [col for col in df.columns if col not in ["target"]]
    numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

    X = df[numeric_cols]
    y = df["target"]

    return X, y

def train_and_predict_future(data: pd.DataFrame, horizon_str: str, model_type: str):
    horizon_days = get_future_horizon_days(horizon_str)
    X, y = prepare_features_for_model(data, horizon_days)
    if X.empty or y.empty:
        return None, None, None, None, f"Not enough data to train the model for {horizon_str} prediction."

    X_train, X_test = X.iloc[:-1], X.iloc[-1:]
    y_train, y_test = y.iloc[:-1], y.iloc[-1:]

    if model_type == 'intraday':
        model = LinearRegression()
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42)

    model.fit(X_train, y_train)
    future_pred = model.predict(X_test)[0]

    # Calculate R² as accuracy metric (on train set)
    train_preds = model.predict(X_train)
    r2 = r2_score(y_train, train_preds)

    last_date = data["datetime"].max()
    predict_date = last_date + datetime.timedelta(days=horizon_days)

    # Residual std for band
    residuals = y_train - train_preds
    residual_std = np.std(residuals) if len(residuals) > 1 else 0

    return future_pred, predict_date, residual_std, r2, None

def get_future_horizon_days(future_horizon_str):
    if future_horizon_str == "3 Months":
        return 90
    elif future_horizon_str == "6 Months":
        return 180
    elif future_horizon_str == "1 Year":
        return 365
    elif future_horizon_str == "2 Years":
        return 730
    elif future_horizon_str == "3 Years":
        return 1095
    elif future_horizon_str == "5 Years":
        return 1825
    else:
        return 365

def evaluate_strategy(data: pd.DataFrame, initial_capital=10000):
    # Very simplistic backtest:
    # Whenever signal=1: go long at Close price. 
    # Close this position on next time we have either a signal=-1 or at the end.
    # Similarly for short.
    # We'll compute net profit, percent profitable, and profit factor from these trades.
    # This is a simplistic placeholder.

    positions = []
    balance = initial_capital
    trade_results = []

    current_position = None  # ('long' or 'short', entry_price)
    for i, row in data.iterrows():
        sig = row.get('signal', 0)
        price = row['Close']

        if current_position is None:
            # If we get a signal=1, enter long
            if sig == 1:
                current_position = ('long', price)
            elif sig == -1:
                current_position = ('short', price)
        else:
            pos_type, entry_price = current_position
            # Check if we should exit
            # Exit conditions from pine logic:
            # If we are long and 'ExitLong' is True
            if pos_type == 'long' and row.get('ExitLong', False):
                profit = (price - entry_price)
                trade_results.append(profit)
                current_position = None
            # If we are short and 'ExitShort' is True
            elif pos_type == 'short' and row.get('ExitShort', False):
                # Profit from short = entry_price - exit_price
                profit = (entry_price - price)
                trade_results.append(profit)
                current_position = None
            # If we see a reverse signal, we exit and enter opposite
            elif pos_type == 'long' and sig == -1:
                # Exit long
                profit = (price - entry_price)
                trade_results.append(profit)
                # Enter short
                current_position = ('short', price)
            elif pos_type == 'short' and sig == 1:
                # Exit short
                profit = (entry_price - price)
                trade_results.append(profit)
                # Enter long
                current_position = ('long', price)

    # If still in position at the end, close it at last price
    if current_position is not None:
        pos_type, entry_price = current_position
        last_price = data.iloc[-1]['Close']
        if pos_type == 'long':
            trade_results.append(last_price - entry_price)
        else:
            trade_results.append(entry_price - last_price)
        current_position = None

    if len(trade_results) == 0:
        return 0, 0, 0  # No trades made

    net_profit = sum(trade_results)
    winners = [tr for tr in trade_results if tr > 0]
    losers = [tr for tr in trade_results if tr < 0]
    percent_profitable = (len(winners) / len(trade_results)) * 100 if len(trade_results) > 0 else 0
    profit_factor = (sum(winners) / abs(sum(losers))) if len(losers) > 0 else np.inf

    return net_profit, percent_profitable, profit_factor

def plot_all_in_one(data):
    fig = go.Figure()

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=data['datetime'],
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name='Price'
    ))

    # EMA
    if 'EMA20' in data.columns:
        fig.add_trace(go.Scatter(x=data['datetime'], y=data['EMA20'], line=dict(color='red', width=1), name='EMA20'))
    if 'EMA50' in data.columns:
        fig.add_trace(go.Scatter(x=data['datetime'], y=data['EMA50'], line=dict(color='blue', width=1), name='EMA50'))
    if 'EMA35' in data.columns:
        fig.add_trace(go.Scatter(x=data['datetime'], y=data['EMA35'], line=dict(color='green', width=1), name='EMA35'))

    # RSI
    if 'RSI' in data.columns:
        fig.add_trace(go.Scatter(x=data['datetime'], y=data['RSI'], line=dict(color='yellow', width=1, dash='dot'), name='RSI'))

    # Signals
    if 'signal' in data.columns:
        buy_points = data[data['signal'] == 1]
        sell_points = data[data['signal'] == -1]
        if not buy_points.empty:
            fig.add_trace(go.Scatter(
                x=buy_points['datetime'],
                y=buy_points['Close'],
                mode='markers',
                marker=dict(symbol='triangle-up', color='green', size=10),
                name='Buy Signal'
            ))
        if not sell_points.empty:
            fig.add_trace(go.Scatter(
                x=sell_points['datetime'],
                y=sell_points['Close'],
                mode='markers',
                marker=dict(symbol='triangle-down', color='red', size=10),
                name='Short Signal'
            ))

    fig.update_layout(
        title="All-in-One Indicators Plot",
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_dark",
        xaxis_rangeslider_visible=False
    )

    return fig

def prepare_prophet_data(data: pd.DataFrame) -> pd.DataFrame:
    df = data[['datetime', 'Close']].rename(columns={'datetime': 'ds', 'Close': 'y'})
    return df

def forecast_with_prophet(data: pd.DataFrame, periods=180, freq='D'):
    df = prepare_prophet_data(data)
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=periods, freq=freq)
    forecast = model.predict(future)
    return forecast, model

def plot_prediction_line(data, predict_date, future_pred, residual_std):
    fig = go.Figure()

    # Plot historical candlestick data
    fig.add_trace(go.Candlestick(
        x=data['datetime'],
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name='Historical Price'
    ))

    # Plot Bollinger Bands if present
    if {'BB_Upper', 'BB_Middle', 'BB_Lower'}.issubset(data.columns):
        fig.add_trace(go.Scatter(
            x=data['datetime'],
            y=data['BB_Upper'],
            line=dict(color='rgba(173,216,230,0.5)', width=1),
            name='BB Upper'
        ))
        fig.add_trace(go.Scatter(
            x=data['datetime'],
            y=data['BB_Middle'],
            line=dict(color='blue', width=1),
            name='BB Middle'
        ))
        fig.add_trace(go.Scatter(
            x=data['datetime'],
            y=data['BB_Lower'],
            line=dict(color='rgba(173,216,230,0.5)', width=1),
            name='BB Lower'
        ))

    # Plot Moving Averages if present
    if 'MA_Short' in data.columns:
        fig.add_trace(go.Scatter(
            x=data['datetime'],
            y=data['MA_Short'],
            line=dict(color='orange', width=1),
            name='Short MA'
        ))

    if 'MA_Long' in data.columns:
        fig.add_trace(go.Scatter(
            x=data['datetime'],
            y=data['MA_Long'],
            line=dict(color='purple', width=1),
            name='Long MA'
        ))

    # Plot MACD if present
    # MACD is often plotted on a separate subplot, but here we put everything together.
    if {'MACD', 'MACD_Signal', 'MACD_Hist'}.issubset(data.columns):
        fig.add_trace(go.Scatter(
            x=data['datetime'],
            y=data['MACD'],
            line=dict(color='cyan', width=1),
            name='MACD'
        ))
        fig.add_trace(go.Scatter(
            x=data['datetime'],
            y=data['MACD_Signal'],
            line=dict(color='magenta', width=1),
            name='MACD_Signal'
        ))
        fig.add_trace(go.Bar(
            x=data['datetime'],
            y=data['MACD_Hist'],
            marker_color='white',
            name='MACD_Hist'
        ))

    # Identify last date and price
    last_date = data['datetime'].max()
    last_close = data.loc[data['datetime'] == last_date, 'Close'].values[0]

    # Future prediction line
    future_line_x = [last_date, predict_date]
    future_line_y = [last_close, future_pred]

    # Upper and lower band for prediction (2 std dev)
    upper_line_y = [last_close + 2 * residual_std, future_pred + 2 * residual_std]
    lower_line_y = [last_close - 2 * residual_std, future_pred - 2 * residual_std]

    # Add the band as a filled area
    fig.add_trace(go.Scatter(
        x=future_line_x + future_line_x[::-1],
        y=upper_line_y + lower_line_y[::-1],
        fill='toself',
        fillcolor='rgba(255,215,0,0.2)',
        line=dict(color='rgba(255,215,0,0)'),
        name='Prediction Band'
    ))

    # Add the future prediction center line
    fig.add_trace(go.Scatter(
        x=future_line_x,
        y=future_line_y,
        line=dict(color='gold', width=2, dash='dash'),
        name='Future Prediction'
    ))

    fig.update_layout(
        title="Historical Price + Indicators with Future Prediction",
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_dark",
        xaxis_rangeslider_visible=False
    )

    return fig

def plot_prophet_forecast(data: pd.DataFrame, forecast: pd.DataFrame):
    fig = go.Figure()
    # Historical
    fig.add_trace(go.Scatter(
        x=data['datetime'], y=data['Close'],
        mode='lines', name='Historical', line=dict(color='white')
    ))
    # Forecast line
    fig.add_trace(go.Scatter(
        x=forecast['ds'], y=forecast['yhat'],
        mode='lines', line=dict(color='gold', width=2, dash='dash'),
        name='Forecast'
    ))
    # Confidence intervals
    fig.add_trace(go.Scatter(
        x=pd.concat([forecast['ds'], forecast['ds'][::-1]]),
        y=pd.concat([forecast['yhat_upper'], forecast['yhat_lower'][::-1]]),
        fill='toself', fillcolor='rgba(255,215,0,0.2)',
        line=dict(color='rgba(255,215,0,0)'),
        name='Confidence Interval'
    ))

    fig.update_layout(
        title="Historical Data with Prophet Forecast",
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_dark",
        xaxis_rangeslider_visible=False
    )
    return fig

def main():
    st.set_page_config(page_title="Indian Stock Market Analysis", layout="wide")
    st.title("Indian Stock Market Analysis")

    # Sidebar
    st.sidebar.header("Settings")
    data_source = st.sidebar.selectbox("Data Source", ["yfinance", "alpha_vantage"])
    symbol = st.sidebar.text_input("Stock Symbol (e.g. RELIANCE)", value="RELIANCE")
    exchange = st.sidebar.selectbox("Exchange", ["NSE", "BSE"])
    strategy_type = st.sidebar.selectbox("Strategy Type", ["Intraday", "Swing"])

    future_horizon_options = ["3 Months", "6 Months", "1 Year", "2 Years", "3 Years", "5 Years"]
    future_horizon = st.sidebar.selectbox("Prediction Horizon", future_horizon_options)

    # Additional dropdown for Prophet forecast horizon
    prophet_future_options = ["3 Months", "6 Months", "1 Year"]
    prophet_horizon = st.sidebar.selectbox("Prophet Forecast Horizon", prophet_future_options)

    today = datetime.date.today()
    start_date = today - relativedelta(years=10)
    end_date = today

    if strategy_type == "Intraday":
        chosen_date = end_date
        start = datetime.datetime.combine(chosen_date, datetime.time(9, 15))
        end = datetime.datetime.combine(chosen_date, datetime.time(15, 30))
        interval = "15m"
        model_type = 'intraday'
    else:
        start = datetime.datetime.combine(start_date, datetime.time(0, 0))
        end = datetime.datetime.combine(end_date, datetime.time(23, 59))
        interval = "1d"
        model_type = 'swing'

    if st.sidebar.button("Fetch Data and Analyze"):
        with st.spinner("Fetching data..."):
            data = fetch_data(data_source, symbol, exchange, start, end, interval)

        if data.empty:
            st.error("No data returned. Check symbol and date range or symbol correctness.")
            return

        # Ensure a 'datetime' column
        if 'Date' in data.columns:
            data.rename(columns={'Date': 'datetime'}, inplace=True)
        if 'Datetime' in data.columns:
            data.rename(columns={'Datetime': 'datetime'}, inplace=True)

        # Flatten multi-index if present
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = ['_'.join(tuple(filter(None, col))) for col in data.columns]

        # Remove suffixes (e.g. _RELIANCE.NS)
        for col in data.columns:
            if '_' in col and col.lower() != 'datetime':
                base_col = col.split('_', 1)[0]
                data.rename(columns={col: base_col}, inplace=True)

        # Standardize column names
        col_map = {c.lower(): c for c in data.columns}
        standard_cols = {
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'adj close': 'Adj Close',
            'volume': 'Volume'
        }
        for lc_name, std_name in standard_cols.items():
            if lc_name in col_map:
                curr_name = col_map[lc_name]
                if curr_name != std_name:
                    data.rename(columns={curr_name: std_name}, inplace=True)

        if 'datetime' not in data.columns or 'Close' not in data.columns:
            st.error("Required columns not found.")
            st.write("Columns:", data.columns.tolist())
            return

        st.subheader(f"Raw Data for {symbol} ({exchange}) - {strategy_type}")
        st.dataframe(data.head(50))

        # Add technical indicators
        data = add_all_indicators(data)
        # Apply Pine logic
        data = apply_pine_strategy_logic(data)
        # Apply chosen strategy logic if needed
        if strategy_type == "Intraday":
            data = intraday_strategy(data)
        else:
            data = swing_strategy(data)

        # Evaluate strategy performance
        net_profit, percent_profitable, profit_factor = evaluate_strategy(data)

        # Train and predict future
        future_pred, predict_date, residual_std, r2, err_msg = train_and_predict_future(data, future_horizon, model_type)
        if err_msg:
            st.warning(err_msg)

        # Main Chart
        st.subheader("All-in-One Indicators Plot")
        fig_all = plot_all_in_one(data)
        st.plotly_chart(fig_all, use_container_width=True)

        # Prediction Chart
        if future_pred is not None and predict_date is not None:
            st.subheader("Future Prediction Plot")
            fig_pred = plot_prediction_line(data, predict_date, future_pred, residual_std)
            st.plotly_chart(fig_pred, use_container_width=True)

        # Determine prophet horizon in days
        prophet_days = get_future_horizon_days(prophet_horizon)
        forecast, model = forecast_with_prophet(data, periods=prophet_days)
        fig_forecast = plot_prophet_forecast(data, forecast)
        st.plotly_chart(fig_forecast, use_container_width=True)

        # Display Metrics
        st.subheader("Performance & Model Metrics")
        st.write(f"**Net Profit:** {net_profit:.2f}")
        st.write(f"**Percent Profitable:** {percent_profitable:.2f}%")
        st.write(f"**Profit Factor:** {profit_factor:.2f}")
        if r2 is not None:
            st.write(f"**Model R² (Train):** {r2:.4f}")
        if future_pred is not None and predict_date is not None:
            st.write(f"**Predicted Close Price on {predict_date.date()}:** {future_pred:.2f}")

        # --- Sentiment Analysis Section ---
        st.header("Sentiment Analysis")

        # Fetch company name
        with st.spinner("Fetching company name..."):
            company_name = fetch_company_name(symbol)

        if not company_name:
            st.warning(f"Failed to fetch company name for {symbol}. Using only stock ticker for filtering.")
            company_name = symbol  # Fallback to use the ticker as the filter

        # Fetch news headlines
        with st.spinner("Fetching news headlines..."):
            headlines = fetch_news_headlines(symbol, company_name, limit=15)

        st.subheader("Latest News Headlines")
        if headlines:
            for idx, headline in enumerate(headlines, 1):
                st.write(f"{idx}. {headline}")
        else:
            st.write(f"No relevant news articles found for {symbol}.")

        # Perform sentiment analysis
        if headlines:
            with st.spinner("Performing sentiment analysis..."):
                sentiment_df = perform_sentiment_analysis(headlines)
            st.subheader("Sentiment Analysis of Headlines")
            st.dataframe(sentiment_df)

            # Display average sentiment score
            avg_sentiment = sentiment_df['score'].mean()
            st.write(f"**Average Sentiment Score:** {avg_sentiment:.2f}")
        else:
            sentiment_df = pd.DataFrame()
            st.write("No sentiment data available.")

        # Train sentiment model
        if not sentiment_df.empty:
            with st.spinner("Training sentiment model..."):
                model_sentiment, r2_sentiment = train_sentiment_model(sentiment_df)
            if model_sentiment:
                st.write(f"**Sentiment Model R² Score:** {r2_sentiment:.4f}")
            else:
                st.write("Sentiment model could not be trained due to insufficient data.")
        else:
            model_sentiment = None
            r2_sentiment = None

        # Predict future sentiment
        if model_sentiment is not None:
            with st.spinner("Predicting future sentiment..."):
                last_day = sentiment_df.index.max() + 1  # Assuming days start at 1
                prediction_df = predict_future_sentiment(model_sentiment, last_day, days_ahead=3)
            st.subheader("Predicted Sentiment for Next 3 Days")
            st.dataframe(prediction_df)
        else:
            prediction_df = pd.DataFrame()
            st.write("No predictions available.")

        # Plot sentiment
        if not sentiment_df.empty and not prediction_df.empty:
            with st.spinner("Plotting sentiment analysis..."):
                sentiment_fig = plot_sentiment(sentiment_df, prediction_df)
            st.subheader("Sentiment Analysis & Prediction Plot")
            st.plotly_chart(sentiment_fig, use_container_width=True)
        else:
            st.write("Not enough data to plot sentiment analysis.")

        st.success("Analysis complete.")

    else:
        st.info("Set your parameters in the sidebar and click 'Fetch Data and Analyze' to begin.")

if __name__ == "__main__":
    main()
