# plot_utils.py

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

def plot_candles_with_bollinger_and_ma(data: pd.DataFrame) -> go.Figure:
    """
    Plot candlestick chart with Bollinger Bands and Moving Averages.
    Requires 'Open', 'High', 'Low', 'Close' columns.
    Also expects 'BB_Upper', 'BB_Middle', 'BB_Lower', 'MA_Short', 'MA_Long' columns.
    """

    fig = go.Figure()

    # Candlesticks
    fig.add_trace(go.Candlestick(
        x=data['datetime'],
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name='Price',
        increasing_line_color='green', 
        decreasing_line_color='red'
    ))

    # Bollinger Bands
    fig.add_trace(go.Scatter(
        x=data['datetime'],
        y=data['BB_Upper'],
        line=dict(color='rgba(173,216,230,0.5)', width=1),
        name='BB Upper'
    ))
    fig.add_trace(go.Scatter(
        x=data['datetime'],
        y=data['BB_Middle'],
        line=dict(color='rgba(0,0,255,0.5)', width=1),
        name='BB Middle'
    ))
    fig.add_trace(go.Scatter(
        x=data['datetime'],
        y=data['BB_Lower'],
        line=dict(color='rgba(173,216,230,0.5)', width=1),
        name='BB Lower'
    ))

    # Moving Averages
    fig.add_trace(go.Scatter(
        x=data['datetime'],
        y=data['MA_Short'],
        line=dict(color='orange', width=1),
        name='Short MA'
    ))
    fig.add_trace(go.Scatter(
        x=data['datetime'],
        y=data['MA_Long'],
        line=dict(color='purple', width=1),
        name='Long MA'
    ))

    fig.update_layout(
        title="Candlestick with Bollinger Bands and MAs",
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        template="plotly_dark"
    )
    return fig

def plot_rsi(data: pd.DataFrame) -> go.Figure:
    """
    Plot RSI over time.
    Expects a 'RSI' column.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data['datetime'],
        y=data['RSI'],
        line=dict(color='yellow', width=1),
        name='RSI'
    ))

    # Add RSI thresholds (30, 70)
    fig.add_hline(y=30, line_color="red", line_dash="dot", annotation_text="Oversold (30)")
    fig.add_hline(y=70, line_color="green", line_dash="dot", annotation_text="Overbought (70)")

    fig.update_layout(
        title="RSI",
        xaxis_title="Date",
        yaxis_title="RSI Value",
        template="plotly_dark"
    )
    return fig

def plot_macd(data: pd.DataFrame) -> go.Figure:
    """
    Plot MACD, Signal, and Histogram.
    Expects 'MACD', 'MACD_Signal', and 'MACD_Hist' columns.
    """
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.02,
                        row_heights=[0.3, 0.7])

    # Price chart on top (optional if needed)
    if all(col in data.columns for col in ["Open", "High", "Low", "Close"]):
        fig.add_trace(go.Candlestick(
            x=data['datetime'],
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='Price'
        ), row=1, col=1)

    # MACD line
    fig.add_trace(go.Scatter(
        x=data['datetime'],
        y=data['MACD'],
        line=dict(color='cyan', width=1),
        name='MACD'
    ), row=2, col=1)

    # MACD Signal line
    fig.add_trace(go.Scatter(
        x=data['datetime'],
        y=data['MACD_Signal'],
        line=dict(color='magenta', width=1),
        name='Signal'
    ), row=2, col=1)

    # MACD Histogram
    fig.add_trace(go.Bar(
        x=data['datetime'],
        y=data['MACD_Hist'],
        marker_color='white',
        name='Histogram'
    ), row=2, col=1)

    fig.update_layout(
        title="MACD",
        xaxis_title="Date",
        yaxis_title="Value",
        template="plotly_dark"
    )

    return fig

def plot_signals_on_chart(data: pd.DataFrame) -> go.Figure:
    """
    Plot Buy/Sell signals on a candlestick chart.
    Expects 'signal' column where 1=buy and -1=sell.
    """
    fig = go.Figure()

    # Candlesticks
    fig.add_trace(go.Candlestick(
        x=data['datetime'],
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name='Price'
    ))

    # Identify buy and sell points
    buy_points = data[data['signal'] == 1]
    sell_points = data[data['signal'] == -1]

    fig.add_trace(go.Scatter(
        x=buy_points['datetime'],
        y=buy_points['Close'],
        mode='markers',
        marker=dict(symbol='triangle-up', color='green', size=10),
        name='Buy Signal'
    ))

    fig.add_trace(go.Scatter(
        x=sell_points['datetime'],
        y=sell_points['Close'],
        mode='markers',
        marker=dict(symbol='triangle-down', color='red', size=10),
        name='Sell Signal'
    ))

    fig.update_layout(
        title="Buy/Sell Signals",
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_dark",
        xaxis_rangeslider_visible=False
    )

    return fig
