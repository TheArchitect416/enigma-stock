# strategies.py

import pandas as pd

def intraday_strategy(data: pd.DataFrame) -> pd.DataFrame:
    """
    A simple intraday strategy example:
    - Use RSI: if RSI < 30 -> Buy signal, if RSI > 70 -> Sell signal.
    - For intraday, we assume the data is at a short interval (e.g. 5m, 15m)
      and the strategy runs from market open to close of the same day.

    Assumptions:
    - 'RSI' column exists in `data`.
    - 'datetime' is present and data is sorted in ascending order.
    - We'll generate a 'signal' column: 1 for buy, -1 for sell, 0 for hold.
    """
    data = data.copy()
    if 'datetime' not in data.columns:
        # Handle the error or return data unchanged
        return data
    data["signal"] = 0

    # Simple logic: If RSI < 30 at any candle => Buy
    # If RSI > 70 at any candle => Sell
    # Only consider the first signal in the day for simplicity
    # (In practice, more complex logic or multiple signals might be used)
    # We'll just mark signals and let the user interpret or filter them.

    # Identify market date from datetime (assuming local data)
    data['date'] = data['datetime'].dt.date

    # For each date, we look for RSI triggers
    grouped = data.groupby('date')
    def intraday_signals_for_day(df):
        # If RSI < 30 at any point, set a buy signal at that point (if no previous signals)
        buy_points = df.index[df["RSI"] < 30].tolist()
        sell_points = df.index[df["RSI"] > 70].tolist()
        # We can pick the earliest buy and earliest sell for simplicity
        if buy_points:
            df.at[buy_points[0], "signal"] = 1
        if sell_points:
            df.at[sell_points[0], "signal"] = -1
        return df

    data = grouped.apply(intraday_signals_for_day)
    data.drop(columns=['date'], inplace=True)
    return data

def swing_strategy(data: pd.DataFrame) -> pd.DataFrame:
    """
    A simple swing trading strategy example:
    - Use MACD crossovers: 
      If MACD crosses above Signal line => Buy
      If MACD crosses below Signal line => Sell
    - This applies over a period of days to months, so daily data is typically used.

    Assumptions:
    - 'MACD' and 'MACD_Signal' columns exist in `data`.
    - 'datetime' is present and data is sorted by datetime ascending.
    - We'll generate a 'signal' column: 1 for buy, -1 for sell, 0 for hold.
    """

    data = data.copy()
    data["signal"] = 0

    # Detect MACD line crossing above/below the signal line
    # We look for points where MACD(t) > MACD_Signal(t) and previously MACD(t-1) <= MACD_Signal(t-1) => Buy
    # And vice versa for Sell.
    data["prev_MACD"] = data["MACD"].shift(1)
    data["prev_Signal"] = data["MACD_Signal"].shift(1)

    # Buy signal
    buy_condition = (data["MACD"] > data["MACD_Signal"]) & (data["prev_MACD"] <= data["prev_Signal"])
    # Sell signal
    sell_condition = (data["MACD"] < data["MACD_Signal"]) & (data["prev_MACD"] >= data["prev_Signal"])

    data.loc[buy_condition, "signal"] = 1
    data.loc[sell_condition, "signal"] = -1

    data.drop(columns=["prev_MACD", "prev_Signal"], inplace=True)
    return data
