# data_fetch.py

import os
import requests
import pandas as pd
import yfinance as yf
from datetime import datetime
from config import ALPHA_VANTAGE_API_KEY

def get_yfinance_symbol(symbol: str, exchange: str) -> str:
    """
    Converts a plain symbol and exchange choice into a yfinance-compatible symbol.
    For NSE we often add '.NS' suffix.
    For BSE we often add '.BO' suffix.
    """
    if exchange.upper() == "NSE":
        return symbol.upper() + ".NS"
    elif exchange.upper() == "BSE":
        return symbol.upper() + ".BO"
    else:
        # Default fallback, no suffix
        return symbol

def get_yfinance_data(symbol: str, exchange: str, start: datetime, end: datetime, interval: str = "1d") -> pd.DataFrame:
    yf_symbol = get_yfinance_symbol(symbol, exchange)
    data = yf.download(yf_symbol, start=start, end=end, interval=interval)

    if not data.empty:
        data.reset_index(inplace=True)

        # If we got a MultiIndex or suffixed columns, flatten them
        if isinstance(data.columns, pd.MultiIndex):
            # Flatten multi-level columns: ('Adj Close', 'RELIANCE.NS') -> 'Adj Close_RELIANCE.NS'
            data.columns = ['_'.join(tuple(filter(None, col))) for col in data.columns]
        
        # Rename Datetime or Date column to 'datetime'
        if 'Date' in data.columns:
            data.rename(columns={'Date': 'datetime'}, inplace=True)
        if 'Datetime' in data.columns:
            data.rename(columns={'Datetime': 'datetime'}, inplace=True)

        # If columns have suffixes like _RELIANCE.NS, remove them:
        # Example: 'Close_RELIANCE.NS' -> 'Close'
        for col in data.columns:
            if 'RELIANCE.NS' in col or 'BO' in col:
                new_col = col.split('_')[0]
                data.rename(columns={col: new_col}, inplace=True)

    return data

def get_alpha_vantage_data(symbol: str, exchange: str, start: datetime, end: datetime, interval: str = "1d") -> pd.DataFrame:
    """
    Fetches historical stock data from Alpha Vantage.
    Uses TIME_SERIES_DAILY or INTRADAY endpoints depending on interval.
    NOTE: Alpha Vantage may not directly provide NSE/BSE data. 
    For demonstration, this code attempts to fetch global symbols if available.
    The user must ensure the symbol is recognized by Alpha Vantage.
    """
    base_url = "https://www.alphavantage.co/query"
    params = {
        "apikey": ALPHA_VANTAGE_API_KEY
    }

    # Choose function based on interval
    # For intraday (like '5min'), use TIME_SERIES_INTRADAY
    # For daily, TIME_SERIES_DAILY_ADJUSTED is used
    if interval.endswith('m'):  # Intraday logic: '1m', '5m', '15m', '30m', '60m'
        # Alpha Vantage supports certain fixed intervals: 1min, 5min, 15min, 30min, 60min
        # Map the given interval to something alpha vantage understands
        valid_intraday_intervals = ["1min", "5min", "15min", "30min", "60min"]
        alpha_interval = interval if interval in valid_intraday_intervals else "5min"
        params.update({
            "function": "TIME_SERIES_INTRADAY",
            "symbol": symbol,
            "interval": alpha_interval,
            "outputsize": "full"
        })
    else:
        # Daily
        params.update({
            "function": "TIME_SERIES_DAILY_ADJUSTED",
            "symbol": symbol,
            "outputsize": "full"
        })

    response = requests.get(base_url, params=params)
    data = response.json()

    # Parse JSON depending on function used
    if "Time Series" not in str(data.keys()):
        # In case we did not receive the expected format, return empty DataFrame
        return pd.DataFrame()

    # Identify the key for time series
    time_series_key = None
    for k in data.keys():
        if "Time Series" in k:
            time_series_key = k
            break
    
    ts_data = data.get(time_series_key, {})
    df = pd.DataFrame.from_dict(ts_data, orient='index')
    df.index = pd.to_datetime(df.index)
    df.reset_index(inplace=True)
    df.rename(columns={"index": "datetime"}, inplace=True)
    df.sort_index(inplace=True)
    df = df[(df.index >= start) & (df.index <= end)]
    # Rename columns to standard OHLCV
    rename_map = {
        '1. open': 'Open',
        '2. high': 'High',
        '3. low': 'Low',
        '4. close': 'Close',
        '5. adjusted close': 'Adj Close',
        '5. volume': 'Volume',  # For intraday, keys differ slightly
        '6. volume': 'Volume',
        '6. dividend amount': 'Dividend',
        '7. split coefficient': 'Split Coefficient'
    }

    df.rename(columns=rename_map, inplace=True)
    # Ensure required columns
    for c in ["Open", "High", "Low", "Close"]:
        if c not in df.columns:
            df[c] = None
    if "Volume" not in df.columns:
        df["Volume"] = None

    df.reset_index(inplace=True)
    df.rename(columns={"index": "datetime"}, inplace=True)
    return df

def fetch_data(source: str, symbol: str, exchange: str, start: datetime, end: datetime, interval: str = "1d") -> pd.DataFrame:
    """
    Fetch data from the chosen source.
    :param source: 'yfinance' or 'alpha_vantage'
    :param symbol: Ticker symbol (without exchange suffix)
    :param exchange: 'NSE' or 'BSE'
    :param start: start datetime
    :param end: end datetime
    :param interval: data interval
    """
    if source == "yfinance":
        return get_yfinance_data(symbol, exchange, start, end, interval)
    elif source == "alpha_vantage":
        return get_alpha_vantage_data(symbol, exchange, start, end, interval)
    else:
        raise ValueError("Invalid data source. Choose 'yfinance' or 'alpha_vantage'.")
