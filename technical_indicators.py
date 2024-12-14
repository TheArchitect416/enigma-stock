# technical_indicators.py

import pandas as pd
import numpy as np

def calculate_rsi(data: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Calculate the RSI (Relative Strength Index).
    :param data: DataFrame containing at least 'Close' column, indexed by time.
    :param period: RSI calculation period (default=14)
    :return: DataFrame with 'RSI' column added.
    """
    data = data.copy()

    # Compute price changes
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0))
    loss = (-delta.where(delta < 0, 0))

    # Exponential Moving Average for the gains and losses
    gain_ema = gain.ewm(com=(period - 1), adjust=False).mean()
    loss_ema = loss.ewm(com=(period - 1), adjust=False).mean()

    # Calculate RSI
    rs = gain_ema / loss_ema
    data['RSI'] = 100 - (100 / (1 + rs))
    return data

def calculate_macd(data: pd.DataFrame, fastperiod: int = 12, slowperiod: int = 26, signalperiod: int = 9) -> pd.DataFrame:
    """
    Calculate the MACD (Moving Average Convergence Divergence).
    :param data: DataFrame containing 'Close' column.
    :param fastperiod: Fast EMA period
    :param slowperiod: Slow EMA period
    :param signalperiod: Signal line EMA period
    :return: DataFrame with 'MACD', 'MACD_Signal', 'MACD_Hist' columns added.
    """
    data = data.copy()
    # EMAs
    ema_fast = data['Close'].ewm(span=fastperiod, adjust=False).mean()
    ema_slow = data['Close'].ewm(span=slowperiod, adjust=False).mean()

    data['MACD'] = ema_fast - ema_slow
    data['MACD_Signal'] = data['MACD'].ewm(span=signalperiod, adjust=False).mean()
    data['MACD_Hist'] = data['MACD'] - data['MACD_Signal']
    return data

def calculate_bollinger_bands(data: pd.DataFrame, period: int = 20, num_std: float = 2) -> pd.DataFrame:
    """
    Calculate Bollinger Bands.
    :param data: DataFrame with 'Close' column.
    :param period: period for moving average
    :param num_std: number of standard deviations
    :return: DataFrame with 'BB_Middle', 'BB_Upper', 'BB_Lower' columns.
    """
    data = data.copy()
    data['BB_Middle'] = data['Close'].rolling(window=period).mean()
    data['BB_STD'] = data['Close'].rolling(window=period).std()
    data['BB_Upper'] = data['BB_Middle'] + (num_std * data['BB_STD'])
    data['BB_Lower'] = data['BB_Middle'] - (num_std * data['BB_STD'])
    data.drop(columns=['BB_STD'], inplace=True)
    return data

def calculate_moving_averages(data: pd.DataFrame, short_window: int = 20, long_window: int = 50) -> pd.DataFrame:
    """
    Calculate short-term and long-term moving averages.
    :param data: DataFrame with 'Close' column.
    :param short_window: period for short-term MA
    :param long_window: period for long-term MA
    :return: DataFrame with 'MA_Short' and 'MA_Long' columns.
    """
    data = data.copy()
    data['MA_Short'] = data['Close'].rolling(window=short_window).mean()
    data['MA_Long'] = data['Close'].rolling(window=long_window).mean()
    return data

def add_all_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience function to add all indicators (RSI, MACD, BB, MA) to the DataFrame.
    """
    data = calculate_rsi(data)
    data = calculate_macd(data)
    data = calculate_bollinger_bands(data)
    data = calculate_moving_averages(data)
    return data
