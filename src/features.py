"""Feature engineering. Every feature at row t uses only data up to day t,
and the target is the Close price of day t+1 (no look-ahead leakage)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    c = out["Close"]
    out["ret_1d"] = c.pct_change()
    out["ret_5d"] = c.pct_change(5)
    for w in (5, 10, 20, 50):
        out[f"sma_{w}"] = c.rolling(w).mean()
        out[f"close_to_sma_{w}"] = c / out[f"sma_{w}"] - 1
    out["ema_12"] = c.ewm(span=12, adjust=False).mean()
    out["ema_26"] = c.ewm(span=26, adjust=False).mean()
    out["macd"] = out["ema_12"] - out["ema_26"]
    out["rsi_14"] = rsi(c)
    out["vol_10d"] = out["ret_1d"].rolling(10).std()
    out["hl_range"] = (out["High"] - out["Low"]) / c
    out["vol_chg"] = out["Volume"].pct_change()
    for lag in (1, 2, 3, 5):
        out[f"lag_close_{lag}"] = c.shift(lag)
    return out


FEATURES = [
    "Open", "High", "Low", "Close", "Volume", "ret_1d", "ret_5d",
    "sma_5", "sma_10", "sma_20", "sma_50",
    "close_to_sma_5", "close_to_sma_10", "close_to_sma_20", "close_to_sma_50",
    "ema_12", "ema_26", "macd", "rsi_14", "vol_10d", "hl_range", "vol_chg",
    "lag_close_1", "lag_close_2", "lag_close_3", "lag_close_5",
]


def make_dataset(df: pd.DataFrame):
    """Return (X, y, latest_row). y = next day's Close."""
    feat = add_features(df).replace([np.inf, -np.inf], np.nan)
    feat["target"] = feat["Close"].shift(-1)
    latest = feat[FEATURES].dropna().iloc[[-1]]
    data = feat.dropna(subset=FEATURES + ["target"])
    return data[FEATURES], data["target"], latest
