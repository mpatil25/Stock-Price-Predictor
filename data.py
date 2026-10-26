"""Data loading: Yahoo Finance via yfinance, a CSV upload, or a synthetic fallback."""
from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED = ["Open", "High", "Low", "Close", "Volume"]


def synthetic_prices(n_days: int = 1500, seed: int = 42) -> pd.DataFrame:
    """Geometric-random-walk OHLCV data, used offline or when downloads fail."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0005, 0.015, n_days)
    close = 100 * np.exp(np.cumsum(returns))
    open_ = close * (1 + rng.normal(0, 0.003, n_days))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.005, n_days)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.005, n_days)))
    volume = rng.integers(1_000_000, 5_000_000, n_days)
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n_days)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=idx,
    )


def load_yahoo(ticker: str, start: str = "2018-01-01", end: str | None = None) -> pd.DataFrame:
    import yfinance as yf

    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df is None or df.empty:
        raise ValueError(f"No data returned for '{ticker}'.")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return clean(df)


def load_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    date_col = next((c for c in df.columns if c.lower() in ("date", "datetime", "timestamp")), None)
    if date_col is None:
        raise ValueError("CSV needs a Date column.")
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col)
    df.columns = [c.title() for c in df.columns]
    return clean(df)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    df = df[REQUIRED].sort_index()
    df = df[~df.index.duplicated(keep="last")].dropna()
    return df
