"""Model training, chronological evaluation and multi-day forecasting."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURES, add_features


class ReturnModel(BaseEstimator, RegressorMixin):
    """Wraps a regressor so it predicts next-day *return*; output is converted back to a price.
    Tree models cannot extrapolate raw price levels, but returns are stationary."""

    def __init__(self, base=None):
        self.base = base

    def fit(self, X, y):
        self.base_ = clone(self.base).fit(X, y / X["Close"] - 1)
        return self

    def predict(self, X):
        return X["Close"].values * (1 + self.base_.predict(X))


def get_models() -> dict:
    models = {
        "Linear Regression": make_pipeline(StandardScaler(), LinearRegression()),
        "Ridge Regression": make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=10, n_jobs=-1, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42),
    }
    return {k: ReturnModel(v) for k, v in models.items()}


def chrono_split(X, y, test_size: float = 0.2):
    """Chronological split: the test set is always the most recent data."""
    cut = int(len(X) * (1 - test_size))
    return X.iloc[:cut], X.iloc[cut:], y.iloc[:cut], y.iloc[cut:]


def metrics(y_true, y_pred) -> dict:
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAPE %": float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100),
        "R²": r2_score(y_true, y_pred),
    }


def direction_accuracy(X_test, y_true, y_pred) -> float:
    """How often the model gets up/down right relative to today's Close."""
    actual = np.sign(y_true.values - X_test["Close"].values)
    pred = np.sign(np.asarray(y_pred) - X_test["Close"].values)
    return float((actual == pred).mean() * 100)


def evaluate_all(X, y, test_size: float = 0.2):
    """Fit every model, plus a naive 'tomorrow = today' baseline."""
    X_tr, X_te, y_tr, y_te = chrono_split(X, y, test_size)
    rows, preds, fitted = [], {}, {}

    base = X_te["Close"].values
    rows.append({"Model": "Naive baseline", **metrics(y_te.values, base),
                 "Direction acc %": float("nan")})
    preds["Naive baseline"] = pd.Series(base, index=y_te.index)

    for name, model in get_models().items():
        model.fit(X_tr, y_tr)
        p = model.predict(X_te)
        rows.append({"Model": name, **metrics(y_te.values, p),
                     "Direction acc %": direction_accuracy(X_te, y_te, p)})
        preds[name] = pd.Series(p, index=y_te.index)
        fitted[name] = model

    table = pd.DataFrame(rows).set_index("Model").sort_values("RMSE")
    return table, preds, fitted, (X_tr, X_te, y_tr, y_te)


def forecast(model, history: pd.DataFrame, days: int = 7) -> pd.DataFrame:
    """Recursive forecast: feed each prediction back in as the next Close.
    Uncertainty grows with horizon, so treat later days with caution."""
    hist = history.copy()
    out = []
    next_date = hist.index[-1]
    for _ in range(days):
        feats = add_features(hist)[FEATURES].dropna().iloc[[-1]]
        pred = float(model.predict(feats)[0])
        next_date = next_date + pd.tseries.offsets.BDay(1)
        out.append((next_date, pred))
        last = hist.iloc[-1]
        hist.loc[next_date] = {
            "Open": pred, "High": max(pred, last["Close"]), "Low": min(pred, last["Close"]),
            "Close": pred, "Volume": last["Volume"],
        }
    return pd.DataFrame(out, columns=["Date", "Predicted Close"]).set_index("Date")


def feature_importance(model) -> pd.Series | None:
    est = model.base_ if hasattr(model, "base_") else model
    est = est[-1] if hasattr(est, "steps") else est
    if hasattr(est, "feature_importances_"):
        s = pd.Series(est.feature_importances_, index=FEATURES)
    elif hasattr(est, "coef_"):
        s = pd.Series(np.abs(est.coef_), index=FEATURES)
        s = s / s.sum()
    else:
        return None
    return s.sort_values(ascending=False)
