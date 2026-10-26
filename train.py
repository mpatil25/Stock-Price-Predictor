"""Command-line training script.

Example:
    python train.py --ticker AAPL --start 2018-01-01
    python train.py --synthetic
"""
import argparse

import joblib

from src.data import load_yahoo, synthetic_prices
from src.features import make_dataset
from src.models import evaluate_all, forecast


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="AAPL")
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--synthetic", action="store_true", help="use generated data (offline)")
    ap.add_argument("--days", type=int, default=7, help="forecast horizon")
    args = ap.parse_args()

    df = synthetic_prices() if args.synthetic else load_yahoo(args.ticker, args.start)
    X, y, _ = make_dataset(df)
    table, _, fitted, _ = evaluate_all(X, y)
    print(table.round(3).to_string())

    best = table.drop(index="Naive baseline").index[0]
    print(f"\nBest model by RMSE: {best}")
    print(forecast(fitted[best], df, args.days).round(2).to_string())
    joblib.dump(fitted[best], "best_model.joblib")
    print("\nSaved best_model.joblib")


if __name__ == "__main__":
    main()
