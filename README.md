# Stock Price Predictor

Predicts the next trading day's closing price from historical data, with a Streamlit dashboard.

## Quick start
```bash
pip install -r requirements.txt
streamlit run app.py          # dashboard
python train.py --ticker AAPL # CLI: metrics, forecast, saves best_model.joblib
python train.py --synthetic   # offline demo
```

## Project structure
```
stock_predictor/
├── app.py            # Streamlit dashboard
├── train.py          # CLI training/evaluation
├── requirements.txt
└── src/
    ├── data.py       # Yahoo Finance / CSV / synthetic loaders
    ├── features.py   # technical indicators, target creation
    └── models.py     # models, chronological split, metrics, forecasting
```

## Method
- **Features:** returns, SMA (5/10/20/50), EMA, MACD, RSI-14, 10-day volatility, high-low range, lagged closes.
- **Target:** next day's Close (shifted by -1, so there is no leakage).
- **Models:** Linear Regression, Ridge, Random Forest, Gradient Boosting.
- **Baseline:** naive "tomorrow = today". A model is only useful if it beats this.
- **Evaluation:** chronological 80/20 split; MAE, RMSE, MAPE, R², direction accuracy.
- **Forecast:** recursive multi-day prediction with a rough uncertainty band.

## Dashboard features
Candlestick chart with SMAs, volume and RSI; model comparison table; actual vs predicted plot; residuals and
feature importance; forecast with downloadable CSV; CSV upload; offline demo mode.

## Ideas to extend
LSTM/GRU model, walk-forward cross-validation (`TimeSeriesSplit`), news sentiment features, hyperparameter tuning,
predicting returns instead of price.

*Educational project only. Not financial advice.*
