"""Streamlit dashboard for the Stock Price Predictor.  Run:  streamlit run app.py"""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.data import load_csv, load_yahoo, synthetic_prices
from src.features import add_features, make_dataset
from src.models import evaluate_all, feature_importance, forecast

st.set_page_config(page_title="Stock Price Predictor", page_icon="📈", layout="wide")

# ---------------------------------------------------------------- sidebar
st.sidebar.title("📈 Settings")
source = st.sidebar.radio("Data source", ["Yahoo Finance", "Upload CSV", "Demo (synthetic)"])

df, label = None, ""
if source == "Yahoo Finance":
    ticker = st.sidebar.text_input("Ticker", "AAPL").upper().strip()
    start = st.sidebar.date_input("Start date", pd.Timestamp("2018-01-01"))
    label = ticker
elif source == "Upload CSV":
    up = st.sidebar.file_uploader("CSV with Date, Open, High, Low, Close, Volume", type="csv")
    label = "Uploaded data"
else:
    label = "Demo data"

test_size = st.sidebar.slider("Test set size (most recent %)", 10, 40, 20, step=5) / 100
horizon = st.sidebar.slider("Forecast horizon (business days)", 1, 30, 7)


@st.cache_data(show_spinner="Loading data...")
def get_data(src, tkr=None, start=None, file_bytes=None):
    if src == "Yahoo Finance":
        return load_yahoo(tkr, str(start))
    if src == "Upload CSV":
        import io
        return load_csv(io.BytesIO(file_bytes))
    return synthetic_prices()


try:
    if source == "Yahoo Finance":
        df = get_data(source, ticker, start)
    elif source == "Upload CSV":
        if up is not None:
            df = get_data(source, file_bytes=up.getvalue())
    else:
        df = get_data(source)
except Exception as e:
    st.error(f"Could not load data: {e}")
    st.info("Tip: if you are offline, choose **Demo (synthetic)** in the sidebar.")
    st.stop()

if df is None:
    st.title("Stock Price Predictor")
    st.info("Upload a CSV in the sidebar to begin.")
    st.stop()
if len(df) < 200:
    st.error("Need at least 200 rows of history. Pick an earlier start date.")
    st.stop()


@st.cache_data(show_spinner="Training models...")
def train(data, test_size):
    X, y, latest = make_dataset(data)
    table, preds, fitted, split = evaluate_all(X, y, test_size)
    return table, preds, fitted, split


table, preds, fitted, (X_tr, X_te, y_tr, y_te) = train(df, test_size)

# ---------------------------------------------------------------- header
st.title(f"Stock Price Predictor: {label}")
st.caption("Predicts the next trading day's closing price from historical price, volume and technical indicators. "
           "Educational project, not financial advice.")

last, prev = df["Close"].iloc[-1], df["Close"].iloc[-2]
best = table.drop(index="Naive baseline").index[0]
next_pred = forecast(fitted[best], df, horizon)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Last close", f"{last:,.2f}", f"{(last / prev - 1) * 100:+.2f}%")
c2.metric(f"Next-day forecast ({best})", f"{next_pred['Predicted Close'].iloc[0]:,.2f}",
          f"{(next_pred['Predicted Close'].iloc[0] / last - 1) * 100:+.2f}%")
c3.metric("Best model RMSE", f"{table.loc[best, 'RMSE']:.2f}",
          f"{table.loc[best, 'RMSE'] - table.loc['Naive baseline', 'RMSE']:+.2f} vs baseline", delta_color="inverse")
c4.metric("Direction accuracy", f"{table.loc[best, 'Direction acc %']:.1f}%")

tab1, tab2, tab3, tab4 = st.tabs(["📊 Market", "🤖 Model comparison", "🔮 Forecast", "ℹ️ About"])

# ---------------------------------------------------------------- market
with tab1:
    feat = add_features(df)
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.6, 0.2, 0.2], vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(x=df.index, open=df.Open, high=df.High, low=df.Low, close=df.Close, name="Price"), 1, 1)
    for w, col in ((20, "#f59e0b"), (50, "#3b82f6")):
        fig.add_trace(go.Scatter(x=feat.index, y=feat[f"sma_{w}"], name=f"SMA {w}", line=dict(color=col, width=1.2)), 1, 1)
    fig.add_trace(go.Bar(x=df.index, y=df.Volume, name="Volume", marker_color="#94a3b8"), 2, 1)
    fig.add_trace(go.Scatter(x=feat.index, y=feat.rsi_14, name="RSI 14", line=dict(color="#8b5cf6")), 3, 1)
    fig.add_hline(y=70, line_dash="dot", line_color="red", row=3, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="green", row=3, col=1)
    fig.update_layout(height=700, xaxis_rangeslider_visible=False, legend=dict(orientation="h", y=1.02))
    st.plotly_chart(fig, width="stretch")
    with st.expander("Raw data"):
        st.dataframe(df.tail(100).sort_index(ascending=False), width="stretch")

# ---------------------------------------------------------------- models
with tab2:
    st.subheader("Held-out test performance")
    st.caption(f"Trained on the first {len(X_tr)} days, tested on the most recent {len(X_te)} days (no shuffling).")
    st.dataframe(table.round(3).style.highlight_min(subset=["MAE", "RMSE", "MAPE %"], color="#bbf7d0")
                 .highlight_max(subset=["R²", "Direction acc %"], color="#bbf7d0"),
                 width="stretch")
    chosen = st.multiselect("Models to plot", list(preds), default=["Naive baseline", best])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_te.index, y=y_te, name="Actual", line=dict(color="black", width=2)))
    for m in chosen:
        fig.add_trace(go.Scatter(x=preds[m].index, y=preds[m], name=m, opacity=0.85))
    fig.update_layout(height=450, yaxis_title="Close price", legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, width="stretch")

    left, right = st.columns(2)
    with left:
        err = (y_te - preds[best])
        fig = go.Figure(go.Histogram(x=err, nbinsx=40, marker_color="#3b82f6"))
        fig.update_layout(title=f"{best}: residuals", height=320, xaxis_title="Actual - Predicted")
        st.plotly_chart(fig, width="stretch")
    with right:
        imp = feature_importance(fitted[best])
        if imp is not None:
            top = imp.head(10)[::-1]
            fig = go.Figure(go.Bar(x=top.values, y=top.index, orientation="h", marker_color="#10b981"))
            fig.update_layout(title=f"{best}: top features", height=320)
            st.plotly_chart(fig, width="stretch")

    st.warning("If a model barely beats the naive baseline, that is normal: next-day prices are very close to "
               "today's price and are hard to predict. Compare against the baseline, not just R².")

# ---------------------------------------------------------------- forecast
with tab3:
    pick = st.selectbox("Model", [m for m in table.index if m != "Naive baseline"],
                        index=[m for m in table.index if m != "Naive baseline"].index(best))
    fc = forecast(fitted[pick], df, horizon)
    recent = df["Close"].tail(90)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recent.index, y=recent, name="History", line=dict(color="black")))
    fig.add_trace(go.Scatter(x=fc.index, y=fc["Predicted Close"], name="Forecast",
                             line=dict(color="#ef4444", dash="dash"), mode="lines+markers"))
    rmse = table.loc[pick, "RMSE"]
    band = rmse * (pd.Series(range(1, horizon + 1), index=fc.index) ** 0.5)
    fig.add_trace(go.Scatter(x=list(fc.index) + list(fc.index[::-1]),
                             y=list(fc["Predicted Close"] + band) + list((fc["Predicted Close"] - band)[::-1]),
                             fill="toself", fillcolor="rgba(239,68,68,0.15)", line=dict(width=0),
                             name="Approx. uncertainty"))
    fig.update_layout(height=450, yaxis_title="Close price")
    st.plotly_chart(fig, width="stretch")
    out = fc.copy()
    out["Change vs last close %"] = (out["Predicted Close"] / last - 1) * 100
    st.dataframe(out.round(2), width="stretch")
    st.download_button("Download forecast CSV", out.to_csv().encode(), "forecast.csv", "text/csv")
    st.caption("Multi-day forecasts feed predictions back in recursively, so error compounds. "
               "The shaded band is a rough sqrt(time) heuristic based on test RMSE, not a statistical guarantee.")

# ---------------------------------------------------------------- about
with tab4:
    st.markdown("""
**Pipeline**
1. **Data**: daily OHLCV from Yahoo Finance (or CSV / synthetic demo).
2. **Features**: returns, SMA/EMA, MACD, RSI, volatility, high-low range, lagged closes. All use past data only.
3. **Target**: next trading day's Close.
4. **Models**: Linear, Ridge, Random Forest, Gradient Boosting, plus a naive "tomorrow = today" baseline.
5. **Validation**: chronological train/test split (no shuffling, so no look-ahead leakage).
6. **Metrics**: MAE, RMSE, MAPE, R², and up/down direction accuracy.

**Limitations**: markets are noisy; news and events are not modelled; past performance does not predict future results.
""")
