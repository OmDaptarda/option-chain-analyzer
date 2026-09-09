import math
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


NORMAL = 1.0 / math.sqrt(2.0 * math.pi)


def normal_pdf(x: float) -> float:
    return NORMAL * math.exp(-0.5 * x * x)


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def d1_d2(
    spot: float, strike: float, time: float, rate: float, volatility: float, dividend: float
) -> tuple[float, float]:
    root_time = math.sqrt(time)
    d1 = (
        math.log(spot / strike)
        + (rate - dividend + 0.5 * volatility * volatility) * time
    ) / (volatility * root_time)
    return d1, d1 - volatility * root_time


def model_price(
    spot: float,
    strike: float,
    time: float,
    rate: float,
    volatility: float,
    option_type: str,
    dividend: float = 0.0,
) -> float:
    d1, d2 = d1_d2(spot, strike, time, rate, volatility, dividend)
    discount_spot = spot * math.exp(-dividend * time)
    discount_strike = strike * math.exp(-rate * time)
    if option_type == "Call":
        return discount_spot * normal_cdf(d1) - discount_strike * normal_cdf(d2)
    return discount_strike * normal_cdf(-d2) - discount_spot * normal_cdf(-d1)


def greeks(
    spot: float,
    strike: float,
    time: float,
    rate: float,
    volatility: float,
    option_type: str,
    dividend: float = 0.0,
) -> dict[str, float]:
    d1, d2 = d1_d2(spot, strike, time, rate, volatility, dividend)
    sign = 1.0 if option_type == "Call" else -1.0
    discounted_spot = spot * math.exp(-dividend * time)
    discounted_strike = strike * math.exp(-rate * time)
    delta = sign * math.exp(-dividend * time) * normal_cdf(sign * d1)
    gamma = math.exp(-dividend * time) * normal_pdf(d1) / (spot * volatility * math.sqrt(time))
    vega = discounted_spot * normal_pdf(d1) * math.sqrt(time) / 100.0
    theta = (
        -discounted_spot * normal_pdf(d1) * volatility / (2.0 * math.sqrt(time))
        - sign * rate * discounted_strike * normal_cdf(sign * d2)
        + sign * dividend * discounted_spot * normal_cdf(sign * d1)
    ) / 365.0
    rho = sign * time * discounted_strike * normal_cdf(sign * d2) / 100.0
    return {"Delta": delta, "Gamma": gamma, "Vega": vega, "Theta": theta, "Rho": rho}


def implied_volatility(
    market_price: float,
    spot: float,
    strike: float,
    time: float,
    rate: float,
    option_type: str,
    dividend: float = 0.0,
) -> Optional[float]:
    if market_price <= 0 or min(spot, strike, time) <= 0:
        return None
    low, high = 1e-6, 5.0
    low_price = model_price(spot, strike, time, rate, low, option_type, dividend)
    high_price = model_price(spot, strike, time, rate, high, option_type, dividend)
    if market_price < low_price or market_price > high_price:
        return None
    for _ in range(100):
        mid = (low + high) / 2.0
        price = model_price(spot, strike, time, rate, mid, option_type, dividend)
        if abs(price - market_price) < 1e-8:
            return mid
        if price < market_price:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0


def sample_chain(spot: float, time: float, rate: float, volatility: float) -> pd.DataFrame:
    rows = []
    for strike in range(round(spot * 0.85), round(spot * 1.16), 5):
        for option_type in ("Call", "Put"):
            fair = model_price(spot, strike, time, rate, volatility, option_type)
            adjustment = 1.0 + (0.025 if strike == round(spot / 5) * 5 and option_type == "Call" else 0.0)
            rows.append(
                {
                    "symbol": "DEMO",
                    "option_type": option_type,
                    "strike": float(strike),
                    "market_price": round(fair * adjustment, 2),
                    "bid": round(fair * adjustment - 0.1, 2),
                    "ask": round(fair * adjustment + 0.1, 2),
                }
            )
    return pd.DataFrame(rows)


def analyze_chain(
    chain: pd.DataFrame, spot: float, time: float, rate: float, dividend: float, volatility: float
) -> pd.DataFrame:
    required = {"option_type", "strike", "market_price"}
    missing = required - set(chain.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {', '.join(sorted(missing))}")
    result = chain.copy()
    result["model_price"] = result.apply(
        lambda row: model_price(
            spot, float(row["strike"]), time, rate, volatility, str(row["option_type"]), dividend
        ),
        axis=1,
    )
    result["implied_volatility"] = result.apply(
        lambda row: implied_volatility(
            float(row["market_price"]),
            spot,
            float(row["strike"]),
            time,
            rate,
            str(row["option_type"]),
            dividend,
        ),
        axis=1,
    )
    result["edge"] = result["model_price"] - result["market_price"]
    result["edge_pct"] = result["edge"] / result["market_price"].replace(0, math.nan)
    result["signal"] = result["edge_pct"].apply(
        lambda value: "UNDERPRICED" if value >= 0.05 else "OVERPRICED" if value <= -0.05 else "FAIR"
    )
    greek_rows = result.apply(
        lambda row: greeks(
            spot, float(row["strike"]), time, rate, volatility, str(row["option_type"]), dividend
        ),
        axis=1,
    )
    return pd.concat([result, pd.DataFrame(greek_rows.tolist(), index=result.index)], axis=1)


def payoff_chart(spot: float, strike: float, premium: float, option_type: str) -> go.Figure:
    prices = [spot * (0.5 + i / 100.0) for i in range(101)]
    sign = 1 if option_type == "Call" else -1
    payoffs = [max(sign * (price - strike), 0.0) - premium for price in prices]
    figure = go.Figure(go.Scatter(x=prices, y=payoffs, mode="lines", name=f"{option_type} P/L"))
    figure.add_hline(y=0, line_dash="dot", line_color="gray")
    figure.add_vline(x=strike, line_dash="dot", line_color="orange", annotation_text="Strike")
    figure.update_layout(xaxis_title="Underlying price at expiry", yaxis_title="Profit / loss per share")
    return figure


st.set_page_config(page_title="Option Chain Analyzer", layout="wide")
st.title("Option Chain Analyzer")
st.caption("Black–Scholes fair value, implied volatility, Greeks, payoff, and model-vs-market signals")

with st.sidebar:
    st.header("Market assumptions")
    spot = st.number_input("Underlying price", min_value=0.01, value=100.0, step=0.5)
    days = st.number_input("Days to expiry", min_value=1, value=30, step=1)
    rate = st.number_input("Risk-free rate (%)", value=5.0, step=0.1) / 100.0
    dividend = st.number_input("Dividend yield (%)", value=0.0, step=0.1) / 100.0
    volatility = st.number_input("Model volatility (%)", min_value=0.1, value=25.0, step=0.5) / 100.0
    threshold = st.slider("Mispricing threshold (%)", 1, 25, 5) / 100.0
    time = days / 365.0

uploaded = st.file_uploader("Upload option chain CSV (optional)", type="csv")
chain = pd.read_csv(uploaded) if uploaded else sample_chain(spot, time, rate, volatility)
try:
    analyzed = analyze_chain(chain, spot, time, rate, dividend, volatility)
except ValueError as error:
    st.error(str(error))
    st.stop()

analyzed["signal"] = analyzed["edge_pct"].apply(
    lambda value: "UNDERPRICED" if value >= threshold else "OVERPRICED" if value <= -threshold else "FAIR"
)

col1, col2, col3 = st.columns(3)
col1.metric("Quotes analyzed", len(analyzed))
col2.metric("Underpriced candidates", int((analyzed["signal"] == "UNDERPRICED").sum()))
col3.metric("Overpriced candidates", int((analyzed["signal"] == "OVERPRICED").sum()))

display_columns = [
    column
    for column in [
        "symbol", "option_type", "strike", "market_price", "model_price",
        "implied_volatility", "edge_pct", "signal", "Delta", "Gamma", "Vega", "Theta", "Rho",
    ]
    if column in analyzed.columns
]
st.dataframe(
    analyzed[display_columns].style.format(
        {
            "market_price": "{:.2f}", "model_price": "{:.2f}", "implied_volatility": "{:.2%}",
            "edge_pct": "{:.2%}", "Delta": "{:.4f}", "Gamma": "{:.6f}",
            "Vega": "{:.4f}", "Theta": "{:.4f}", "Rho": "{:.4f}",
        },
        na_rep="n/a",
    ),
    use_container_width=True,
)

st.subheader("Payoff explorer")
selected = st.selectbox("Quote", analyzed.index, format_func=lambda index: (
    f"{analyzed.loc[index, 'option_type']} {analyzed.loc[index, 'strike']:.2f} "
    f"@ {analyzed.loc[index, 'market_price']:.2f}"
))
selected_row = analyzed.loc[selected]
st.plotly_chart(
    payoff_chart(
        spot,
        float(selected_row["strike"]),
        float(selected_row["market_price"]),
        str(selected_row["option_type"]),
    ),
    use_container_width=True,
)
