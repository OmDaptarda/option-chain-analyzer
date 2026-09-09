import math
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


NORMAL = 1.0 / math.sqrt(2.0 * math.pi)
REQUIRED_COLUMNS = {"option_type", "strike", "market_price"}


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
            adjustment = 1.0 + (
                0.025 if strike == round(spot / 5) * 5 and option_type == "Call" else 0.0
            )
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


def validate_chain(chain: pd.DataFrame) -> pd.DataFrame:
    if chain.empty:
        raise ValueError("The uploaded CSV is empty. Add at least one option quote.")

    result = chain.copy()
    result.columns = [str(column).strip().lower() for column in result.columns]
    missing = REQUIRED_COLUMNS - set(result.columns)
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"Missing required column(s): {names}. See the CSV format in the Learn tab.")

    option_types = result["option_type"].astype(str).str.strip().str.title()
    invalid_types = ~option_types.isin(["Call", "Put"])
    if invalid_types.any():
        rows = ", ".join(str(index + 2) for index in result.index[invalid_types][:5])
        raise ValueError(f"option_type must be Call or Put (check CSV row(s): {rows}).")
    result["option_type"] = option_types

    for column in ("strike", "market_price"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
        invalid = result[column].isna() | ~result[column].apply(math.isfinite) | (result[column] <= 0)
        if invalid.any():
            rows = ", ".join(str(index + 2) for index in result.index[invalid][:5])
            raise ValueError(f"{column} must contain positive numbers (check CSV row(s): {rows}).")

    return result


def analyze_chain(
    chain: pd.DataFrame, spot: float, time: float, rate: float, dividend: float, volatility: float
) -> pd.DataFrame:
    result = validate_chain(chain)
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
    figure.add_hline(y=0, line_dash="dot", line_color="#64748b")
    figure.add_vline(x=strike, line_dash="dot", line_color="#f59e0b", annotation_text="Strike")
    figure.update_layout(
        template="plotly_white",
        margin={"l": 10, "r": 10, "t": 20, "b": 10},
        height=390,
        xaxis_title="Underlying price at expiry",
        yaxis_title="Profit / loss per share",
        hovermode="x unified",
    )
    return figure


st.set_page_config(page_title="Option Chain Analyzer", page_icon="📈", layout="wide")
st.markdown(
    """
    <style>
    :root { --ink:#172033; --muted:#5b667a; --line:#dbe3ee; --navy:#12304a; --teal:#0f766e; }
    .block-container { max-width: 1240px; padding-top: 2rem; padding-bottom: 3rem; }
    h1, h2, h3 { color: var(--ink); letter-spacing: -0.02em; }
    h1 { font-size: 2.35rem; margin-bottom: .25rem; }
    .app-subtitle { color: var(--muted); font-size: 1.05rem; margin-bottom: 1.5rem; }
    .hero { background: #eef6f5; border: 1px solid #c9e5e0; border-radius: 12px; padding: 1rem 1.2rem; margin: .5rem 0 1.5rem; }
    .hero strong { color: var(--navy); }
    .section-note { color: var(--muted); font-size: .92rem; margin-top: -.55rem; margin-bottom: 1rem; }
    [data-testid="stMetric"] { background: #f8fafc; border: 1px solid var(--line); padding: .8rem 1rem; border-radius: 10px; }
    [data-testid="stMetricLabel"] { color: var(--muted); }
    .disclaimer { color: #667085; font-size: .82rem; border-top: 1px solid var(--line); padding-top: 1rem; margin-top: 2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Option Chain Analyzer")
st.markdown(
    '<div class="app-subtitle">A calm, transparent way to compare market quotes with Black–Scholes estimates.</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="hero"><strong>Research mode:</strong> This app explains model output and paper-trading ideas only. '
    "It never places orders.</div>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("1. Set assumptions")
    st.caption("These inputs drive the model estimate for every quote.")
    spot = st.number_input("Underlying price", min_value=0.01, value=100.0, step=0.5)
    days = st.number_input("Days to expiry", min_value=1, max_value=3650, value=30, step=1)
    rate = st.number_input("Risk-free rate (%)", min_value=-25.0, max_value=100.0, value=5.0, step=0.1) / 100.0
    dividend = st.number_input("Dividend yield (%)", min_value=0.0, max_value=100.0, value=0.0, step=0.1) / 100.0
    volatility = st.number_input("Model volatility (%)", min_value=0.1, max_value=500.0, value=25.0, step=0.5) / 100.0
    threshold = st.slider("Signal threshold (%)", 1, 50, 5) / 100.0
    st.divider()
    st.header("2. Choose your data")
    uploaded = st.file_uploader("Upload option chain CSV", type="csv", help="Required: option_type, strike, market_price")
    st.caption("No file? A small demo chain is ready to explore.")

time = days / 365.0
chain = pd.read_csv(uploaded) if uploaded else sample_chain(spot, time, rate, volatility)

try:
    analyzed = analyze_chain(chain, spot, time, rate, dividend, volatility)
except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as error:
    st.error(str(error))
    st.stop()

analyzed["signal"] = analyzed["edge_pct"].apply(
    lambda value: "UNDERPRICED" if value >= threshold else "OVERPRICED" if value <= -threshold else "FAIR"
)

overview_tab, quotes_tab, payoff_tab, learn_tab = st.tabs(
    ["Overview", "Quote scanner", "Payoff explorer", "Learn the basics"]
)

with overview_tab:
    st.header("Your model snapshot")
    st.markdown(
        f'<p class="section-note">Based on a ${spot:,.2f} underlying, {int(days)} days to expiry, and '
        f'{volatility:.1%} model volatility.</p>',
        unsafe_allow_html=True,
    )
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Quotes analyzed", f"{len(analyzed):,}")
    col2.metric("Underpriced", f"{int((analyzed['signal'] == 'UNDERPRICED').sum()):,}")
    col3.metric("Overpriced", f"{int((analyzed['signal'] == 'OVERPRICED').sum()):,}")
    col4.metric("Average model edge", f"{analyzed['edge_pct'].mean():.2%}")

    st.subheader("What does this mean?")
    st.write(
        f"The model estimates a theoretical value for each quote. A quote is **underpriced** when the "
        f"model is at least {threshold:.0%} above its market price, **overpriced** when it is at least "
        f"{threshold:.0%} below, and **fair** when the difference falls in between."
    )
    st.info("Start with the Quote scanner to filter the chain, then use Payoff explorer to understand one quote at expiry.")

with quotes_tab:
    st.header("Quote scanner")
    st.markdown('<p class="section-note">Filter the chain to focus on the quotes you want to understand.</p>', unsafe_allow_html=True)
    filter_col1, filter_col2, filter_col3 = st.columns([1, 1, 2])
    signal_filter = filter_col1.multiselect("Signal", ["UNDERPRICED", "FAIR", "OVERPRICED"], default=["UNDERPRICED", "FAIR", "OVERPRICED"])
    type_filter = filter_col2.multiselect("Option type", ["Call", "Put"], default=["Call", "Put"])
    strike_min, strike_max = float(analyzed["strike"].min()), float(analyzed["strike"].max())
    strike_filter = filter_col3.slider("Strike range", min_value=strike_min, max_value=strike_max, value=(strike_min, strike_max))
    filtered = analyzed[
        analyzed["signal"].isin(signal_filter)
        & analyzed["option_type"].isin(type_filter)
        & analyzed["strike"].between(*strike_filter)
    ]
    st.caption(f"Showing {len(filtered):,} of {len(analyzed):,} quotes")

    display_columns = [
        column
        for column in [
            "symbol", "option_type", "strike", "market_price", "model_price",
            "implied_volatility", "edge_pct", "signal", "Delta", "Gamma", "Vega", "Theta", "Rho",
        ]
        if column in filtered.columns
    ]
    if filtered.empty:
        st.warning("No quotes match these filters. Widen the range or select another signal.")
    else:
        st.dataframe(
            filtered[display_columns].style.format(
                {
                    "market_price": "{:.2f}", "model_price": "{:.2f}", "implied_volatility": "{:.2%}",
                    "edge_pct": "{:.2%}", "Delta": "{:.4f}", "Gamma": "{:.6f}",
                    "Vega": "{:.4f}", "Theta": "{:.4f}", "Rho": "{:.4f}",
                },
                na_rep="n/a",
            ),
            use_container_width=True,
            hide_index=True,
        )
        st.download_button(
            "Download analyzed CSV",
            data=filtered.to_csv(index=False).encode("utf-8"),
            file_name="analyzed-option-chain.csv",
            mime="text/csv",
        )

with payoff_tab:
    st.header("Payoff explorer")
    st.markdown('<p class="section-note">See how one long option position changes value at expiry. This is not a trade recommendation.</p>', unsafe_allow_html=True)
    quote_options = analyzed.index.tolist()
    selected = st.selectbox(
        "Choose a quote",
        quote_options,
        format_func=lambda index: (
            f"{analyzed.loc[index, 'option_type']} {analyzed.loc[index, 'strike']:.2f} "
            f"at ${analyzed.loc[index, 'market_price']:.2f} · {analyzed.loc[index, 'signal']}"
        ),
    )
    selected_row = analyzed.loc[selected]
    chart_col, detail_col = st.columns([2.2, 1])
    with chart_col:
        st.plotly_chart(
            payoff_chart(
                spot,
                float(selected_row["strike"]),
                float(selected_row["market_price"]),
                str(selected_row["option_type"]),
            ),
            use_container_width=True,
        )
    with detail_col:
        st.subheader("Selected quote")
        st.metric("Market price", f"${selected_row['market_price']:.2f}")
        st.metric("Model price", f"${selected_row['model_price']:.2f}")
        st.metric("Model edge", f"{selected_row['edge_pct']:.2%}")
        st.write(
            f"At expiry, this long {str(selected_row['option_type']).lower()} starts profitable above "
            f"the strike plus premium (call) or below the strike minus premium (put)."
        )

with learn_tab:
    st.header("Learn the basics")
    st.markdown(
        """
        **Black–Scholes price** is a theoretical estimate based on the assumptions in the sidebar. It is a useful
        baseline, not a promise of what an option is worth.

        **Implied volatility** is the volatility that would make the model price equal the market price. “n/a” means
        the quote is outside the model's solvable price range.

        **Greeks** describe sensitivity: Delta tracks a small underlying move, Gamma tracks how Delta changes,
        Vega tracks volatility changes, Theta tracks one day of time decay, and Rho tracks rate changes.

        **Model edge** is `model price - market price`. It is shown as a percentage of market price and drives the
        configurable signal labels. Fees, spreads, liquidity, early exercise, volatility skew, and corporate actions
        are not modeled.

        **CSV format:** include `option_type`, `strike`, and `market_price`. `symbol`, `bid`, and `ask` are optional
        and will be preserved. Option type must be `Call` or `Put`; prices and strikes must be positive numbers.
        """
    )

st.markdown(
    '<div class="disclaimer">For education and research only. This is a paper-trading analysis tool, not financial advice, '
    "and it does not execute orders.</div>",
    unsafe_allow_html=True,
)
