import math
from typing import Optional

import numpy as np
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


@st.cache_data(show_spinner=False)
def analyze_chain(
    chain: pd.DataFrame, spot: float, time: float, rate: float, dividend: float, volatility: float
) -> pd.DataFrame:
    result = validate_chain(chain)
    strikes = result["strike"].to_numpy(dtype=float)
    option_types = result["option_type"].to_numpy()
    result["model_price"] = np.array(
        [
            model_price(spot, strike, time, rate, volatility, option_type, dividend)
            for strike, option_type in zip(strikes, option_types)
        ]
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


def payoff_chart(
    spot: float, strike: float, premium: float, option_type: str, position: str
) -> go.Figure:
    prices = np.linspace(spot * 0.5, spot * 1.5, 151)
    option_sign = 1 if option_type == "Call" else -1
    position_sign = 1 if position == "Long" else -1
    payoffs = position_sign * (np.maximum(option_sign * (prices - strike), 0.0) - premium)
    break_even = strike + premium if option_type == "Call" else strike - premium
    figure = go.Figure(
        go.Scatter(x=prices, y=payoffs, mode="lines", name=f"{position} {option_type} P/L",
                   line={"color": "#0f766e", "width": 3})
    )
    figure.add_hline(y=0, line_dash="dot", line_color="#64748b")
    figure.add_vline(x=strike, line_dash="dot", line_color="#f59e0b", annotation_text="Strike")
    figure.add_vline(x=spot, line_dash="dash", line_color="#12304a", annotation_text="Spot")
    figure.add_vline(x=break_even, line_dash="dot", line_color="#7c3aed", annotation_text="Break-even")
    figure.update_layout(
        template="plotly_white",
        margin={"l": 10, "r": 10, "t": 20, "b": 10},
        height=390,
        xaxis_title="Underlying price at expiry",
        yaxis_title="Profit / loss per share",
        hovermode="x unified",
    )
    return figure


def opportunity_chart(analyzed: pd.DataFrame) -> go.Figure:
    figure = go.Figure()
    colors = {"UNDERPRICED": "#0f766e", "OVERPRICED": "#c2410c", "FAIR": "#64748b"}
    for option_type in ("Call", "Put"):
        subset = analyzed[analyzed["option_type"] == option_type]
        for signal, group in subset.groupby("signal", sort=False):
            figure.add_trace(
                go.Scatter(
                    x=group["strike"],
                    y=group["edge_pct"],
                    mode="markers",
                    name=f"{option_type} · {signal.title()}",
                    marker={"color": colors[signal], "size": 10, "symbol": "circle" if option_type == "Call" else "diamond"},
                    customdata=group[["option_type", "market_price", "model_price", "implied_volatility"]].fillna(0).to_numpy(),
                    hovertemplate=(
                        "<b>%{customdata[0]} %{x:.2f}</b><br>"
                        "Market $%{customdata[1]:.2f}<br>Model $%{customdata[2]:.2f}<br>"
                        "IV %{customdata[3]:.1%}<br>Edge %{y:.2%}<extra></extra>"
                    ),
                )
            )
    figure.add_hline(y=0, line_dash="dot", line_color="#94a3b8")
    figure.update_layout(
        template="plotly_white", height=390, margin={"l": 10, "r": 10, "t": 20, "b": 10},
        xaxis_title="Strike", yaxis_title="Model edge (% of market price)",
        legend={"orientation": "h", "y": 1.12}, hovermode="closest",
    )
    return figure


def volatility_smile_chart(analyzed: pd.DataFrame) -> go.Figure:
    figure = go.Figure()
    for option_type, color in (("Call", "#0f766e"), ("Put", "#7c3aed")):
        subset = analyzed[(analyzed["option_type"] == option_type) & analyzed["implied_volatility"].notna()]
        figure.add_trace(
            go.Scatter(
                x=subset["strike"], y=subset["implied_volatility"], mode="lines+markers",
                name=option_type, line={"color": color, "width": 2},
                hovertemplate=f"{option_type} · Strike %{{x:.2f}}<br>Implied vol %{{y:.1%}}<extra></extra>",
            )
        )
    figure.update_layout(
        template="plotly_white", height=350, margin={"l": 10, "r": 10, "t": 20, "b": 10},
        xaxis_title="Strike", yaxis_title="Implied volatility", hovermode="x unified",
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
    .eyebrow { color: var(--teal); text-transform: uppercase; letter-spacing: .12em; font-size: .72rem; font-weight: 700; margin-bottom: .2rem; }
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
    ["Market map", "Quote scanner", "Scenario lab", "Learn the basics"]
)

with overview_tab:
    st.markdown('<div class="eyebrow">Live market map</div>', unsafe_allow_html=True)
    st.header("Find the interesting quotes")
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

    chart_col, list_col = st.columns([1.8, 1])
    with chart_col:
        st.plotly_chart(opportunity_chart(analyzed), use_container_width=True, config={"displaylogo": False})
    with list_col:
        st.subheader("Top model edges")
        top_edges = analyzed.nlargest(5, "edge_pct")[["option_type", "strike", "edge_pct", "signal"]].copy()
        top_edges["strike"] = top_edges["strike"].map(lambda value: f"${value:,.2f}")
        top_edges["edge_pct"] = top_edges["edge_pct"].map(lambda value: f"{value:.2%}")
        st.dataframe(top_edges, hide_index=True, use_container_width=True)
        st.caption("Teal points are model-underpriced; orange points are model-overpriced. Circles are calls, diamonds are puts.")

    smile_col, explainer_col = st.columns([1.4, 1])
    with smile_col:
        st.subheader("Implied volatility shape")
        st.plotly_chart(volatility_smile_chart(analyzed), use_container_width=True, config={"displaylogo": False})
    with explainer_col:
        st.subheader("How to read this")
        st.write(
            f"The opportunity map plots each quote's model edge. A point at +10% means the model estimates "
            f"a value 10% above the market quote. The signal threshold is currently {threshold:.0%}."
        )
        st.info("Use the scanner to narrow the chain, then open Scenario lab for a selected quote.")

    st.subheader("What does this mean?")
    st.write(
        f"The model estimates a theoretical value for each quote. A quote is **underpriced** when the "
        f"model is at least {threshold:.0%} above its market price, **overpriced** when it is at least "
        f"{threshold:.0%} below, and **fair** when the difference falls in between."
    )
    st.info("This is a research signal, not a trade recommendation. Spreads, liquidity, exercise style, and skew can change the real-world result.")

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
    st.markdown('<div class="eyebrow">Interactive what-if</div>', unsafe_allow_html=True)
    st.header("Scenario lab")
    st.markdown('<p class="section-note">Change one assumption at a time and see how the selected position responds. Nothing here executes a trade.</p>', unsafe_allow_html=True)
    quote_options = analyzed.index.tolist()
    selected = st.selectbox(
        "Choose a quote to stress-test",
        quote_options,
        format_func=lambda index: (
            f"{analyzed.loc[index, 'option_type']} {analyzed.loc[index, 'strike']:.2f} "
            f"at ${analyzed.loc[index, 'market_price']:.2f} · {analyzed.loc[index, 'signal']}"
        ),
    )
    selected_row = analyzed.loc[selected]
    control_col, detail_col = st.columns([1, 2.2])
    with control_col:
        position = st.radio("Position", ["Long", "Short"], horizontal=True)
        scenario_spot = st.slider(
            "Underlying at expiry",
            min_value=float(spot * 0.5),
            max_value=float(spot * 1.5),
            value=float(spot),
            step=max(float(spot) / 100, 0.01),
        )
        scenario_volatility = st.slider("Volatility scenario (%)", 1, 200, int(volatility * 100), step=1) / 100
        scenario_days = st.slider("Days remaining", 1, int(days), int(days), step=1)
        scenario_model_price = model_price(
            scenario_spot, float(selected_row["strike"]), scenario_days / 365, rate,
            scenario_volatility, str(selected_row["option_type"]), dividend,
        )
        st.metric("Scenario model price", f"${scenario_model_price:.2f}")
    with detail_col:
        st.subheader("Expiry payoff")
        st.plotly_chart(
            payoff_chart(
                spot,
                float(selected_row["strike"]),
                float(selected_row["market_price"]),
                str(selected_row["option_type"]),
                position,
            ),
            use_container_width=True,
            config={"displaylogo": False},
        )
    break_even = (
        float(selected_row["strike"]) + float(selected_row["market_price"])
        if str(selected_row["option_type"]) == "Call"
        else float(selected_row["strike"]) - float(selected_row["market_price"])
    )
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    metric_col1.metric("Break-even", f"${break_even:,.2f}")
    metric_col2.metric("Current delta", f"{selected_row['Delta']:.3f}")
    metric_col3.metric("Current theta / day", f"{selected_row['Theta']:.4f}")
    metric_col4.metric("Current vega / 1%", f"{selected_row['Vega']:.4f}")
    st.caption(
        f"At the selected expiry price of ${scenario_spot:,.2f}, the estimated payoff is shown at expiry. "
        f"Changing volatility and days updates the scenario model price above."
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
