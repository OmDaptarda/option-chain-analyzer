# Option Chain Analyzer

A beginner-friendly Streamlit research tool for comparing option-chain quotes with Black–Scholes estimates. It includes implied volatility, Greeks, configurable model-edge signals, an interactive opportunity map, volatility smile, filters, and a scenario lab with expiry payoff analysis.

This is a **research and paper-trading tool only**. It does not place orders or connect to a brokerage.

## Run locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

The app opens with a generated demo chain, so it is usable without a data file. Use the sidebar to change assumptions or upload your own CSV.

## Explore the app

- **Market map:** scan model edge by strike, compare calls and puts, review the top model edges, and inspect the implied-volatility shape.
- **Quote scanner:** filter by signal, option type, and strike range, then download the filtered analysis.
- **Scenario lab:** choose a quote and stress-test long/short position, underlying price at expiry, volatility, and time remaining. Break-even and key Greeks stay visible beside the payoff chart.
- **Learn the basics:** see the model assumptions, signal calculation, and CSV rules in plain English.

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub (the app entry point is `app.py`).
2. Sign in at [share.streamlit.io](https://share.streamlit.io/) with GitHub.
3. Select **Create app**, then choose this repository and the branch to deploy.
4. Set **Main file path** to `app.py`.
5. Select **Deploy**. Streamlit Cloud installs the packages listed in `requirements.txt` automatically.
6. After deployment, open **Settings → General** to set a custom app URL or manage access.

No secrets are required. If you later add a data provider, store credentials in Streamlit Cloud **Secrets** rather than committing them.

## CSV format

The upload must include these columns (case and surrounding whitespace are ignored):

```csv
option_type,strike,market_price
Call,100,4.25
Put,100,3.80
```

`option_type` must be `Call` or `Put`. `strike` and `market_price` must be positive numbers. Optional columns such as `symbol`, `bid`, and `ask` are preserved in the analyzed download.

## Reading the output

- **Model price:** theoretical Black–Scholes estimate from the sidebar assumptions.
- **Implied volatility:** volatility that makes the model equal the market price; `n/a` means no solution in the model range.
- **Model edge:** `model price - market price`, shown as a percentage of market price.
- **Signals:** `UNDERPRICED`, `OVERPRICED`, or `FAIR` based on the configurable threshold.
- **Greeks:** Delta, Gamma, Vega, Theta, and Rho sensitivities.

The model does not account for fees, bid/ask execution, liquidity, American exercise, volatility skew, or corporate actions. Outputs are educational estimates, not financial advice.
