# Option Chain Analyzer

Standalone Streamlit prototype for analyzing an option chain with Black–Scholes pricing.

## Run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

The app starts with a generated demo chain. Upload a CSV with these required columns:

```text
option_type,strike,market_price
Call,100,4.25
Put,100,3.80
```

Optional columns such as `symbol`, `bid`, and `ask` are preserved and displayed.

## Interpretation

`model_price - market_price` is the model edge. The signal uses the configurable percentage threshold:

- `UNDERPRICED`: model price is above market price
- `OVERPRICED`: model price is below market price
- `FAIR`: difference is within the threshold

This is a research and paper-trading tool. It does not place orders and does not account for all real-world factors such as bid/ask execution, American exercise, volatility skew, corporate actions, or liquidity.
