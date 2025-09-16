# Trading 212 Streamlit Dashboard (Read‑Only)

This app pulls your **cash, positions, orders, transactions and dividends** from the Trading 212 Public API (read‑only) and computes portfolio KPIs. Optionally it uses Yahoo Finance to compute TWR, drawdowns, volatility, Sharpe, and compare to a benchmark.

## Deploy (free, Streamlit Community Cloud)
1. Put `app.py` and `requirements.txt` in a GitHub repo.
2. Go to https://share.streamlit.io, log in with GitHub and **Deploy an app** pointing to your repo (`app.py`).
3. In the Streamlit app sidebar, paste your Trading 212 API key (or add it as a Secret named `T212_API_KEY`).

## API Scopes to enable
- Account data
- Portfolio
- History – Transactions
- History – Orders
- History – Dividends
- Metadata (for instrument info)
- Pies – Read (optional)

## Notes
- The app respects Trading 212 rate limits with basic backoff. If you hit 429, wait and retry.
- Price‑powered analytics are optional; if Yahoo symbols don’t match your Trading212 tickers, add mappings in the sidebar (e.g., `VUSA_UK_ETF,VUSA.L`).