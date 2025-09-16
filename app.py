# app.py
# Streamlit dashboard for Trading 212 (read-only)
# Enter your API key (or save it in Streamlit Secrets) and click Refresh.
# Optional: provide a mapping from Trading212 tickers to Yahoo Finance symbols to enable
# price-driven analytics (TWR, beta, drawdowns, benchmark).
#
# © You. MIT License

import os
import time
import math
import json
import datetime as dt
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import requests
import pandas as pd
import numpy as np
import streamlit as st

# Optional analytics (prices and benchmarks)
try:
    import yfinance as yf
except Exception:
    yf = None

# ----------------------------
# App configuration
# ----------------------------
st.set_page_config(page_title="Trading 212 Dashboard", layout="wide")

st.title("📈 Trading 212 Portfolio Dashboard")
st.caption("Read-only dashboard. Enter your API key in the sidebar.")

# ----------------------------
# Small utilities
# ----------------------------

def _to_datetime(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    try:
        return pd.to_datetime(x, utc=True)
    except Exception:
        return None

def _fmtccy(x, ccy=""):
    try:
        return f"{x:,.2f} {ccy}".strip()
    except Exception:
        return str(x)

def _as_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan

def _safe_get(d: dict, *keys, default=None):
    cur = d
    for k in keys:
        if isinstance(cur, dict) and (k in cur):
            cur = cur[k]
        else:
            return default
    return cur

# ----------------------------
# API client
# ----------------------------

@dataclass
class T212Config:
    mode: str = "live"  # 'live' or 'demo'
    api_key: str = ""
    timeout: int = 30

class T212Client:
    def __init__(self, cfg: T212Config):
        self.cfg = cfg
        self.session = requests.Session()
        self.session.headers.update({"Authorization": cfg.api_key})

        self.base = "https://live.trading212.com/api/v0" if cfg.mode == "live" else "https://demo.trading212.com/api/v0"

    def _get(self, path: str, params: dict = None, retry=3):
        url = f"{self.base}{path}"
        for i in range(retry):
            r = self.session.get(url, params=params, timeout=self.cfg.timeout)
            if r.status_code == 429:
                # Rate limit; respect Retry-After if present
                wait_s = int(r.headers.get("Retry-After", "5"))
                time.sleep(min(10, wait_s))
                continue
            if r.ok:
                try:
                    return r.json()
                except Exception:
                    return None
            # transient? backoff
            time.sleep(1 + i)
        r.raise_for_status()

    # Endpoints (subset used by app)
    def get_account_cash(self):
        return self._get("/equity/account/cash")

    def get_account_meta(self):
        return self._get("/equity/account")

    def get_positions(self):
        return self._get("/equity/portfolio/position")

    def get_orders_history(self, **params):
        return self._get("/equity/history/orders", params=params)

    def get_transactions(self, **params):
        return self._get("/equity/history/transactions", params=params)

    def get_dividends(self, **params):
        return self._get("/equity/history/dividends", params=params)

    def get_pies(self):
        return self._get("/equity/pies")

    def get_instruments(self):
        return self._get("/equity/metadata/instruments")


# ----------------------------
# Finance utilities (IRR/XIRR, TWR, risk)
# ----------------------------

def xirr(cashflows: List[Tuple[pd.Timestamp, float]], guess: float = 0.1, tol=1e-6, max_iter=100):
    """
    Compute XIRR for irregular cash flows.
    cashflows: list of (timestamp, amount), positive for inflow, negative for outflow.
    Returns annualized rate.
    """
    # filter and sort
    cfs = [(pd.Timestamp(d, tz="UTC"), float(a)) for d, a in cashflows if not np.isclose(a, 0.0)]
    if not cfs:
        return np.nan
    cfs.sort(key=lambda x: x[0])
    t0 = cfs[0][0]

    def npv(rate):
        return sum(a / ((1 + rate) ** ((d - t0).days / 365.25)) for d, a in cfs)

    r = guess
    for _ in range(max_iter):
        # derivative
        f = 0.0
        df = 0.0
        for d, a in cfs:
            t = (d - t0).days / 365.25
            denom = (1 + r) ** (t + 1e-12)
            f += a / denom
            df -= a * t / denom / (1 + r)
        if abs(df) < 1e-12:
            break
        nr = r - f / df
        if abs(nr - r) < tol:
            r = nr
            break
        r = nr
    return r

def geometric_link(returns: pd.Series) -> float:
    """Compound (1+r) and subtract 1."""
    returns = returns.dropna()
    if returns.empty:
        return np.nan
    return np.exp(np.log1p(returns).sum()) - 1

def max_drawdown(series: pd.Series) -> float:
    """Max drawdown from an equity curve series (index: date)."""
    if series.empty:
        return np.nan
    cummax = series.cummax()
    dd = (series - cummax) / cummax
    return dd.min()

def annualized_vol(returns: pd.Series, periods_per_year=252) -> float:
    returns = returns.dropna()
    if returns.empty:
        return np.nan
    return returns.std() * math.sqrt(periods_per_year)

def sharpe_ratio(returns: pd.Series, rf_rate_annual: float = 0.0, periods_per_year=252):
    returns = returns.dropna()
    if returns.empty:
        return np.nan
    excess = returns - (rf_rate_annual / periods_per_year)
    mu = excess.mean() * periods_per_year
    sigma = returns.std() * math.sqrt(periods_per_year)
    return mu / sigma if sigma else np.nan

# ----------------------------
# Data transformers
# ----------------------------

def df_from_positions(data: list, account_ccy: str) -> pd.DataFrame:
    # Attempt to normalize common fields; structures may vary slightly
    rows = []
    for it in data or []:
        row = {
            "ticker": it.get("ticker") or it.get("symbol"),
            "name": it.get("name") or it.get("shortName"),
            "quantity": float(it.get("quantity") or 0.0),
            "avg_price": float((it.get("averagePrice") or {}).get("value") if isinstance(it.get("averagePrice"), dict) else (it.get("averagePrice") or 0.0)),
            "avg_price_ccy": (it.get("averagePrice") or {}).get("currencyCode") if isinstance(it.get("averagePrice"), dict) else it.get("currencyCode"),
            "current_price": float((it.get("currentPrice") or {}).get("value") if isinstance(it.get("currentPrice"), dict) else (it.get("currentPrice") or 0.0)),
            "current_price_ccy": (it.get("currentPrice") or {}).get("currencyCode") if isinstance(it.get("currentPrice"), dict) else it.get("currencyCode"),
            "pnl": float(it.get("pnl") or it.get("unrealizedPl") or 0.0),
            "fxRate": float(it.get("fxRate") or 1.0),
            "type": it.get("type"),
            "isin": it.get("isin")
        }
        row["current_value_native"] = row["quantity"] * row["current_price"]
        rows.append(row)
    df = pd.DataFrame(rows)
    return df

def df_from_dividends(data: list) -> pd.DataFrame:
    rows = []
    for it in data or []:
        amt = it.get("amount")
        if isinstance(amt, dict):
            amount = float(amt.get("value") or 0.0)
            ccy = amt.get("currencyCode")
        else:
            amount = float(amt or 0.0)
            ccy = it.get("currencyCode")
        rows.append({
            "paidOn": pd.to_datetime(it.get("paidOn") or it.get("date"), utc=True),
            "ticker": it.get("ticker"),
            "name": it.get("name"),
            "amount": amount,
            "currencyCode": ccy
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("paidOn")
    return df

def df_from_transactions(data: list) -> pd.DataFrame:
    rows = []
    for it in data or []:
        amt = it.get("amount")
        if isinstance(amt, dict):
            amount = float(amt.get("value") or 0.0)
            ccy = amt.get("currencyCode")
        else:
            amount = float(amt or 0.0)
            ccy = it.get("currencyCode")
        rows.append({
            "date": pd.to_datetime(it.get("date") or it.get("time"), utc=True),
            "type": it.get("type"),
            "amount": amount,
            "currencyCode": ccy,
            "description": it.get("description") or it.get("reason")
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("date")
    return df

def df_from_orders(data: list) -> pd.DataFrame:
    rows = []
    for it in data or []:
        avgp = it.get("fillPrice") or it.get("averagePrice")
        if isinstance(avgp, dict):
            avgp = float(avgp.get("value") or 0.0)
        rows.append({
            "submittedAt": pd.to_datetime(it.get("submittedAt") or it.get("placedAt"), utc=True),
            "ticker": it.get("ticker"),
            "side": it.get("side"),
            "quantity": float(it.get("quantity") or 0.0),
            "avgFillPrice": float(avgp or 0.0),
            "status": it.get("status"),
            "id": it.get("id")
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("submittedAt")
    return df

# ----------------------------
# Price helpers (optional analytics)
# ----------------------------

def guess_yahoo_symbol(row: pd.Series) -> Optional[str]:
    """Heuristic: strip suffix like _US_EQ -> use left part. For LSE we may append .L if currency is GBP."""
    t = str(row.get("ticker") or "")
    if "_" in t:
        base = t.split("_")[0]
    else:
        base = t
    ccy = str(row.get("current_price_ccy") or row.get("avg_price_ccy") or "").upper()
    if ccy in ("GBX", "GBP") and not base.endswith(".L"):
        return base + ".L"
    return base or None

def fetch_price_history(y_symbols: List[str], start: str, end: str) -> pd.DataFrame:
    if yf is None or not y_symbols:
        return pd.DataFrame()
    df = yf.download(y_symbols, start=start, end=end, group_by="ticker", progress=False)["Adj Close"]
    if isinstance(df, pd.Series):
        df = df.to_frame(name=y_symbols[0])
    return df.ffill()

# ----------------------------
# Sidebar: configuration
# ----------------------------

with st.sidebar:
    st.header("Settings")
    default_key = st.secrets.get("T212_API_KEY", "")
    api_key = st.text_input("Trading 212 API key", type="password", value=default_key)
    mode = st.radio("Mode", ["live", "demo"], horizontal=True)
    risk_free = st.number_input("Risk-free rate (annual, %)", 0.0, 10.0, 0.0, step=0.25) / 100.0
    enable_prices = st.toggle("Enable price-powered analytics (optional)", value=False, help="Uses Yahoo Finance to compute TWR, volatility, drawdowns, beta, and benchmark comparison.")
    benchmark = st.text_input("Benchmark symbol (Yahoo)", value="^GSPC")
    price_lookback_years = st.slider("Price lookback (years)", 1, 10, 3)

    mapping_text = st.text_area("Optional: Mapping lines 'T212Ticker,YahooSymbol'", value="", height=100, help="Only needed if our heuristic guesses wrong. One mapping per line.")

    refresh = st.button("🔄 Refresh")

if not api_key:
    st.info("Add your API key in the sidebar to begin. You can find it in Trading 212 settings.")

cfg = T212Config(mode=mode, api_key=api_key)
client = T212Client(cfg)

@st.cache_data(ttl=60)
def fetch_all(client: T212Client):
    cash = client.get_account_cash()
    pos = client.get_positions()
    ords = client.get_orders_history()
    txs = client.get_transactions()
    dvs = client.get_dividends()
    pies = client.get_pies()
    return cash, pos, ords, txs, dvs, pies

data = None
if api_key:
    try:
        if refresh:
            fetch_all.clear()
        data = fetch_all(client)
    except Exception as e:
        st.error(f"API error: {e}")

# ----------------------------
# Main views
# ----------------------------

if data:
    cash_raw, positions_raw, orders_raw, txs_raw, dividends_raw, pies_raw = data

    # Account currency
    account_ccy = (cash_raw or {}).get("currencyCode") or "EUR"
    cash_amount = float(((cash_raw or {}).get("cash") or {}).get("value") if isinstance((cash_raw or {}).get("cash"), dict) else (cash_raw or {}).get("cash") or 0.0)

    # DataFrames
    df_pos = df_from_positions(positions_raw or [], account_ccy)
    df_div = df_from_dividends(dividends_raw or [])
    df_txs = df_from_transactions(txs_raw or [])
    df_ord = df_from_orders(orders_raw or [])

    # Compute current NAV from positions (native currency)
    df_pos["value_native"] = df_pos["current_value_native"].fillna(0.0)
    nav_positions = df_pos["value_native"].sum()
    nav_total = nav_positions + (cash_amount or 0.0)

    # Cashflow list for XIRR
    cf_rows = []
    for _, r in (df_txs or pd.DataFrame()).iterrows():
        if pd.isna(r["date"]) or pd.isna(r["amount"]):
            continue
        cf_rows.append((r["date"], r["amount"]))
    cf_rows.append((pd.Timestamp.utcnow().tz_localize("UTC"), -nav_total))
    portfolio_xirr = xirr(cf_rows) if len(cf_rows) >= 2 else np.nan

    # Dividends stats
    if not df_div.empty:
        last_12m = df_div[df_div["paidOn"] >= (pd.Timestamp.utcnow().tz_localize("UTC") - pd.DateOffset(years=1))]
        ttm_div = last_12m["amount"].sum()
        monthly = df_div.copy()
        monthly["month"] = monthly["paidOn"].dt.to_period("M").dt.to_timestamp()
        div_by_month = monthly.groupby("month")["amount"].sum().reset_index()
    else:
        ttm_div = 0.0
        div_by_month = pd.DataFrame(columns=["month", "amount"])

    # Allocation
    alloc = df_pos.copy()
    if not alloc.empty:
        alloc["weight"] = alloc["value_native"] / nav_total if nav_total else np.nan
        alloc_small = alloc[["ticker", "name", "value_native", "weight"]].sort_values("weight", ascending=False)

    # --------- Overview KPIs ---------
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("NAV (incl. cash)", f"{nav_total:,.2f} {account_ccy}")
    col2.metric("Cash", f"{cash_amount:,.2f} {account_ccy}")
    col3.metric("Positions value", f"{nav_positions:,.2f} {account_ccy}")
    col4.metric("XIRR (since inception)", f"{portfolio_xirr*100:,.2f}%" if not np.isnan(portfolio_xirr) else "—")

    st.subheader("Holdings (live)")
    st.dataframe(alloc_small if not alloc.empty else pd.DataFrame())

    st.subheader("Dividends (history)")
    st.line_chart(div_by_month.set_index("month")["amount"]) if not div_by_month.empty else st.info("No dividends yet.")

    # --------- Optional: price-driven analytics ---------
    if enable_prices and yf is not None:
        st.subheader("Price‑powered analytics")
        lookback_start = (pd.Timestamp.utcnow() - pd.DateOffset(years=price_lookback_years)).strftime("%Y-%m-%d")
        lookback_end = (pd.Timestamp.utcnow() + pd.DateOffset(days=1)).strftime("%Y-%m-%d")

        # Build mapping dict from text area
        manual_map = {}
        if mapping_text.strip():
            for line in mapping_text.splitlines():
                if "," in line:
                    t, y = [s.strip() for s in line.split(",", 1)]
                    manual_map[t] = y

        df_pos["yahoo"] = df_pos.apply(lambda r: manual_map.get(str(r["ticker"]), guess_yahoo_symbol(r)), axis=1)
        y_syms = [s for s in df_pos["yahoo"].dropna().unique().tolist() if s]

        if not y_syms:
            st.warning("No Yahoo symbols resolved. Add mappings in the sidebar to enable analytics.")
        else:
            prices = fetch_price_history(y_syms, start=lookback_start, end=lookback_end)
            if not prices.empty:
                pivot_qty = df_pos.set_index("yahoo")["quantity"].to_dict()
                common = [s for s in prices.columns if s in pivot_qty]
                val = prices[common].copy()
                for c in common:
                    val[c] = val[c] * pivot_qty[c]
                equity_curve = val.sum(axis=1)
                ret = equity_curve.pct_change().dropna()

                twr_total = float(np.exp(np.log1p(ret).sum()) - 1)
                mdd = float((equity_curve / equity_curve.cummax() - 1).min())
                vol = float(ret.std() * math.sqrt(252))
                shp = float(((ret - (risk_free/252)).mean() * 252) / (ret.std() * math.sqrt(252)) if ret.std() > 0 else np.nan)

                if benchmark:
                    b = fetch_price_history([benchmark], start=lookback_start, end=lookback_end)
                    if not b.empty:
                        bmk = b.iloc[:, 0]
                        st.line_chart(pd.DataFrame({"Portfolio": equity_curve, "Benchmark": bmk}).dropna())
                    else:
                        st.line_chart(equity_curve.rename("Portfolio"))
                else:
                    st.line_chart(equity_curve.rename("Portfolio"))

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("TWR (approx., lookback)", f"{twr_total*100:,.2f}%")
                c2.metric("Max Drawdown", f"{mdd*100:,.2f}%")
                c3.metric("Volatility (ann.)", f"{vol*100:,.2f}%")
                c4.metric("Sharpe (ann.)", f"{shp:,.2f}")

            else:
                st.warning("Could not fetch prices for selected period.")
    elif enable_prices and yf is None:
        st.warning("yfinance not installed. Add it to requirements.txt.")

    # --------- Detailed tabs ---------
    st.markdown("---")
    t1, t2, t3, t4, t5 = st.tabs(["Holdings", "Dividends", "Transactions", "Orders", "Pies"])

    with t1:
        st.dataframe(df_pos)

    with t2:
        st.dataframe(df_div if not df_div.empty else pd.DataFrame({"info": ["No dividends found"]}))

    with t3:
        st.dataframe(df_txs if not df_txs.empty else pd.DataFrame({"info": ["No transactions found"]}))

    with t4:
        st.dataframe(df_ord if not df_ord.empty else pd.DataFrame({"info": ["No orders found"]}))

    with t5:
        st.json(pies_raw or {})
else:
    st.stop()