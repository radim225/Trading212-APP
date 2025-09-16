# app.py — Trading 212 Portfolio Summary
# Shows current holdings, values in CZK, and cash balance

import os
import time
import requests
import pandas as pd
import streamlit as st
from typing import Dict, Any, Optional, List

# ---------------------------- UI setup ----------------------------
st.set_page_config(page_title="Trading 212 Portfolio", layout="wide")
st.title("📊 Trading 212 Portfolio")
st.caption("View your current holdings and portfolio value")

def _as_float(x) -> float:
    try:
        return float(x)
    except (ValueError, TypeError):
        return 0.0

# ---------------------------- helpers ----------------------------
def now_utc() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")

def _fmtccy(x, ccy="") -> str:
    try:
        return f"{float(x):,.2f} {ccy}".strip()
    except Exception:
        return str(x)

def _as_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return np.nan

def _safe_get(d: Dict[str, Any], *keys, default=None):
    cur = d
    for k in keys:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur

def _as_list(obj: Any) -> List[Dict[str, Any]]:
    """
    Normalize any T212 API response shape (list/dict) to a list of dict rows.
    - If dict, look for common container keys and return their list.
    - Else return [].
    """
    if obj is None:
        return []
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        for k in ("items", "dividends", "transactions", "orders", "positions",
                  "portfolio", "data", "results"):
            v = obj.get(k)
            if isinstance(v, list):
                return v
        # fallback: first list value
        for v in obj.values():
            if isinstance(v, list):
                return v
        return []
    return []

class T212Client:
    def __init__(self, api_key: str, is_demo: bool = False):
        self.session = requests.Session()
        self.session.headers.update({"Authorization": api_key})
        self.base = (
            "https://demo.trading212.com/api/v0" if is_demo 
            else "https://live.trading212.com/api/v0"
        )

    def _get(self, endpoint: str):
        try:
            response = self.session.get(f"{self.base}{endpoint}", timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"Error fetching {endpoint}: {str(e)}")
            return None

    def get_cash_balance(self):
        return self._get("/equity/account/cash")

    def get_positions(self):
        return self._get("/equity/portfolio")

    def get_account_info(self):
        return self._get("/equity/account/info")

# ---------------------------- finance utils ----------------------------
def xirr(cashflows: List[Tuple[pd.Timestamp, float]], guess: float = 0.1, tol=1e-6, max_iter=100) -> float:
    """
    XIRR for irregular cash flows.
    We add a terminal -NAV to close the position at 'today'.
    """
    cfs = [(pd.Timestamp(d, tz="UTC"), float(a)) for d, a in cashflows if not np.isclose(a, 0.0)]
    if not cfs:
        return np.nan
    cfs.sort(key=lambda x: x[0])
    t0 = cfs[0][0]

    def f(rate):
        return sum(a / ((1 + rate) ** ((d - t0).days / 365.25)) for d, a in cfs)

    r = guess
    for _ in range(max_iter):
        h = 1e-6
        f0 = f(r)
        f1 = f(r + h)
        der = (f1 - f0) / h
        if abs(der) < 1e-12:
            break
        nr = r - f0 / der
        if abs(nr - r) < tol:
            r = nr
            break
        r = nr
    return r

def geometric_link(returns: pd.Series) -> float:
    returns = returns.dropna()
    if returns.empty:
        return np.nan
    return float(np.exp(np.log1p(returns).sum()) - 1)

def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return np.nan
    cummax = equity.cummax()
    dd = (equity / cummax) - 1.0
    return float(dd.min())

def annualized_vol(returns: pd.Series, periods_per_year=252) -> float:
    returns = returns.dropna()
    if returns.empty:
        return np.nan
    return float(returns.std() * (periods_per_year ** 0.5))

def sharpe_ratio(returns: pd.Series, rf_rate_annual: float = 0.0, periods_per_year=252) -> float:
    returns = returns.dropna()
    if returns.empty:
        return np.nan
    mu = (returns.mean() - (rf_rate_annual / periods_per_year)) * periods_per_year
    sigma = returns.std() * (periods_per_year ** 0.5)
    return float(mu / sigma) if sigma else np.nan

# ---------------------------- transformers ----------------------------
def df_from_positions(data: Any) -> pd.DataFrame:
    rows = []
    for it in _as_list(data):
        if not isinstance(it, dict):
            continue
        avg = it.get("averagePrice")
        cur = it.get("currentPrice")
        rows.append({
            "ticker": it.get("ticker") or it.get("symbol"),
            "name": it.get("name") or it.get("shortName"),
            "quantity": _as_float(it.get("quantity")),
            "avg_price": _as_float(avg.get("value") if isinstance(avg, dict) else avg),
            "avg_price_ccy": (avg or {}).get("currencyCode") if isinstance(avg, dict) else it.get("currencyCode"),
            "current_price": _as_float(cur.get("value") if isinstance(cur, dict) else cur),
            "current_price_ccy": (cur or {}).get("currencyCode") if isinstance(cur, dict) else it.get("currencyCode"),
            "pnl": _as_float(it.get("pnl") or it.get("unrealizedPl")),
            "isin": it.get("isin"),
            "type": it.get("type"),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["current_value_native"] = df["quantity"].fillna(0) * df["current_price"].fillna(0)
    return df

def df_from_dividends(data: Any) -> pd.DataFrame:
    rows = []
    for it in _as_list(data):
        if not isinstance(it, dict):
            continue
        amt = it.get("amount")
        amount = None
        ccy = None
        
        if isinstance(amt, dict):
            amount = _as_float(amt.get("value"))
            ccy = amt.get("currencyCode")
        elif amt is not None:  # Handle case where amt is a string or number
            amount = _as_float(amt)
            
        ccy = ccy or it.get("currencyCode") or account_ccy
        
        rows.append({
            "paidOn": pd.to_datetime(it.get("paidOn") or it.get("date"), utc=True, errors="coerce"),
            "ticker": it.get("ticker"),
            "name": it.get("name"),
            "amount": amount or 0.0,
            "currencyCode": ccy
        })
    
    if not rows:
        return pd.DataFrame(columns=["paidOn", "ticker", "name", "amount", "currencyCode"])
        
    df = pd.DataFrame(rows)
    return df.sort_values("paidOn") if not df.empty else df

def df_from_transactions(data: Any) -> pd.DataFrame:
    rows = []
    for it in _as_list(data):
        if not isinstance(it, dict):
            continue
        amt = it.get("amount")
        if isinstance(amt, dict):
            amount = _as_float(amt.get("value"))
            ccy    = amt.get("currencyCode") or it.get("currencyCode")
        else:
            amount = _as_float(amt)
            ccy    = it.get("currencyCode")
        rows.append({
            "date": pd.to_datetime(it.get("date") or it.get("time"), utc=True, errors="coerce"),
            "type": it.get("type"),
            "amount": amount,
            "currencyCode": ccy,
            "description": it.get("description") or it.get("reason")
        })
    df = pd.DataFrame(rows)
    return df.sort_values("date") if not df.empty else df

def df_from_orders(data: Any) -> pd.DataFrame:
    rows = []
    for it in _as_list(data):
        if not isinstance(it, dict):
            continue
        fp = it.get("fillPrice") or it.get("averagePrice")
        if isinstance(fp, dict):
            fp = fp.get("value")
        rows.append({
            "submittedAt": pd.to_datetime(it.get("submittedAt") or it.get("placedAt"), utc=True, errors="coerce"),
            "ticker": it.get("ticker"),
            "side": it.get("side"),
            "quantity": _as_float(it.get("quantity")),
            "avgFillPrice": _as_float(fp),
            "status": it.get("status"),
            "id": it.get("id"),
        })
    df = pd.DataFrame(rows)
    return df.sort_values("submittedAt") if not df.empty else df

# ---------------------------- prices (optional) ----------------------------
def guess_yahoo_symbol(row: pd.Series) -> Optional[str]:
    t = str(row.get("ticker") or "")
    base = t.split("_")[0] if "_" in t else t
    ccy = str(row.get("current_price_ccy") or row.get("avg_price_ccy") or "").upper()
    if ccy in ("GBP", "GBX") and not base.endswith(".L"):
        return base + ".L"
    return base or None

def fetch_price_history(y_symbols: List[str], start: str, end: str) -> pd.DataFrame:
    if yf is None or not y_symbols:
        return pd.DataFrame()
    df = yf.download(y_symbols, start=start, end=end, group_by="ticker", progress=False)["Adj Close"]
    if isinstance(df, pd.Series):
        df = df.to_frame(name=y_symbols[0])
    return df.ffill()

# ---------------------------- Sidebar ----------------------------
with st.sidebar:
    st.header("Settings")
    default_key = st.secrets.get("T212_API_KEY", os.environ.get("T212_API_KEY", ""))
    api_key = st.text_input("Trading 212 API key", type="password", value=default_key)
    is_demo = st.checkbox("Use Demo Account", value=False)
    refresh = st.button("🔄 Refresh Data")

if not api_key:
    st.info("🔑 Please add your Trading 212 API key in the sidebar")
    st.stop()

# Initialize client
client = T212Client(api_key, is_demo)

# Fetch data
with st.spinner("Fetching portfolio data..."):
    try:
        # Get cash balance
        cash_data = client.get_cash_balance()
        cash_balance = _as_float(cash_data.get("free", {}).get("value", 0))
        
        # Get positions
        positions = client.get_positions() or []
        
        # Process positions
        holdings = []
        total_value = 0
        
        for pos in positions:
            ticker = pos.get("ticker")
            name = pos.get("name", ticker)
            quantity = _as_float(pos.get("quantity"))
            
            # Get current price
            price_data = pos.get("currentPrice", {})
            price = _as_float(price_data.get("value"))
            
            # Get average price
            avg_price_data = pos.get("averagePrice", {})
            avg_price = _as_float(avg_price_data.get("value"))
            
            # Calculate values
            current_value = quantity * price
            invested_amount = quantity * avg_price
            pnl = current_value - invested_amount
            pnl_pct = (pnl / invested_amount * 100) if invested_amount else 0
            
            holdings.append({
                "Ticker": ticker,
                "Name": name,
                "Quantity": quantity,
                "Current Price (CZK)": price,
                "Avg. Price (CZK)": avg_price,
                "Current Value (CZK)": current_value,
                "Invested (CZK)": invested_amount,
                "P&L (CZK)": pnl,
                "P&L (%)": pnl_pct
            })
            
            total_value += current_value
        
        # Create DataFrame
        df_holdings = pd.DataFrame(holdings)
        
        # Display summary
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Portfolio Value (CZK)", f"{total_value:,.2f}")
        with col2:
            st.metric("Cash Balance (CZK)", f"{cash_balance:,.2f}")
        with col3:
            st.metric("Total Value (CZK)", f"{total_value + cash_balance:,.2f}")
        
        # Display holdings table
        st.subheader("Current Holdings")
        if not df_holdings.empty:
            # Format numbers
            format_dict = {
                'Current Price (CZK)': '{:,.2f}',
                'Avg. Price (CZK)': '{:,.2f}',
                'Current Value (CZK)': '{:,.2f}',
                'Invested (CZK)': '{:,.2f}',
                'P&L (CZK)': '{:,.2f}',
                'P&L (%)': '{:.2f}%',
                'Quantity': '{:,.2f}'
            }
            
            # Apply formatting
            for col, fmt in format_dict.items():
                if col in df_holdings.columns:
                    df_holdings[col] = df_holdings[col].apply(lambda x: fmt.format(x) if pd.notnull(x) else "")
            
            st.dataframe(
                df_holdings,
                column_config={
                    "Ticker": "Ticker",
                    "Name": "Company Name",
                    "Quantity": "Shares",
                    "Current Price (CZK)": st.column_config.NumberColumn("Current Price"),
                    "Avg. Price (CZK)": st.column_config.NumberColumn("Avg. Cost"),
                    "Current Value (CZK)": st.column_config.NumberColumn("Market Value"),
                    "Invested (CZK)": st.column_config.NumberColumn("Invested"),
                    "P&L (CZK)": st.column_config.NumberColumn("P&L"),
                    "P&L (%)": st.column_config.NumberColumn("P&L %")
                },
                hide_index=True,
                use_container_width=True
            )
        else:
            st.info("No positions found in your portfolio.")
            
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        st.info("Please check your API key and try again.")
        if st.checkbox("Show error details"):
            st.exception(e)

# ---------------------------- optional price analytics ----------------------------
if enable_prices and yf is not None:
    st.subheader("Price-powered analytics")
    start = (now_utc() - pd.DateOffset(years=lookback_years)).strftime("%Y-%m-%d")
    end = (now_utc() + pd.DateOffset(days=1)).strftime("%Y-%m-%d")

    manual_map: Dict[str, str] = {}
    if mapping_text.strip():
        for line in mapping_text.splitlines():
            if "," in line:
                t, y = [s.strip() for s in line.split(",", 1)]
                manual_map[t] = y

    df_pos["yahoo"] = df_pos.apply(
        lambda r: manual_map.get(str(r["ticker"]), guess_yahoo_symbol(r)), axis=1
    )
    syms = [s for s in df_pos["yahoo"].dropna().unique().tolist() if s]
    prices = fetch_price_history(syms, start, end)

    if prices.empty:
        st.warning("No price data resolved. Add mappings if needed.")
    else:
        qty_map = df_pos.set_index("yahoo")["quantity"].to_dict()
        common = [c for c in prices.columns if c in qty_map]
        val = prices[common].copy()
        for c in common:
            val[c] = val[c] * qty_map[c]
        equity = val.sum(axis=1)
        ret = equity.pct_change().dropna()

        twr = geometric_link(ret)
        mdd = max_drawdown(equity)
        vol = annualized_vol(ret)
        shp = sharpe_ratio(ret, rf_rate_annual=risk_free)

        if benchmark:
            b = fetch_price_history([benchmark], start, end)
            if not b.empty:
                st.line_chart(pd.DataFrame({"Portfolio": equity, "Benchmark": b.iloc[:, 0]}).dropna())
            else:
                st.line_chart(equity.rename("Portfolio"))
        else:
            st.line_chart(equity.rename("Portfolio"))

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("TWR (lookback)", f"{twr*100:,.2f}%")
        k2.metric("Max drawdown", f"{mdd*100:,.2f}%")
        k3.metric("Volatility (ann.)", f"{vol*100:,.2f}%")
        k4.metric("Sharpe (ann.)", f"{shp:,.2f}")
elif enable_prices and yf is None:
    st.warning("yfinance not installed. Add it to requirements.txt to enable price analytics.")

# ---------------------------- detail tabs ----------------------------
st.markdown("---")
t1, t2, t3, t4, t5 = st.tabs(["Holdings", "Dividends", "Transactions", "Orders", "Pies"])

with t1:
    st.dataframe(df_pos if not df_pos.empty else pd.DataFrame({"info": ["No positions"]}))

with t2:
    st.dataframe(df_div if not df_div.empty else pd.DataFrame({"info": ["No dividends"]}))

with t3:
    st.dataframe(df_txs if not df_txs.empty else pd.DataFrame({"info": ["No transactions"]}))

with t4:
    st.dataframe(df_ord if not df_ord.empty else pd.DataFrame({"info": ["No orders"]}))

with t5:
    st.json(pies_raw or {})
