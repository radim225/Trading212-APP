# app.py — Trading 212 Streamlit Dashboard (read-only)
# Paste your API key in the sidebar or store it as Secret T212_API_KEY (or env var T212_API_KEY).

import os, time, math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

import requests
import pandas as pd
import numpy as np
import streamlit as st

# Optional: price/benchmark analytics
try:
    import yfinance as yf
except Exception:
    yf = None

# ---------------------------- UI setup ----------------------------
st.set_page_config(page_title="Trading 212 Dashboard", layout="wide")
st.title("📈 Trading 212 Portfolio Dashboard")
st.caption("Read-only. Enter your API key in the sidebar (or set Secret T212_API_KEY).")

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

# ---------------------------- API client ----------------------------
@dataclass
class T212Config:
    mode: str = "live"   # "live" or "demo"
    api_key: str = ""
    timeout: int = 30

class T212Client:
    def __init__(self, cfg: T212Config):
        self.cfg = cfg
        self.session = requests.Session()
        # Trading212 expects the API key directly in Authorization header (no 'Bearer' prefix)
        self.session.headers.update({"Authorization": cfg.api_key})
        self.base = (
            "https://live.trading212.com/api/v0"
            if cfg.mode == "live"
            else "https://demo.trading212.com/api/v0"
        )

    def _get(self, path: str, params: Optional[dict] = None, retry: int = 3):
        url = f"{self.base}{path}"
        err = None
        for i in range(retry):
            r = self.session.get(url, params=params, timeout=self.cfg.timeout)
            if r.status_code == 429:
                wait_s = int(r.headers.get("Retry-After", "5"))
                time.sleep(min(wait_s, 10))
                continue
            if r.ok:
                try:
                    return r.json()
                except Exception:
                    return None
            err = r
            time.sleep(1 + i)  # small backoff
        if err is not None:
            try:
                err.raise_for_status()
            except Exception as e:
                body = (err.text or "")[:300]
                raise RuntimeError(f"{e} | body={body}")
        return None

    # ---- Correct endpoint paths ----
    def get_account_cash(self):
        return self._get("/equity/account/cash")

    def get_account_info(self):
        return self._get("/equity/account/info")

    def get_positions(self):
        # Correct path (not /equity/portfolio/position)
        return self._get("/equity/portfolio")

    def get_orders_history(self, **params):
        return self._get("/equity/history/orders", params=params)

    def get_transactions(self, **params):
        # history endpoints under /history/...
        return self._get("/history/transactions", params=params)

    def get_dividends(self, **params):
        return self._get("/history/dividends", params=params)

    def get_pies(self):
        return self._get("/equity/pies")

    def get_instruments(self):
        return self._get("/equity/metadata/instruments")

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
        if isinstance(amt, dict):
            amount = _as_float(amt.get("value"))
            ccy    = amt.get("currencyCode") or it.get("currencyCode")
        else:
            amount = _as_float(amt)
            ccy    = it.get("currencyCode")
        rows.append({
            "paidOn": pd.to_datetime(it.get("paidOn") or it.get("date"), utc=True, errors="coerce"),
            "ticker": it.get("ticker"),
            "name":   it.get("name"),
            "amount": amount,
            "currencyCode": ccy
        })
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

# ---------------------------- sidebar ----------------------------
with st.sidebar:
    st.header("Settings")
    default_key = st.secrets.get("T212_API_KEY", os.environ.get("T212_API_KEY", ""))
    api_key = st.text_input("Trading 212 API key", type="password", value=default_key)
    mode = st.radio("Mode", ["live", "demo"], horizontal=True)
    risk_free = st.number_input("Risk-free rate (annual, %)", 0.0, 10.0, 0.0, step=0.25) / 100.0
    enable_prices = st.toggle("Enable price-powered analytics (optional)", value=False)
    benchmark = st.text_input("Benchmark (Yahoo)", value="^GSPC")
    lookback_years = st.slider("Price lookback (years)", 1, 10, 3)
    mapping_text = st.text_area("Ticker mapping lines  T212,Yahoo", value="", height=80)
    refresh = st.button("🔄 Refresh")

if not api_key:
    st.info("Add your API key in the sidebar (or set Secret T212_API_KEY).")
    st.stop()

cfg = T212Config(mode=mode, api_key=api_key)
client = T212Client(cfg)

# ---------------------------- cached fetch (hash-safe) ----------------------------
@st.cache_data(ttl=60, show_spinner=False)
def fetch_all(_client: T212Client):
    cash = _client.get_account_cash()
    pos  = _client.get_positions()
    ords = _client.get_orders_history()
    txs  = _client.get_transactions()
    dvs  = _client.get_dividends()
    pies = _client.get_pies()
    return cash, pos, ords, txs, dvs, pies

try:
    if refresh:
        fetch_all.clear()
    cash_raw, positions_raw, orders_raw, txs_raw, dividends_raw, pies_raw = fetch_all(client)
except Exception as e:
    st.error(f"API error: {e}")
    st.stop()

# ---------------------------- processing ----------------------------
account_ccy = (cash_raw or {}).get("currencyCode") or "EUR"
cash_val = _as_float(_safe_get(cash_raw, "cash", "value") or (cash_raw or {}).get("cash") or 0.0)

df_pos = df_from_positions(positions_raw)
df_div = df_from_dividends(dividends_raw)
df_txs = df_from_transactions(txs_raw)
df_ord = df_from_orders(orders_raw)

df_pos["value_native"] = df_pos.get("current_value_native", pd.Series(dtype=float)).fillna(0.0)
nav_positions = float(df_pos["value_native"].sum()) if not df_pos.empty else 0.0
nav_total = nav_positions + (cash_val or 0.0)

# XIRR from cashflows (treat positive amounts as inflows; add terminal -NAV)
cf = []
for _, r in (df_txs or pd.DataFrame()).iterrows():
    if pd.isna(r.get("date")) or pd.isna(r.get("amount")):
        continue
    cf.append((r["date"], r["amount"]))
cf.append((now_utc(), -nav_total))
xirr_all = xirr(cf) if len(cf) >= 2 else np.nan

# Dividends by month / TTM
if not df_div.empty:
    last12 = df_div[df_div["paidOn"] >= (now_utc() - pd.DateOffset(years=1))]
    ttm_div = float(last12["amount"].sum())
    dmon = df_div.copy()
    dmon["month"] = pd.to_datetime(dmon["paidOn"]).dt.to_period("M").dt.to_timestamp()
    div_by_month = dmon.groupby("month")["amount"].sum()
else:
    ttm_div = 0.0
    div_by_month = pd.Series(dtype=float)

# Allocation view
alloc = df_pos.copy()
if not alloc.empty and nav_total > 0:
    alloc["weight"] = alloc["value_native"] / nav_total
    alloc_view = alloc[["ticker", "name", "value_native", "weight"]].sort_values("weight", ascending=False)
else:
    alloc_view = pd.DataFrame()

# ---------------------------- overview ----------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("NAV (incl. cash)", _fmtccy(nav_total, account_ccy))
c2.metric("Cash", _fmtccy(cash_val, account_ccy))
c3.metric("Positions value", _fmtccy(nav_positions, account_ccy))
c4.metric("XIRR (since inception)", f"{xirr_all*100:,.2f}%" if pd.notna(xirr_all) else "—")

st.subheader("Holdings")
st.dataframe(alloc_view if not alloc_view.empty else pd.DataFrame({"info": ["No positions"]}))

st.subheader("Dividends")
if not div_by_month.empty:
    st.line_chart(div_by_month)
else:
    st.info("No dividends yet.")

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
    