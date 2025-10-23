"""
Trading212 CSV Parser
Parses Trading212 exported CSV and builds invested time series in CZK.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict

CSV_COLUMNS = {
    "Action": "action",
    "Time": "time",
    "ISIN": "isin",
    "Ticker": "ticker",
    "Name": "name",
    "Notes": "notes",
    "ID": "id",
    "No. of shares": "shares",
    "Price / share": "price",
    "Currency (Price / share)": "price_ccy",
    "Exchange rate": "fx_rate",
    "Result": "result",
    "Currency (Result)": "result_ccy",
    "Total": "total",
    "Currency (Total)": "total_ccy",
    "Withholding tax": "withholding_tax",
    "Currency (Withholding tax)": "withholding_tax_ccy",
    "Currency conversion from amount": "cc_from_amount",
    "Currency (Currency conversion from amount)": "cc_from_ccy",
    "Currency conversion to amount": "cc_to_amount",
    "Currency (Currency conversion to amount)": "cc_to_ccy",
    "Currency conversion fee": "cc_fee",
    "Currency (Currency conversion fee)": "cc_fee_ccy",
    "Merchant name": "merchant_name",
    "Merchant category": "merchant_category",
}


def _to_num(x):
    try:
        if pd.isna(x):
            return np.nan
        return float(str(x).replace(",", ""))
    except Exception:
        return np.nan


def normalize_csv(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns, parse types."""
    # Rename
    cols = {c: CSV_COLUMNS.get(c, c) for c in df.columns}
    df = df.rename(columns=cols).copy()

    # Parse time
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], errors="coerce")

    # Numeric fields
    for col in [
        "shares",
        "price",
        "fx_rate",
        "result",
        "total",
        "withholding_tax",
        "cc_from_amount",
        "cc_to_amount",
        "cc_fee",
    ]:
        if col in df.columns:
            df[col] = df[col].apply(_to_num)

    # Standardize action
    if "action" in df.columns:
        df["action_std"] = (
            df["action"]
            .str.lower()
            .str.strip()
            .str.replace("market buy", "BUY")
            .str.replace("market sell", "SELL")
            .str.replace("dividend", "DIVIDEND")
            .str.replace("currency conversion", "FX_CONV")
        )

    return df


def convert_row_to_czk(row: pd.Series) -> float:
    """
    Convert the row's Total to CZK.
    Priority:
    1) If total_ccy == 'CZK' -> use total
    2) Else if fx_rate present -> assume fx_rate is quoted as foreign->CZK inverse; use total / fx_rate
    3) Else return NaN (will try to match FX conversion rows separately)
    """
    total = row.get("total", np.nan)
    total_ccy = row.get("total_ccy")
    fx_rate = row.get("fx_rate", np.nan)

    if pd.notna(total) and str(total_ccy).upper() == "CZK":
        return float(total)

    if pd.notna(total) and pd.notna(fx_rate) and fx_rate not in (0, 1):
        try:
            return float(total) / float(fx_rate)
        except Exception:
            return np.nan

    return np.nan


def attach_czk_amounts(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns total_czk and result_czk using available info."""
    df = df.copy()
    df["total_czk"] = df.apply(convert_row_to_czk, axis=1)

    # Result (realized P&L)
    def _result_czk(row):
        res = row.get("result", np.nan)
        if pd.isna(res):
            return np.nan
        ccy = str(row.get("result_ccy", "")).upper()
        fx = row.get("fx_rate", np.nan)
        if ccy == "CZK":
            return float(res)
        if pd.notna(fx) and fx not in (0, 1):
            try:
                return float(res) / float(fx)
            except Exception:
                return np.nan
        return np.nan

    df["result_czk"] = df.apply(_result_czk, axis=1)
    return df


def match_fx_conversions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Try to fill missing total_czk for SELL rows by matching nearby FX conversion rows.
    Matching logic: same 'from amount' and currency within ±1 day.
    """
    df = df.copy()
    sells = df[(df["action_std"] == "SELL") & (df["total_czk"].isna())]
    conv = df[df["action_std"] == "FX_CONV"]

    if sells.empty or conv.empty:
        return df

    for idx, row in sells.iterrows():
        amt = row.get("total", np.nan)
        ccy = str(row.get("total_ccy", "")).upper()
        t = row.get("time")
        if pd.isna(amt) or not ccy or pd.isna(t):
            continue
        # Find conversion rows same day or next day
        mask = (
            conv["cc_from_amount"].round(2) == float(pd.to_numeric(amt, errors="coerce")).round(2)
        ) & (conv["cc_from_ccy"].str.upper() == ccy)
        near = conv[mask & (conv["time"].between(t - pd.Timedelta(days=1), t + pd.Timedelta(days=2)))]
        if not near.empty:
            # Take first match
            m = near.iloc[0]
            if str(m.get("cc_to_ccy", "")).upper() == "CZK" and pd.notna(m.get("cc_to_amount")):
                df.at[idx, "total_czk"] = float(m["cc_to_amount"])
    return df


def parse_trading212_csv(csv_df: pd.DataFrame) -> pd.DataFrame:
    """Full pipeline to parse raw CSV into normalized trades with CZK amounts."""
    df = normalize_csv(csv_df)
    df = attach_czk_amounts(df)
    df = match_fx_conversions(df)

    # Keep only supported actions
    df_trades = df[df["action_std"].isin(["BUY", "SELL", "DIVIDEND"])].copy()
    df_trades = df_trades.sort_values("time").reset_index(drop=True)
    return df_trades


def build_invested_timeseries(trades: pd.DataFrame) -> pd.DataFrame:
    """Build net invested (CZK) and realized P&L (CZK) time series."""
    if trades.empty:
        return pd.DataFrame()

    ts = trades[["time", "action_std", "total_czk", "result_czk"]].copy()

    # Net invested: BUY adds, SELL subtracts the cash received
    def flow(row):
        if row["action_std"] == "BUY":
            return float(row.get("total_czk", 0) or 0)
        if row["action_std"] == "SELL":
            # Cash back to you, reduce invested
            return -float(row.get("total_czk", 0) or 0)
        return 0.0

    ts["cash_flow_czk"] = ts.apply(flow, axis=1)
    ts = ts.dropna(subset=["time"]).sort_values("time")

    daily = ts.groupby(ts["time"].dt.date).agg({
        "cash_flow_czk": "sum",
        "result_czk": "sum",
    }).rename_axis("date").reset_index()

    daily["net_invested_czk"] = daily["cash_flow_czk"].cumsum()
    daily["realized_pnl_czk"] = daily["result_czk"].fillna(0).cumsum()

    # Convert date to Timestamp for plotting
    daily["date"] = pd.to_datetime(daily["date"])
    return daily
