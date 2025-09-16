import os
import streamlit as st
import pandas as pd
import requests
from typing import Dict, Any, List, Optional

# ---------------------------- UI Setup ----------------------------
st.set_page_config(page_title="Trading 212 Portfolio", layout="wide")
st.title("📊 Trading 212 Portfolio")
st.caption("View your current holdings and portfolio value")

# ---------------------------- Helper Functions ----------------------------
def _as_float(x, default: float = 0.0) -> float:
    try:
        return float(x)
    except (ValueError, TypeError):
        return default

def _as_list(obj: Any) -> List[Dict[str, Any]]:
    if obj is None:
        return []
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        for k in ["data", "items", "results"]:
            if k in obj and isinstance(obj[k], list):
                return obj[k]
    return []

# ---------------------------- Trading 212 Client ----------------------------
class T212Client:
    def __init__(self, api_key: str, is_demo: bool = False):
        self.session = requests.Session()
        self.session.headers.update({"Authorization": api_key})
        self.base = (
            "https://demo.trading212.com/api/v0" if is_demo 
            else "https://live.trading212.com/api/v0"
        )
    
    def _get(self, endpoint: str) -> Any:
        try:
            response = self.session.get(f"{self.base}{endpoint}", timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"Error fetching {endpoint}: {str(e)}")
            return None
    
    def get_cash_balance(self) -> Dict[str, Any]:
        return self._get("/equity/account/cash")
    
    def get_positions(self) -> List[Dict[str, Any]]:
        return self._get("/equity/portfolio")
    
    def get_account_info(self) -> Dict[str, Any]:
        return self._get("/equity/account/info")

# ---------------------------- Main App ----------------------------
# Sidebar
with st.sidebar:
    st.header("Settings")
    default_key = st.secrets.get("T212_API_KEY", os.environ.get("T212_API_KEY", ""))
    api_key = st.text_input("Trading 212 API key", type="password", value=default_key)
    is_demo = st.checkbox("Use Demo Account", value=False)
    
    if not api_key:
        st.warning("Please enter your Trading 212 API key")
        st.stop()

try:
    # Initialize client
    client = T212Client(api_key, is_demo)
    
    with st.spinner("Fetching portfolio data..."):
        # Get cash balance
        cash_data = client.get_cash_balance()
        if not cash_data:
            st.error("Failed to fetch cash balance. Please check your API key and try again.")
            st.stop()
        
        # Handle different response formats
        if isinstance(cash_data, dict):
            # New format: {"free": {"value": 123.45}}
            cash_balance = _as_float(cash_data.get("free", {}).get("value", 0))
        else:
            # Fallback: assume it's the direct numeric value
            cash_balance = _as_float(cash_data)
        
        # Get positions
        positions = client.get_positions()
        if positions is None:
            st.error("Failed to fetch positions. Please try again later.")
            st.stop()
            
        positions = positions or []
        
        # Process positions
        holdings = []
        total_value = 0
        
        for pos in positions:
            ticker = pos.get("ticker")
            name = pos.get("name", ticker)
            quantity = _as_float(pos.get("quantity"))
            
            # Get current price and value
            current_price = _as_float(pos.get("currentPrice", {}).get("value"))
            current_value = quantity * current_price
            
            # Get average price and invested amount
            avg_price = _as_float(pos.get("averagePrice"))
            invested = quantity * avg_price
            
            # Calculate P&L
            pnl = current_value - invested
            pnl_pct = (pnl / invested * 100) if invested else 0
            
            holdings.append({
                "Ticker": ticker,
                "Name": name,
                "Quantity": quantity,
                "Current Price (CZK)": current_price,
                "Avg. Price (CZK)": avg_price,
                "Current Value (CZK)": current_value,
                "Invested (CZK)": invested,
                "P&L (CZK)": pnl,
                "P&L (%)": pnl_pct
            })
            
            total_value += current_value
        
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
        if not holdings:
            st.info("No positions found in your portfolio.")
        else:
            df_holdings = pd.DataFrame(holdings)
            
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
        
        # Add some spacing at the bottom
        st.markdown("---")
        st.caption("Last updated: " + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"))

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    if st.checkbox("Show error details"):
        st.exception(e)
