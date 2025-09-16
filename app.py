import os
import sys
import streamlit as st
import pandas as pd
import requests
import json
import logging
import traceback
from typing import Dict, List, Any, Optional

# Configure logging to stderr (will be captured in Streamlit logs)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)

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
        self.last_response = None
    
    def _get(self, endpoint: str):
        url = f"{self.base}{endpoint}"
        try:
            logger.info(f"Making request to: {url}")
            self.last_response = self.session.get(url, timeout=30)
            
            # Log request details
            logger.info(f"Status Code: {self.last_response.status_code}")
            logger.debug("Response Headers:")
            for k, v in self.last_response.headers.items():
                logger.debug(f"  {k}: {v}")
            
            # Try to parse JSON
            try:
                data = self.last_response.json()
                logger.debug(f"Response JSON: {json.dumps(data, indent=2)}")
                return data
            except ValueError as e:
                logger.warning(f"Failed to parse JSON response: {e}")
                logger.debug(f"Raw response: {self.last_response.text}")
                return self.last_response.text
                
        except Exception as e:
            logger.error(f"Request failed: {str(e)}")
            logger.error(traceback.format_exc())
            raise
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
    
    # Add demo mode option
    use_demo_data = st.checkbox("View sample data (demo mode)", value=False)
    
    if not use_demo_data:
        api_key = st.text_input("Trading 212 API key", type="password", value=default_key)
        is_demo = st.checkbox("Use Demo Account", value=False)
        
        if not api_key:
            st.warning("Please enter your Trading 212 API key or enable demo mode")
            st.stop()
    else:
        api_key = ""
        is_demo = False

try:
    if use_demo_data:
        # Show sample data in demo mode
        st.info("Viewing sample data in demo mode. No API key required.")
        
        # Sample data for demo
        cash_balance = 1500.75
        positions = [
            {"ticker": "AAPL", "name": "Apple Inc.", "quantity": 10, "currentPrice": {"value": 175.50}, "averagePrice": 160.25},
            {"ticker": "MSFT", "name": "Microsoft Corporation", "quantity": 5, "currentPrice": {"value": 300.20}, "averagePrice": 280.75}
        ]
    else:
        # Initialize client with real API
        client = T212Client(api_key, is_demo)
        
        with st.spinner("Fetching portfolio data..."):
            # Debug: Show request details
            st.sidebar.write("Debug Info:")
            st.sidebar.json({
                "api_key_provided": bool(api_key),
                "is_demo": is_demo,
                "base_url": client.base if hasattr(client, 'base') else 'Not initialized'
            })
            
            # Get cash balance with error handling
            try:
                st.sidebar.write("⏳ Fetching cash balance...")
                logger.info("Fetching cash balance...")
                
                cash_data = client.get_cash_balance()
                logger.info(f"Cash balance response type: {type(cash_data).__name__}")
                
                # Show raw response in debug
                with st.sidebar.expander("Raw Cash Balance Response"):
                    st.json(cash_data if cash_data is not None else "No data returned")
                    
                # Log the raw response for debugging
                logger.debug(f"Cash balance raw response: {cash_data}")
                
                # Handle different response formats
                if isinstance(cash_data, dict):
                    # Try different possible response formats
                    cash_balance = _as_float(
                        cash_data.get("free", {}).get("value", 
                        cash_data.get("cash", {}).get("value", 
                        cash_data.get("balance", 
                        next((v for v in cash_data.values() if isinstance(v, (int, float))), 0)
                    ))))
                else:
                    # If it's already a number, use it directly
                    cash_balance = _as_float(cash_data)
                    
                logger.info(f"Parsed cash balance: {cash_balance}")
                
                st.sidebar.success(f"✅ Cash balance: {cash_balance:.2f}")
                
            except Exception as e:
                error_msg = f"❌ Error fetching cash balance: {str(e)}"
                logger.error(error_msg)
                logger.error(traceback.format_exc())
                st.sidebar.error(error_msg)
                st.error("Failed to fetch cash balance. Please check your API key and try again.")
                st.stop()
            
            # Get positions with error handling
            try:
                st.sidebar.write("⏳ Fetching positions...")
                positions = client.get_positions() or []
                st.sidebar.success(f"✅ Found {len(positions)} positions")
                
                # Show raw positions in debug
                with st.sidebar.expander("Raw Positions Response"):
                    st.json(positions if positions else "No positions found")
                
            except Exception as e:
                st.sidebar.error(f"❌ Error fetching positions: {str(e)}")
                st.error("Failed to fetch positions. Some data may be missing.")
                positions = []
        
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
