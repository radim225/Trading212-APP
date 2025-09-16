import os
import streamlit as st
import pandas as pd
import requests
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
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

def get_company_name(ticker: str) -> str:
    """Map tickers to company names."""
    name_map = {
        'RSG_US_EQ': 'Republic Services',
        'CNX1_EQ': 'Centrica PLC (in GBX)',
        # Add more ticker to name mappings as needed
    }
    return name_map.get(ticker, ticker)

def convert_to_czk(amount: float, currency: str) -> float:
    """Convert amount to CZK using current exchange rates."""
    if currency == 'CZK':
        return amount
    
    try:
        # Simple hardcoded rates - replace with actual API call for live rates
        rates = {
            'USD': 23.5,  # 1 USD = 23.5 CZK
            'EUR': 25.0,  # 1 EUR = 25.0 CZK
            'GBP': 29.0,  # 1 GBP = 29.0 CZK
            'GBX': 0.29   # 1 GBX = 0.29 CZK (1 GBP = 100 GBX)
        }
        return amount * rates.get(currency, 1.0)
    except Exception as e:
        logger.error(f"Error converting {amount} {currency} to CZK: {str(e)}")
        return amount  # Return original amount if conversion fails

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
            logger.info(f"Making request to: {self.base}{endpoint}")
            response = self.session.get(f"{self.base}{endpoint}", timeout=30)
            logger.info(f"Status Code: {response.status_code}")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error fetching {endpoint}: {str(e)}")
            st.error(f"Error fetching {endpoint}: {str(e)}")
            return None
    
    def get_cash_balance(self) -> Dict[str, Any]:
        logger.info("Fetching cash balance...")
        data = self._get("/equity/account/cash")
        if data and isinstance(data, dict):
            logger.info(f"Cash balance response type: {type(data).__name__}")
            logger.info(f"Cash balance response keys: {list(data.keys())}")
            for key, value in data.items():
                try:
                    if isinstance(value, dict) and 'value' in value:
                        logger.info(f"  {key}: {value['value']} (type: {type(value['value']).__name__})")
                    else:
                        logger.info(f"  {key}: {value} (type: {type(value).__name__})")
                except Exception as e:
                    logger.error(f"Error processing {key}: {str(e)}")
                    continue
        return data or {}  # Always return a dict to prevent None errors
    
    def get_positions(self) -> List[Dict[str, Any]]:
        logger.info("Fetching positions...")
        try:
            positions = self._get("/equity/portfolio")
            if positions is None:
                logger.warning("Received None response for positions")
                return []
                
            logger.info(f"Positions data type: {type(positions).__name__}")
            
            if not isinstance(positions, list):
                logger.warning(f"Expected list but got {type(positions).__name__}")
                return []
                
            logger.info(f"Found {len(positions)} positions")
            if positions:
                logger.debug(f"First position: {positions[0]}")
                
            return positions
            
        except Exception as e:
            logger.error(f"Error in get_positions: {str(e)}")
            logger.error(traceback.format_exc())
            return []
    
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
                
        # Get all cash values in CZK
        cash_balance = convert_to_czk(_as_float(cash_data.get("free", {}).get("value", 0)), 'CZK')
        pie_cash = convert_to_czk(_as_float(cash_data.get("pieCash", {}).get("value", 0)), 'CZK')
        
        # Get positions
        positions = client.get_positions()
        if positions is None:
            st.error("Failed to fetch positions. Please try again later.")
            st.stop()
            
        positions = positions or []
        
        # Process positions
        holdings = []
        total_invested = 0
        total_current_value = 0
        
        for pos in positions:
            ticker = pos.get("ticker")
            name = get_company_name(ticker)
            quantity = _as_float(pos.get("quantity"))
            
            # Handle CNX1_EQ (pennies) - convert from GBX to GBP
            if ticker == 'CNX1_EQ':
                current_price_gbx = _as_float(pos.get("currentPrice", {}).get("value"))
                avg_price_gbx = _as_float(pos.get("averagePrice"))
                
                # Convert GBX to GBP (100 GBX = 1 GBP) and then to CZK
                current_price = convert_to_czk(current_price_gbx / 100, 'GBP')
                avg_price = convert_to_czk(avg_price_gbx / 100, 'GBP')
            else:
                # For other stocks, get price in original currency and convert to CZK
                current_price_original = _as_float(pos.get("currentPrice", {}).get("value"))
                avg_price_original = _as_float(pos.get("averagePrice"))
                currency = pos.get("currentPrice", {}).get("currency", "USD")
                
                current_price = convert_to_czk(current_price_original, currency)
                avg_price = convert_to_czk(avg_price_original, currency)
            
            # Calculate values
            current_value = quantity * current_price
            invested = quantity * avg_price
            
            # Calculate P&L
            pnl = current_value - invested
            pnl_pct = (pnl / invested * 100) if invested else 0
            
            # Add to totals
            total_invested += invested
            total_current_value += current_value
            
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
        
        # Calculate totals
        total_portfolio_value = total_current_value + cash_balance + pie_cash
        
        # Display summary
        st.subheader("Portfolio Summary")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Portfolio Value", f"{total_portfolio_value:,.2f} CZK")
        with col2:
            st.metric("Invested Amount", f"{total_invested:,.2f} CZK")
        with col3:
            st.metric("Cash Balance", f"{cash_balance:,.2f} CZK")
        with col4:
            st.metric("PIE Cash", f"{pie_cash:,.2f} CZK")
        
        # Display holdings table
        st.subheader("Current Holdings")
        if not holdings:
            st.info("No positions found in your portfolio.")
        else:
            df_holdings = pd.DataFrame(holdings)
            
            # Sort by current value (descending)
            df_holdings = df_holdings.sort_values("Current Value (CZK)", ascending=False)
            
            # Format numbers
            format_dict = {
                'Current Price (CZK)': '{:,.2f}',
                'Avg. Price (CZK)': '{:,.2f}',
                'Current Value (CZK)': '{:,.2f}',
                'Invested (CZK)': '{:,.2f}',
                'P&L (CZK)': '{:+,.2f}',
                'P&L (%)': '{:+,.2f}%',
                'Quantity': '{:,.2f}'
            }
            
            # Apply formatting
            for col, fmt in format_dict.items():
                if col in df_holdings.columns:
                    df_holdings[col] = df_holdings[col].apply(lambda x: fmt.format(x) if pd.notnull(x) else "")
            
            # Display the table
            st.dataframe(
                df_holdings,
                column_config={
                    "Ticker": "Ticker",
                    "Name": "Company Name",
                    "Quantity": "Shares",
                    "Current Price (CZK)": st.column_config.NumberColumn("Current Price (CZK)"),
                    "Avg. Price (CZK)": st.column_config.NumberColumn("Avg. Cost (CZK)"),
                    "Current Value (CZK)": st.column_config.NumberColumn("Market Value (CZK)"),
                    "Invested (CZK)": st.column_config.NumberColumn("Invested (CZK)"),
                    "P&L (CZK)": st.column_config.NumberColumn("P&L (CZK)"),
                    "P&L (%)": st.column_config.NumberColumn("P&L (%)")
                },
                hide_index=True,
                width='stretch',
                height=min(400, 50 + len(df_holdings) * 35)  # Adjust height based on number of rows
            )
        
        # Add some spacing at the bottom
        st.markdown("---")
        st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        st.caption("Note: All values are displayed in CZK. Exchange rates are approximate.")

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    if st.checkbox("Show error details"):
        st.exception(e)

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    if st.checkbox("Show error details"):
        st.exception(e)
