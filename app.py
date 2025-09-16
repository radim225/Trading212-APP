import os
import sys
import streamlit as st
import pandas as pd
import requests
import json
import logging
import traceback
import time
from typing import Dict, List, Any, Optional, Tuple

# Cache for company names to avoid repeated API calls
COMPANY_NAMES_CACHE = {}

def get_company_name(ticker: str) -> str:
    """
    Get company name from cache, Trading212 API, or fallback to a financial data API.
    Returns the ticker if no name is found.
    """
    if not ticker or ticker == "Unknown":
        return ticker
        
    # Clean the ticker (remove exchange suffixes like .LON, .FRA, etc.)
    base_ticker = ticker.split('.')[0].upper()
    
    # Check cache first
    if base_ticker in COMPANY_NAMES_CACHE:
        return COMPANY_NAMES_CACHE[base_ticker]
    
    try:
        # Try to get from Trading212 API first
        url = f"https://live.trading212.com/api/v0/equity/metadata/instruments/{base_ticker}"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if 'name' in data:
                COMPANY_NAMES_CACHE[base_ticker] = data['name']
                return data['name']
    except Exception as e:
        logger.debug(f"Couldn't fetch name from Trading212 for {base_ticker}: {str(e)}")
    
    # Fallback to Yahoo Finance (if Trading212 fails)
    try:
        yahoo_url = f"https://query2.finance.yahoo.com/v1/finance/search?q={base_ticker}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(yahoo_url, headers=headers, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if 'quotes' in data and len(data['quotes']) > 0:
                name = data['quotes'][0].get('longname') or data['quotes'][0].get('shortname')
                if name:
                    COMPANY_NAMES_CACHE[base_ticker] = name
                    return name
    except Exception as e:
        logger.debug(f"Couldn't fetch name from Yahoo Finance for {base_ticker}: {str(e)}")
    
    # If all else fails, return the ticker and cache it to avoid repeated lookups
    COMPANY_NAMES_CACHE[base_ticker] = base_ticker
    return base_ticker

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
                    
                # Log detailed information about the response
                logger.info(f"Cash balance response type: {type(cash_data).__name__}")
                if isinstance(cash_data, dict):
                    logger.info(f"Cash balance response keys: {list(cash_data.keys())}")
                    for k, v in cash_data.items():
                        logger.info(f"  {k}: {v} (type: {type(v).__name__})")
                else:
                    logger.info(f"Cash balance value: {cash_data}")
                
                # Log the raw response for debugging
                logger.debug(f"Cash balance raw response: {cash_data}")
                
                # Simple conversion to float
                try:
                    if isinstance(cash_data, (int, float)):
                        cash_balance = float(cash_data)
                    elif isinstance(cash_data, dict):
                        # If it's a dict, just take the first numeric value we find
                        for v in cash_data.values():
                            if isinstance(v, (int, float)):
                                cash_balance = float(v)
                                break
                        else:
                            # If no numeric value found, try to convert the first value to float
                            if cash_data:
                                cash_balance = float(next(iter(cash_data.values())))
                            else:
                                cash_balance = 0.0
                    else:
                        # For any other type, try direct conversion
                        cash_balance = float(cash_data)
                        
                    logger.info(f"Parsed cash balance: {cash_balance}")
                    
                except Exception as e:
                    logger.error(f"Error parsing cash balance: {e}")
                    logger.error(f"Cash data type: {type(cash_data).__name__}")
                    logger.error(f"Cash data value: {cash_data}")
                    st.error("Could not determine cash balance. Using 0.0 as default.")
                    cash_balance = 0.0
                
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
                positions = client.get_positions()
                
                # Log the raw positions data for debugging
                logger.info(f"Positions data type: {type(positions).__name__}")
                if isinstance(positions, list):
                    logger.info(f"Found {len(positions)} positions")
                    if positions:
                        logger.info(f"First position: {positions[0]}")
                else:
                    logger.info(f"Unexpected positions format: {positions}")
                
                # Ensure positions is a list
                if not isinstance(positions, list):
                    positions = []
                    logger.warning("Positions data is not a list, using empty list")
                
                st.sidebar.success(f"✅ Found {len(positions)} positions")
                
                # Show raw positions in debug
                with st.sidebar.expander("Raw Positions Response"):
                    st.json(positions if positions else "No positions found")
                
            except Exception as e:
                logger.error(f"Error fetching positions: {str(e)}")
                logger.error(traceback.format_exc())
                st.sidebar.error(f"❌ Error fetching positions: {str(e)}")
                st.error("Failed to fetch positions. Some data may be missing.")
                positions = []
        
        # Process positions with error handling
        holdings = []
        total_value = 0
        
        for pos in positions:
            try:
                # Safely get position data with defaults
                ticker = pos.get("ticker", "Unknown")
                # Use the full company name from our mapping
                name = get_company_name(ticker)
                
                # Handle quantity - could be in different formats
                quantity_data = pos.get("quantity", 0)
                if isinstance(quantity_data, dict):
                    quantity = _as_float(quantity_data.get("value", 0))
                else:
                    quantity = _as_float(quantity_data)
                
                # Handle current price
                current_price_data = pos.get("currentPrice", {})
                if isinstance(current_price_data, dict):
                    current_price = _as_float(current_price_data.get("value", 0))
                else:
                    current_price = _as_float(current_price_data)
                
                # Handle average price
                average_price = _as_float(pos.get("averagePrice", 0))
                
                # Calculate values with safety checks
                value = quantity * current_price if quantity and current_price else 0
                total_value += value
                
                # Calculate P&L with safety checks
                pnl_value = value - (quantity * average_price) if quantity and average_price else 0
                pnl_percent = ((current_price / average_price) - 1) * 100 if average_price and average_price != 0 else 0
                
                holdings.append({
                    "Ticker": ticker,
                    "Name": name,
                    "Quantity": quantity,
                    "Avg Price": average_price,
                    "Current Price": current_price,
                    "Value": value,
                    "P&L": pnl_value,
                    "P&L %": pnl_percent
                })
                
            except Exception as e:
                logger.error(f"Error processing position {pos.get('ticker', 'unknown')}: {str(e)}")
                logger.error(f"Problematic position data: {pos}")
                continue  # Skip this position but continue with others
        
        # Convert holdings to DataFrame for display
        if holdings:
            df_holdings = pd.DataFrame(holdings)
            
            # Format the DataFrame
            df_holdings = df_holdings[["Ticker", "Name", "Quantity", "Avg Price", 
                                    "Current Price", "Value", "P&L", "P&L %"]]
            
            # Format numbers for display
            df_holdings["Quantity"] = df_holdings["Quantity"].apply(lambda x: f"{x:,.2f}")
            df_holdings["Avg Price"] = df_holdings["Avg Price"].apply(lambda x: f"{x:,.2f}")
            df_holdings["Current Price"] = df_holdings["Current Price"].apply(lambda x: f"{x:,.2f}")
            df_holdings["Value"] = df_holdings["Value"].apply(lambda x: f"{x:,.2f}")
            df_holdings["P&L"] = df_holdings["P&L"].apply(lambda x: f"{x:,.2f}")
            df_holdings["P&L %"] = df_holdings["P&L %"].apply(lambda x: f"{x:,.2f}%")
            
            # Display the table
            st.dataframe(df_holdings, use_container_width=True)
        
        # Calculate portfolio metrics
        portfolio_total = total_value + cash_balance
        
        # Display summary with proper formatting
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Invested Value (CZK)", f"{total_value:,.2f}")
        with col2:
            st.metric("Cash Balance (CZK)", f"{cash_balance:,.2f}")
        with col3:
            st.metric("Total Portfolio Value (CZK)", f"{portfolio_total:,.2f}")
            
        # Add some spacing
        st.write("")
        st.write("### Portfolio Holdings")
        
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
