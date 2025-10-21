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

def parse_t212_ticker(ticker: str) -> Tuple[str, str, str]:
    """
    Parse Trading212's ticker format to extract the clean ticker, exchange, and currency.
    Examples:
        'AAPL_US_EQ' -> ('AAPL', 'US', 'USD')
        'CNX1_EQ' -> ('CNX1', 'LON', 'GBX')
        'MSFT_US_EQ' -> ('MSFT', 'US', 'USD')
    """
    if not ticker:
        return ticker, 'Unknown', 'Unknown'
    
    # Trading212 format: TICKER_EXCHANGE_EQ or TICKER_EQ
    parts = ticker.split('_')
    
    if len(parts) >= 2:
        clean_ticker = parts[0]
        exchange = parts[1] if parts[1] != 'EQ' else 'LON'  # Default to London if no exchange specified
        
        # Determine currency based on exchange
        currency_map = {
            'US': 'USD',
            'LON': 'GBX',
            'FRA': 'EUR',
            'GER': 'EUR',
            'SWI': 'CHF',
            'EQ': 'GBX'
        }
        currency = currency_map.get(exchange, 'USD')
        
        return clean_ticker, exchange, currency
    
    return ticker, 'Unknown', 'Unknown'

def get_company_name(ticker: str, api_key: str = None, is_demo: bool = False) -> str:
    """
    Get company name using Trading212 metadata API.
    Returns the ticker if no name is found.
    """
    if not ticker or ticker == "Unknown":
        return ticker
    
    # Parse the T212 ticker format
    clean_ticker, exchange, _ = parse_t212_ticker(ticker)
    
    # Check cache first
    if ticker in COMPANY_NAMES_CACHE:
        return COMPANY_NAMES_CACHE[ticker]
    
    # Try to get from Trading212 metadata API using the original ticker format
    try:
        base_url = "https://demo.trading212.com/api/v0" if is_demo else "https://live.trading212.com/api/v0"
        url = f"{base_url}/equity/metadata/instruments"
        headers = {}
        if api_key:
            headers["Authorization"] = api_key
        
        # Search for the instrument
        response = requests.get(url, headers=headers, timeout=5)
        if response.status_code == 200:
            instruments = response.json()
            # Find matching instrument
            for instrument in instruments:
                if instrument.get('ticker') == ticker:
                    name = instrument.get('name', ticker)
                    COMPANY_NAMES_CACHE[ticker] = name
                    logger.info(f"Found company name for {ticker}: {name}")
                    return name
    except Exception as e:
        logger.debug(f"Couldn't fetch name from Trading212 metadata for {ticker}: {str(e)}")
    
    # Fallback to Yahoo Finance using clean ticker
    try:
        yahoo_url = f"https://query2.finance.yahoo.com/v1/finance/search?q={clean_ticker}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(yahoo_url, headers=headers, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if 'quotes' in data and len(data['quotes']) > 0:
                name = data['quotes'][0].get('longname') or data['quotes'][0].get('shortname')
                if name:
                    COMPANY_NAMES_CACHE[ticker] = name
                    logger.info(f"Found company name via Yahoo for {ticker}: {name}")
                    return name
    except Exception as e:
        logger.debug(f"Couldn't fetch name from Yahoo Finance for {clean_ticker}: {str(e)}")
    
    # If all else fails, return the clean ticker and cache it
    COMPANY_NAMES_CACHE[ticker] = clean_ticker
    return clean_ticker

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
    """Convert various data types to float, including nested dicts."""
    try:
        if isinstance(x, dict) and 'value' in x:
            return float(x['value'])
        return float(x) if x is not None else default
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
    
    def get_metadata_instruments(self) -> List[Dict[str, Any]]:
        """Get list of all available instruments with metadata."""
        return self._get("/equity/metadata/instruments")

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
        
        # Get the total invested amount from the cash balance response
        total_invested = _as_float(cash_data.get('invested', 0))
        
        for pos in positions:
            try:
                # Debug: Log the position data structure
                logger.debug(f"Processing position: {json.dumps(pos, indent=2)}")
                
                # Get ticker and parse it
                ticker = pos.get("ticker", "Unknown")
                clean_ticker, exchange, native_currency = parse_t212_ticker(ticker)
                
                # Get the company name
                name = get_company_name(ticker, api_key, is_demo)
                logger.debug(f"Resolved ticker {ticker} ({clean_ticker}) to name: {name}")
                
                # Get quantity
                quantity = _as_float(pos.get("quantity"))
                
                # Get raw prices (these are in the instrument's native currency: USD, GBX/pence, etc.)
                current_price_native = _as_float(pos.get("currentPrice"))
                average_price_native = _as_float(pos.get("averagePrice"))
                
                # Get P&L values from API (these are ALREADY in CZK and account for FX)
                ppl = _as_float(pos.get("ppl"))  # Price P&L in CZK
                fx_ppl = _as_float(pos.get("fxPpl"))  # FX Impact in CZK
                
                # Total P&L in CZK (includes both price movement and FX impact)
                total_pnl_czk = ppl + fx_ppl
                
                # Get maxBuy/maxSell which are already in CZK
                max_sell = _as_float(pos.get("maxSell", 0))
                
                # Use maxSell as market value (most reliable, already in CZK)
                if max_sell > 0:
                    market_value_czk = max_sell
                    # Calculate invested from: invested = market_value - total_pnl
                    invested_czk = market_value_czk - total_pnl_czk
                else:
                    # Fallback: estimate from quantity * price
                    # This is a rough estimate and may not be accurate
                    market_value_czk = quantity * current_price_native
                    invested_czk = quantity * average_price_native
                
                # Derive current price in CZK from market value
                current_price_czk = market_value_czk / quantity if quantity > 0 else 0
                avg_price_czk = invested_czk / quantity if quantity > 0 else 0
                
                # Format currency symbol for display (handle encoding issues)
                try:
                    if native_currency == 'GBX':
                        currency_symbol = 'p'
                    elif native_currency == 'USD':
                        currency_symbol = '$'
                    elif native_currency == 'EUR':
                        currency_symbol = 'EUR'
                    elif native_currency == 'GBP':
                        currency_symbol = 'GBP'
                    else:
                        currency_symbol = native_currency
                    
                    # Create dual currency display strings
                    avg_price_display = f"{average_price_native:,.2f} {currency_symbol} ({avg_price_czk:,.2f} CZK)"
                    current_price_display = f"{current_price_native:,.2f} {currency_symbol} ({current_price_czk:,.2f} CZK)"
                except Exception as fmt_err:
                    logger.error(f"Error formatting prices for {clean_ticker}: {fmt_err}")
                    # Fallback to simple format
                    avg_price_display = f"{avg_price_czk:,.2f} CZK"
                    current_price_display = f"{current_price_czk:,.2f} CZK"
                
                logger.info(f"Position {clean_ticker}: qty={quantity}, native={current_price_native} {native_currency}, czk={current_price_czk:.2f}, invested={invested_czk:.2f}, market_value={market_value_czk:.2f}, ppl={ppl:.2f}, fxPpl={fx_ppl:.2f}")
                
                # P&L percentage
                pnl_percent = (total_pnl_czk / invested_czk * 100) if invested_czk > 0 else 0
                
                total_value += market_value_czk
                
                holdings.append({
                    "Ticker": clean_ticker,
                    "Company Name": name,
                    "Shares": quantity,
                    "Avg. Cost": avg_price_display,
                    "Current Price": current_price_display,
                    "Market Value (CZK)": market_value_czk,
                    "Invested (CZK)": invested_czk,
                    "P&L (CZK)": total_pnl_czk,
                    "P&L (%)": pnl_percent
                })
                
            except Exception as e:
                error_msg = f"Error processing position {pos.get('ticker', 'unknown')}: {str(e)}"
                logger.error(error_msg)
                logger.error(f"Problematic position data: {pos}")
                logger.error(traceback.format_exc())
                st.sidebar.error(f"⚠️ {error_msg}")
                continue  # Skip this position but continue with others
        
        # Log processing summary
        logger.info(f"Successfully processed {len(holdings)} out of {len(positions)} positions")
        if len(holdings) < len(positions):
            skipped = len(positions) - len(holdings)
            st.sidebar.warning(f"⚠️ Skipped {skipped} positions due to errors")
        
        # Get PIE cash from cash balance response
        pie_cash = _as_float(cash_data.get('pieCash', 0))
        
        # Display Portfolio Summary
        st.subheader("Portfolio Summary")
        
        # Calculate total portfolio value from API
        total_portfolio_value = _as_float(cash_data.get('total', 0))
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Portfolio Value", f"{total_portfolio_value:,.2f} CZK")
        with col2:
            st.metric("Invested Amount", f"{total_invested:,.2f} CZK")
        with col3:
            st.metric("Cash Balance", f"{cash_balance:,.2f} CZK")
        with col4:
            st.metric("PIE Cash", f"{pie_cash:,.2f} CZK")
        
        st.write("")
        st.subheader("Current Holdings")
        if not holdings:
            st.info("No positions found in your portfolio.")
        else:
            df_holdings = pd.DataFrame(holdings)
            
            # Display the dataframe with proper formatting
            st.dataframe(
                df_holdings,
                column_config={
                    "Ticker": st.column_config.TextColumn("Ticker", width="small"),
                    "Company Name": st.column_config.TextColumn("Company Name", width="medium"),
                    "Shares": st.column_config.NumberColumn("Shares", format="%.4f"),
                    "Avg. Cost": st.column_config.TextColumn("Avg. Cost", width="medium", help="Original currency (CZK equivalent)"),
                    "Current Price": st.column_config.TextColumn("Current Price", width="medium", help="Original currency (CZK equivalent)"),
                    "Market Value (CZK)": st.column_config.NumberColumn("Market Value (CZK)", format="%.2f"),
                    "Invested (CZK)": st.column_config.NumberColumn("Invested (CZK)", format="%.2f"),
                    "P&L (CZK)": st.column_config.NumberColumn("P&L (CZK)", format="%.2f"),
                    "P&L (%)": st.column_config.NumberColumn("P&L (%)", format="%.4f")
                },
                hide_index=True,
                use_container_width=True
            )
            
            # Show data format explanation
            with st.expander("📊 Data Format Explanation"):
                st.markdown("""
                **Trading212 API Data Structure:**
                
                - **Ticker Format**: Trading212 uses internal codes like `AAPL_US_EQ` (Apple, US Exchange) or `CNX1_EQ` (London Exchange by default)
                - **Clean Ticker**: We parse these to show just the ticker symbol (e.g., `AAPL`, `CNX1`)
                - **Company Name**: Fetched from Trading212 metadata API or Yahoo Finance
                - **Prices**: All values are converted to CZK by Trading212 API
                - **P&L Calculation**: Includes both price movement (`ppl`) and FX impact (`fxPpl`)
                - **FX Impact**: Currency fluctuations are automatically accounted for by the API
                
                **Example Position Data from API:**
                ```json
                {
                  "ticker": "AAPL_US_EQ",
                  "quantity": 3.19,
                  "averagePrice": 4623.69,  // in CZK
                  "currentPrice": 5888.25,  // in CZK
                  "ppl": 4050.97,           // Price P&L in CZK
                  "fxPpl": -94.87,          // FX impact in CZK
                  "initialFillDate": "2024-05-21T16:37:03.000+03:00",
                  "frontend": "AUTOINVEST"
                }
                ```
                
                **Total P&L** = Price P&L + FX P&L = 4050.97 + (-94.87) = 3956.10 CZK
                """)
        
        # Add some spacing at the bottom
        st.markdown("---")
        st.caption("Last updated: " + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"))

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    if st.checkbox("Show error details"):
        st.exception(e)
