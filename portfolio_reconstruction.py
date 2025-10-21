"""
Portfolio Value Reconstruction Module
Reconstructs historical portfolio values from trade history + external price data
"""

import pandas as pd
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


def fetch_historical_prices_yahoo(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch historical prices from Yahoo Finance.
    ticker: Clean ticker symbol (e.g., 'AAPL', 'GOOGL')
    Returns DataFrame with Date and Close price
    """
    try:
        # Yahoo Finance format
        yahoo_ticker = ticker
        
        # Convert dates to timestamps
        start_ts = int(pd.Timestamp(start_date).timestamp())
        end_ts = int(pd.Timestamp(end_date).timestamp())
        
        url = f"https://query1.finance.yahoo.com/v7/finance/download/{yahoo_ticker}"
        params = {
            'period1': start_ts,
            'period2': end_ts,
            'interval': '1d',
            'events': 'history'
        }
        
        headers = {
            'User-Agent': 'Mozilla/5.0'
        }
        
        response = requests.get(url, params=params, headers=headers, timeout=10)
        
        if response.status_code == 200:
            from io import StringIO
            df = pd.read_csv(StringIO(response.text))
            df['Date'] = pd.to_datetime(df['Date'])
            return df[['Date', 'Close']].rename(columns={'Close': 'Price'})
        else:
            logger.warning(f"Failed to fetch prices for {ticker}: {response.status_code}")
            return pd.DataFrame()
            
    except Exception as e:
        logger.error(f"Error fetching prices for {ticker}: {e}")
        return pd.DataFrame()


def fetch_fx_rates_ecb(currency_pairs: List[str], start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch historical FX rates from ECB API.
    currency_pairs: e.g., ['USD', 'GBP']
    Returns DataFrame with Date and rates
    """
    try:
        # ECB API endpoint
        url = "https://data-api.ecb.europa.eu/service/data/EXR/D"
        
        # For now, return empty - we'll use a simpler approach
        # In production, we'd fetch from ECB or Forex API
        return pd.DataFrame()
        
    except Exception as e:
        logger.error(f"Error fetching FX rates: {e}")
        return pd.DataFrame()


def reconstruct_portfolio_value(
    orders: List[Dict],
    start_date: datetime,
    end_date: datetime,
    progress_callback=None
) -> pd.DataFrame:
    """
    Reconstruct historical portfolio value from orders + external price data.
    
    Returns DataFrame with columns:
    - Date
    - Portfolio_Value_USD (total in USD)
    - Portfolio_Value_CZK (converted to CZK)
    - Cash_Balance
    - Positions (dict of holdings)
    """
    
    if not orders:
        return pd.DataFrame()
    
    # Convert orders to DataFrame
    orders_df = pd.DataFrame(orders)
    
    # Parse dates
    if 'filledDate' in orders_df.columns:
        orders_df['date'] = pd.to_datetime(orders_df['filledDate'])
    elif 'dateCreated' in orders_df.columns:
        orders_df['date'] = pd.to_datetime(orders_df['dateCreated'])
    else:
        return pd.DataFrame()
    
    # Sort by date
    orders_df = orders_df.sort_values('date')
    
    # Extract ticker symbols (clean them)
    tickers = orders_df['ticker'].unique()
    clean_tickers = [t.split('_')[0] for t in tickers]
    
    if progress_callback:
        progress_callback(f"Fetching historical prices for {len(clean_tickers)} stocks...")
    
    # Fetch historical prices for each ticker
    price_data = {}
    for i, (original_ticker, clean_ticker) in enumerate(zip(tickers, clean_tickers)):
        if progress_callback:
            progress_callback(f"Fetching prices for {clean_ticker} ({i+1}/{len(clean_tickers)})...")
        
        prices = fetch_historical_prices_yahoo(
            clean_ticker,
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )
        
        if not prices.empty:
            price_data[original_ticker] = prices.set_index('Date')['Price']
    
    if not price_data:
        logger.warning("No price data fetched")
        return pd.DataFrame()
    
    # Create date range
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Initialize portfolio tracking
    portfolio_values = []
    current_positions = {}  # {ticker: quantity}
    
    for current_date in date_range:
        # Process orders up to this date
        orders_today = orders_df[orders_df['date'].dt.date <= current_date.date()]
        
        # Update positions based on orders
        for _, order in orders_today.iterrows():
            ticker = order['ticker']
            quantity = order.get('filledQuantity', 0)
            order_type = order.get('type', 'BUY')
            
            if ticker not in current_positions:
                current_positions[ticker] = 0
            
            if order_type in ['LIMIT', 'MARKET', 'BUY']:
                current_positions[ticker] += quantity
            elif order_type in ['SELL', 'STOP']:
                current_positions[ticker] -= quantity
        
        # Calculate portfolio value for this date
        total_value = 0
        
        for ticker, quantity in current_positions.items():
            if quantity > 0 and ticker in price_data:
                # Get price for this date
                try:
                    price = price_data[ticker].asof(current_date)
                    if pd.notna(price):
                        total_value += quantity * price
                except Exception:
                    pass
        
        portfolio_values.append({
            'Date': current_date,
            'Portfolio_Value': total_value,
            'Positions': len([q for q in current_positions.values() if q > 0])
        })
        
        if progress_callback and len(portfolio_values) % 30 == 0:
            progress_callback(f"Calculated {len(portfolio_values)} days...")
    
    # Convert to DataFrame
    result_df = pd.DataFrame(portfolio_values)
    
    # Add simple FX conversion (would need real rates for accuracy)
    # For now, use a fixed rate as approximation
    USD_CZK_RATE = 23.0  # Approximate
    result_df['Portfolio_Value_CZK'] = result_df['Portfolio_Value'] * USD_CZK_RATE
    
    return result_df


def get_simplified_approximation(orders_df: pd.DataFrame, current_total: float) -> pd.DataFrame:
    """
    Create a simplified approximation without external APIs.
    Uses linear interpolation between order dates and current value.
    """
    if orders_df.empty:
        return pd.DataFrame()
    
    # Get order dates and cumulative invested
    orders_df = orders_df.sort_values('date')
    orders_df['cumulative_cost'] = orders_df['value'].cumsum()
    
    # Sample key points: first order, some orders in between, last order, today
    key_points = []
    
    # First order
    first_order = orders_df.iloc[0]
    key_points.append({
        'Date': first_order['date'],
        'Invested': first_order['value'],
        'Estimated_Value': first_order['value']  # Assume no gain at purchase
    })
    
    # Last order
    last_order = orders_df.iloc[-1]
    key_points.append({
        'Date': last_order['date'],
        'Invested': last_order['cumulative_cost'],
        'Estimated_Value': last_order['cumulative_cost'] * 1.1  # Assume some growth
    })
    
    # Today
    key_points.append({
        'Date': pd.Timestamp.now(),
        'Invested': orders_df['cumulative_cost'].iloc[-1],
        'Estimated_Value': current_total
    })
    
    df = pd.DataFrame(key_points)
    
    # Interpolate between points
    df = df.set_index('Date').resample('D').interpolate(method='linear').reset_index()
    
    return df
