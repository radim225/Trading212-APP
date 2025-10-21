"""
Historical P&L Chart Module for Trading212 Portfolio App
Provides interactive charting with FX impact visualization
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import List, Dict, Any
import streamlit as st
import logging

# Import portfolio reconstruction
try:
    from portfolio_reconstruction import reconstruct_portfolio_value, get_simplified_approximation
    from smart_reconstruction import smart_reconstruct_portfolio
except ImportError:
    reconstruct_portfolio_value = None
    get_simplified_approximation = None
    smart_reconstruct_portfolio = None

logger = logging.getLogger(__name__)


def process_historical_orders(orders: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Process historical orders into a time-series DataFrame.
    Calculate cumulative P&L with FX impact over time.
    """
    if not orders:
        return pd.DataFrame()
    
    # Convert to DataFrame
    df = pd.DataFrame(orders)
    
    # Parse dates
    if 'dateCreated' in df.columns:
        df['date'] = pd.to_datetime(df['dateCreated'])
    elif 'filledDate' in df.columns:
        df['date'] = pd.to_datetime(df['filledDate'])
    else:
        return pd.DataFrame()
    
    # Sort by date
    df = df.sort_values('date')
    
    # Extract relevant fields
    df['quantity'] = df['filledQuantity'].fillna(0)
    df['price'] = df['fillPrice'].fillna(0)
    df['value'] = df['fillCost'].fillna(0)
    
    return df


def calculate_portfolio_value_over_time(orders_df: pd.DataFrame, current_positions: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Calculate portfolio value over time including FX impact.
    """
    if orders_df.empty:
        return pd.DataFrame()
    
    # Group by date and calculate daily invested amount
    daily_df = orders_df.groupby(orders_df['date'].dt.date).agg({
        'value': 'sum',
        'quantity': 'sum'
    }).reset_index()
    
    # Calculate cumulative invested
    daily_df['cumulative_invested'] = daily_df['value'].cumsum()
    
    # For now, use current portfolio value as the end point
    # In a full implementation, we'd track historical prices
    
    return daily_df


def create_pnl_chart(historical_data: pd.DataFrame, current_total: float, current_invested: float) -> go.Figure:
    """
    Create interactive P&L chart with FX impact breakdown.
    """
    if historical_data.empty:
        # Create empty chart with message
        fig = go.Figure()
        fig.add_annotation(
            text="No historical data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20)
        )
        fig.update_layout(height=400)
        return fig
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Portfolio Value Over Time', 'P&L Breakdown'),
        vertical_spacing=0.15,
        row_heights=[0.6, 0.4]
    )
    
    # Portfolio value line
    fig.add_trace(
        go.Scatter(
            x=historical_data['date'],
            y=historical_data['cumulative_invested'],
            name='Invested Amount',
            line=dict(color='#4a90e2', width=2),
            fill='tozeroy',
            fillcolor='rgba(74, 144, 226, 0.1)'
        ),
        row=1, col=1
    )
    
    # Add current value point
    if len(historical_data) > 0:
        last_date = historical_data['date'].max()
        fig.add_trace(
            go.Scatter(
                x=[last_date],
                y=[current_total],
                name='Current Value',
                mode='markers',
                marker=dict(size=12, color='#2ecc71', symbol='diamond')
            ),
            row=1, col=1
        )
    
    # P&L area chart (simplified for now)
    if current_total > 0 and current_invested > 0:
        pnl = current_total - current_invested
        fig.add_trace(
            go.Bar(
                x=['Total P&L'],
                y=[pnl],
                name='P&L',
                marker_color='#2ecc71' if pnl >= 0 else '#e74c3c'
            ),
            row=2, col=1
        )
    
    # Update layout
    fig.update_layout(
        height=700,
        showlegend=True,
        hovermode='x unified',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
    
    return fig


def display_historical_section(client, current_positions, cash_data):
    """
    Main function to display the historical P&L section with charts.
    """
    st.markdown("---")
    st.header("📈 Historical Performance")
    
    # Reconstruction method selector
    st.subheader("📈 Choose Visualization Method")
    
    method = st.radio(
        "How would you like to view your historical performance?",
        [
            "📊 Simple Approximation (Fast, Estimated)",
            "🔬 Detailed Reconstruction (Slow, Uses Yahoo Finance)",
            "📋 Order History Only"
        ],
        help="Simple uses linear approximation. Detailed fetches real historical prices."
    )
    
    # Info box
    with st.expander("ℹ️ About Each Method"):
        st.markdown("""
        **📊 Simple Approximation (Smart):**
        - Fast and lightweight
        - Separates CLOSED vs OPEN positions
        - For closed trades: Uses actual buy/sell prices
        - For open positions: Uses linear growth estimation
        - Good for quick overview
        
        **🔬 Detailed Reconstruction:**
        - Fetches real historical prices from Yahoo Finance
        - Calculates actual portfolio value for each day
        - More accurate representation
        - Takes longer to load (1-2 minutes)
        - Note: FX rates are approximated
        
        **📋 Order History Only:**
        - Shows when you made trades
        - Cumulative invested amount
        - No estimated portfolio values
        """)
    
    # CSV Upload option
    st.subheader("📤 Upload Historical Data")
    uploaded_file = st.file_uploader(
        "Upload Trading212 CSV export for accurate historical portfolio values",
        type=['csv'],
        help="Download from Trading212 app: Account → Reports → Export"
    )
    
    if uploaded_file is not None:
        try:
            # Read CSV
            csv_df = pd.read_csv(uploaded_file)
            st.success(f"✅ Loaded {len(csv_df)} records from CSV")
            
            # Display preview
            with st.expander("📋 CSV Preview"):
                st.dataframe(csv_df.head(10), use_container_width=True)
            
            # TODO: Process CSV and create historical value chart
            st.info("CSV processing coming soon! For now, showing data from API...")
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
    
    # Fetch historical data
    with st.spinner("Loading historical data..."):
        try:
            st.info("📥 Fetching historical orders from Trading212...")
            orders = client.get_all_historical_orders()
            
            if not orders:
                st.warning("No historical orders found. Start trading to see your performance over time!")
                return
            
            st.success(f"✅ Loaded {len(orders)} historical orders")
            
            # Process orders
            orders_df = process_historical_orders(orders)
            
            if orders_df.empty:
                st.warning("Could not process historical orders.")
                return
            
            # Get current portfolio metrics
            current_total = _as_float(cash_data.get('total', 0))
            current_invested = _as_float(cash_data.get('invested', 0))
            current_pnl = _as_float(cash_data.get('ppl', 0))
            current_fx_pnl = _as_float(cash_data.get('result', 0)) - current_pnl
            
            # Display key metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                total_return = ((current_total - current_invested) / current_invested * 100) if current_invested > 0 else 0
                st.metric("Total Return", f"{total_return:+.2f}%", 
                         delta=f"{current_total - current_invested:,.2f} CZK")
            with col2:
                st.metric("Total Orders", f"{len(orders):,}")
            with col3:
                if len(orders_df) > 0:
                    first_order_date = orders_df['date'].min().strftime('%Y-%m-%d')
                    st.metric("Since", first_order_date)
            with col4:
                if len(orders_df) > 0:
                    # Make datetime timezone-aware to match pandas timestamps
                    now = pd.Timestamp.now(tz=orders_df['date'].min().tz)
                    days_active = (now - orders_df['date'].min()).days
                else:
                    days_active = 0
                st.metric("Days Active", f"{days_active:,}")
            
            # Calculate historical portfolio value based on selected method
            if "Simple Approximation" in method:
                st.info("🔄 Creating smart approximation (optimized for closed/open positions)...")
                
                # Use smart reconstruction if available, otherwise fallback
                if smart_reconstruct_portfolio:
                    hist_data = smart_reconstruct_portfolio(orders_df, holdings, current_total)
                elif get_simplified_approximation:
                    hist_data = get_simplified_approximation(orders_df, current_total)
                else:
                    hist_data = pd.DataFrame()
                
                if not hist_data.empty:
                    # Create enhanced chart with approximation
                    fig = go.Figure()
                    
                    # Invested amount line
                    fig.add_trace(go.Scatter(
                        x=hist_data['Date'],
                        y=hist_data['Invested'],
                        name='Invested Amount',
                        line=dict(color='#3498db', width=2, dash='dash')
                    ))
                    
                    # Estimated portfolio value
                    fig.add_trace(go.Scatter(
                        x=hist_data['Date'],
                        y=hist_data['Estimated_Value'],
                        name='Estimated Portfolio Value',
                        line=dict(color='#2ecc71', width=3),
                        fill='tonexty',
                        fillcolor='rgba(46, 204, 113, 0.1)'
                    ))
                    
                    fig.update_layout(
                        title="Portfolio Value Over Time (Approximation)",
                        xaxis_title="Date",
                        yaxis_title="Value (CZK)",
                        height=500,
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Could not create approximation")
                    
            elif "Detailed Reconstruction" in method and reconstruct_portfolio_value:
                st.warning("⏳ This will take 1-2 minutes. Fetching historical prices...")
                
                # Get date range
                start_date = orders_df['date'].min()
                end_date = pd.Timestamp.now()
                
                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def progress_callback(message):
                    status_text.text(message)
                
                try:
                    # Reconstruct portfolio value
                    hist_data = reconstruct_portfolio_value(
                        orders,
                        start_date,
                        end_date,
                        progress_callback
                    )
                    
                    progress_bar.progress(100)
                    status_text.text("✅ Reconstruction complete!")
                    
                    if not hist_data.empty:
                        # Create detailed chart
                        fig = go.Figure()
                        
                        fig.add_trace(go.Scatter(
                            x=hist_data['Date'],
                            y=hist_data['Portfolio_Value_CZK'],
                            name='Portfolio Value',
                            line=dict(color='#2ecc71', width=2),
                            fill='tozeroy',
                            fillcolor='rgba(46, 204, 113, 0.1)'
                        ))
                        
                        fig.update_layout(
                            title="Portfolio Value Over Time (Reconstructed from Yahoo Finance)",
                            xaxis_title="Date",
                            yaxis_title="Value (CZK)",
                            height=500,
                            hovermode='x unified'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show statistics
                        st.success(f"📊 Calculated {len(hist_data)} days of portfolio data")
                    else:
                        st.error("Failed to reconstruct portfolio value")
                        
                except Exception as e:
                    st.error(f"Error during reconstruction: {e}")
                    st.exception(e)
                    
            else:
                # Order history only
                hist_data = calculate_portfolio_value_over_time(orders_df, current_positions)
                fig = create_pnl_chart(hist_data, current_total, current_invested)
                st.plotly_chart(fig, use_container_width=True)
            
            # Show recent orders table
            with st.expander("📋 Recent Orders"):
                recent_orders = orders_df.tail(10)[[
                    'date', 'ticker', 'type', 'quantity', 'price', 'value'
                ]].copy()
                recent_orders['date'] = recent_orders['date'].dt.strftime('%Y-%m-%d %H:%M')
                st.dataframe(recent_orders, use_container_width=True, hide_index=True)
            
        except Exception as e:
            st.error(f"Error loading historical data: {str(e)}")
            st.exception(e)


def _as_float(x, default: float = 0.0) -> float:
    """Convert various data types to float, including nested dicts."""
    try:
        if isinstance(x, dict) and 'value' in x:
            return float(x['value'])
        return float(x) if x is not None else default
    except (ValueError, TypeError):
        return default
