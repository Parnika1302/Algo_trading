import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

# Use today as the exit date
exit_date = datetime.today().strftime('%Y-%m-%d')

# Entry signals come from two weeks ago
entry_date = (datetime.today() - timedelta(days=14)).strftime('%Y-%m-%d')

# Fetch 30 days of history to compute moving averages
historical_start = (datetime.today() - timedelta(days=30)).strftime('%Y-%m-%d')


# === STATIC TRADE SIGNALS (all trades from 5 Mar 2025 → 20 Mar 2025) ===
trade_signals = pd.DataFrame({
    'Symbol': ['KOTAKBANK.NS', 'TECHM.NS', 'POWERGRID.NS', 'SBILIFE.NS', 'NTPC.NS',
               'LT.NS', 'M&M.NS', 'EICHERMOT.NS', 'INFY.NS', 'MARUTI.NS', 'HINDALCO.NS',
               'TATASTEEL.NS', 'TECHM.NS', 'GRASIM.NS', 'ADANIENT.NS', 'TECHM.NS',
               'HCLTECH.NS', 'RELIANCE.NS', 'COALINDIA.NS', 'BPCL.NS', 'ASIANPAINT.NS', 'SBIN.NS'],
    'Crossover': ['Buy', 'Sell', 'Buy', 'Sell', 'Buy',
                  'Buy', 'Buy', 'Buy', 'Sell', 'Buy',
                  'Buy', 'Buy', 'Buy', 'Sell', 'Buy',
                  'Sell', 'Sell', 'Buy', 'Buy', 'Buy', 'Sell', 'Buy'],
    'Crossover Date': [entry_date] * 22,
    'Signal Effective Date': [exit_date] * 22
})

def validate_date(date_str):
    date = datetime.strptime(date_str, '%Y-%m-%d')
    simulation_end = datetime.strptime(exit_date, '%Y-%m-%d')
    if date > simulation_end:
        return simulation_end.strftime('%Y-%m-%d')
    return date_str

def fetch_historical_data(symbol, date):
    ticker = yf.Ticker(symbol.upper())
    date = validate_date(date)
    start = datetime.strptime(date, '%Y-%m-%d')
    end = (start + timedelta(days=1)).strftime('%Y-%m-%d')
    try:
        df = ticker.history(start=start.strftime('%Y-%m-%d'), end=end)
        return df if not df.empty else None
    except Exception as e:
        st.error(f"{symbol}: error fetching historical data on {date}: {e}")
        return None

def fetch_live_price(symbol):
    ticker = yf.Ticker(symbol.upper())
    start = entry_date
    end = exit_date
    try:
        df = ticker.history(start=start, end=end)
        return df['Close'].iloc[-1] if not df.empty else None
    except Exception as e:
        st.error(f"{symbol}: error fetching simulated live price: {e}")
        return None

def calculate_pnl(signals):
    rows = []
    for _, r in signals.iterrows():
        symbol, cross = r['Symbol'], r['Crossover']
        entry_df = fetch_historical_data(symbol, r['Crossover Date'])
        exit_price = fetch_live_price(symbol)
        if entry_df is None or exit_price is None:
            st.warning(f"{symbol}: skipping (missing data)")
            continue

        entry_price = entry_df['Close'].iloc[0] if cross == 'Buy' else entry_df['Open'].iloc[0]
        pnl = (exit_price - entry_price) if cross == 'Buy' else (entry_price - exit_price)
        pnl_pct = pnl / entry_price * 100

        rows.append({
            'Symbol': symbol,
            'Crossover': cross,
            'Entry Price': entry_price,
            'Exit Price': exit_price,
            'P&L': pnl,
            'P&L (%)': pnl_pct
        })

    return pd.DataFrame(rows)

st.title(f"Paper Trading Simulation ({entry_date} → {exit_date}")
with st.spinner("Calculating P&L..."):
    portfolio_df = calculate_pnl(trade_signals)

if portfolio_df.empty:
    st.write("No simulated trades.")
else:
    # Compute totals on numeric data
    total_pnl = portfolio_df['P&L'].sum()
    total_pct = portfolio_df['P&L (%)'].sum()

    # Format copy for display
    display_df = portfolio_df.copy()
    for col in ['Entry Price', 'Exit Price', 'P&L', 'P&L (%)']:
        display_df[col] = display_df[col].map("{:.2f}".format)

    st.subheader("Open Positions & P&L")
    st.dataframe(display_df.set_index('Symbol'))
    st.markdown(f"**Total P&L:** {total_pnl:.2f} INR ({total_pct:.2f}%)")

    # Build a filename that includes today’s date
    filename = f"simulated_portfolio_{datetime.today().strftime('%Y-%m-%d')}.csv"


    st.download_button(
        "Download Portfolio CSV",
        portfolio_df.to_csv(index=False).encode('utf-8'),
        filename ,
        "text/csv"
    )

    st.subheader("P&L per Position")
    st.bar_chart(portfolio_df.set_index('Symbol')['P&L'])
