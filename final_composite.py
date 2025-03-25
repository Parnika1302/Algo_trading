#@title SMA Strategy Backtest Code (Complete)
import pandas as pd
import yfinance as yf
import numpy as np
from itertools import product
from datetime import datetime, timedelta

# List of tickers to analyze
tickers = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", 
    "HINDUNILVR.NS", "ITC.NS", "BHARTIARTL.NS", "SBIN.NS", "KOTAKBANK.NS",
    "LT.NS", "BAJFINANCE.NS", "HCLTECH.NS", "ASIANPAINT.NS", "MARUTI.NS", 
    "AXISBANK.NS", "SUNPHARMA.NS", "NESTLEIND.NS", "TITAN.NS", "ULTRACEMCO.NS",
    "TATAMOTORS.NS", "WIPRO.NS", "POWERGRID.NS", "ADANIENT.NS", "ADANIGREEN.NS", 
    "NTPC.NS", "JSWSTEEL.NS", "TECHM.NS", "GRASIM.NS", "TATASTEEL.NS",
    "INDUSINDBK.NS", "M&M.NS", "HDFCLIFE.NS", "BAJAJFINSV.NS", "EICHERMOT.NS", 
    "COALINDIA.NS", "ONGC.NS", "DIVISLAB.NS", "BPCL.NS", "CIPLA.NS",
    "BRITANNIA.NS", "APOLLOHOSP.NS", "HEROMOTOCO.NS", "DRREDDY.NS", "HINDALCO.NS", 
    "BAJAJ-AUTO.NS", "TATACONSUM.NS", "SBILIFE.NS", "UPL.NS", "IOC.NS"
]

# Parameters
start_date = '2014-01-01'
end_date = datetime.now().strftime('%Y-%m-%d')  # real-time end date
initial_investment = 10000  # INR per buy call
sma_windows = [9, 21, 50, 100, 200]
thresholds = [0.02, 0.03, 0.05, 0.1]
lookback_period = timedelta(weeks=2)  # past two weeks

def backtest_strategy(data, buy_signals, sell_signals, initial_investment=10000):
    position = 0  # Number of shares held
    cash = initial_investment
    buy_sell_log = []  # To record the trades
    trades = []       # To record trade details
    
    for i in range(len(data)):
        # Check for a buy signal
        if not np.isnan(buy_signals[i]):
            price = data['Close'].iloc[i]
            position = int(cash // price)
            if position == 0:
                continue
            cash -= position * price
            buy_sell_log.append({
                'Date': data.index[i],
                'Action': 'Buy',
                'Price': price,
                'Shares': position
            })
        # Check for a sell signal
        elif not np.isnan(sell_signals[i]) and position > 0 and len(buy_sell_log) > 0:
            price = data['Close'].iloc[i]
            cash += position * price
            profit_per_share = price - buy_sell_log[-1]['Price']
            total_profit = profit_per_share * position
            holding_period = (data.index[i] - buy_sell_log[-1]['Date']).days
            trades.append({
                'Buy Date': buy_sell_log[-1]['Date'],
                'Sell Date': data.index[i],
                'Buy Price': buy_sell_log[-1]['Price'],
                'Sell Price': price,
                'Profit': total_profit,
                'Holding Period': holding_period,
                'Winning Trade': total_profit > 0,
                'Shares': position
            })
            buy_sell_log.append({
                'Date': data.index[i],
                'Action': 'Sell',
                'Price': price,
                'Shares': position
            })
            position = 0

    # Close any open positions at the end of data
    if position > 0 and len(buy_sell_log) > 0:
        price = data['Close'].iloc[-1]
        cash += position * price
        profit_per_share = price - buy_sell_log[-1]['Price']
        total_profit = profit_per_share * position
        holding_period = (data.index[-1] - buy_sell_log[-1]['Date']).days
        trades.append({
            'Buy Date': buy_sell_log[-1]['Date'],
            'Sell Date': data.index[-1],
            'Buy Price': buy_sell_log[-1]['Price'],
            'Sell Price': price,
            'Profit': total_profit,
            'Holding Period': holding_period,
            'Winning Trade': total_profit > 0,
            'Shares': position
        })
        buy_sell_log.append({
            'Date': data.index[-1],
            'Action': 'Sell',
            'Price': price,
            'Shares': position
        })
        position = 0

    winning_trades = [trade for trade in trades if trade['Profit'] > 0]
    losing_trades  = [trade for trade in trades if trade['Profit'] <= 0]

    avg_holding_period = np.mean([trade['Holding Period'] for trade in trades]) if trades else 0
    avg_holding_period_wins = np.mean([trade['Holding Period'] for trade in winning_trades]) if winning_trades else 0
    avg_holding_period_losses = np.mean([trade['Holding Period'] for trade in losing_trades]) if losing_trades else 0

    profit_percent_per_win = np.mean([(trade['Profit']/(trade['Buy Price']*trade['Shares']))*100 for trade in winning_trades]) if winning_trades else 0
    loss_percent_per_loss = np.mean([(abs(trade['Profit'])/(trade['Buy Price']*trade['Shares']))*100 for trade in losing_trades]) if losing_trades else 0

    final_portfolio_value = cash
    total_profit = final_portfolio_value - initial_investment

    # Filter buy/sell log for the past two weeks
    two_weeks_ago = datetime.now() - lookback_period
    recent_buy_sell_log = [log for log in buy_sell_log if log['Date'] >= two_weeks_ago]

    return (final_portfolio_value, total_profit, recent_buy_sell_log, 
            len(winning_trades), len(losing_trades), avg_holding_period, 
            avg_holding_period_wins, avg_holding_period_losses, 
            profit_percent_per_win, loss_percent_per_loss)

def dual_sma_crossover(data, short_window, long_window):
    data[f'SMA{short_window}'] = data['Close'].rolling(window=short_window, min_periods=1).mean()
    data[f'SMA{long_window}'] = data['Close'].rolling(window=long_window, min_periods=1).mean()
    buy_signals = [np.nan] * len(data)
    sell_signals = [np.nan] * len(data)
    position = False

    for i in range(len(data)):
        if data[f'SMA{short_window}'].iloc[i] > data[f'SMA{long_window}'].iloc[i]:
            if not position:
                buy_signals[i] = data['Close'].iloc[i]
                position = True
        elif data[f'SMA{short_window}'].iloc[i] < data[f'SMA{long_window}'].iloc[i]:
            if position:
                sell_signals[i] = data['Close'].iloc[i]
                position = False
    return buy_signals, sell_signals

def sma_support_resistance(data, sma_window):
    data[f'SMA{sma_window}'] = data['Close'].rolling(window=sma_window, min_periods=1).mean()
    buy_signals = [np.nan] * len(data)
    sell_signals = [np.nan] * len(data)
    position = False

    for i in range(len(data)):
        if data['Close'].iloc[i] > data[f'SMA{sma_window}'].iloc[i]:
            if not position:
                buy_signals[i] = data['Close'].iloc[i]
                position = True
        elif data['Close'].iloc[i] < data[f'SMA{sma_window}'].iloc[i]:
            if position:
                sell_signals[i] = data['Close'].iloc[i]
                position = False
    return buy_signals, sell_signals

def sma_filtered_breakout(data, sma_window):
    data[f'SMA{sma_window}'] = data['Close'].rolling(window=sma_window, min_periods=1).mean()
    buy_signals = [np.nan] * len(data)
    sell_signals = [np.nan] * len(data)
    position = False

    for i in range(1, len(data)):
        if data['Close'].iloc[i] > data['Close'].iloc[i-1] and data['Close'].iloc[i] > data[f'SMA{sma_window}'].iloc[i]:
            if not position:
                buy_signals[i] = data['Close'].iloc[i]
                position = True
        elif data['Close'].iloc[i] < data['Close'].iloc[i-1] and data['Close'].iloc[i] < data[f'SMA{sma_window}'].iloc[i]:
            if position:
                sell_signals[i] = data['Close'].iloc[i]
                position = False
    return buy_signals, sell_signals

def sma_mean_reversion(data, sma_window, threshold):
    data[f'SMA{sma_window}'] = data['Close'].rolling(window=sma_window, min_periods=1).mean()
    buy_signals = [np.nan] * len(data)
    sell_signals = [np.nan] * len(data)
    position = False

    for i in range(len(data)):
        if data['Close'].iloc[i] < data[f'SMA{sma_window}'].iloc[i] * (1 - threshold):
            if not position:
                buy_signals[i] = data['Close'].iloc[i]
                position = True
        elif data['Close'].iloc[i] > data[f'SMA{sma_window}'].iloc[i] * (1 + threshold):
            if position:
                sell_signals[i] = data['Close'].iloc[i]
                position = False
    return buy_signals, sell_signals

all_results = []

for ticker in tickers:
    print(f"Processing {ticker}...")
    stock_data = yf.download(ticker, start=start_date, end=end_date)

    if stock_data.empty:
        print(f"No data found for {ticker}. Skipping.")
        continue

    # If the columns are a MultiIndex, flatten them
    if isinstance(stock_data.columns, pd.MultiIndex):
        stock_data.columns = stock_data.columns.get_level_values(0)
    
    # Ensure 'Close' exists; if not, try to rename 'Adj Close'
    if 'Close' not in stock_data.columns:
        if 'Adj Close' in stock_data.columns:
            stock_data.rename(columns={'Adj Close': 'Close'}, inplace=True)
        else:
            print(f"Ticker {ticker} has no 'Close' or 'Adj Close'. Skipping.")
            continue

    # Drop rows with missing 'Close'
    stock_data.dropna(subset=['Close'], inplace=True)

    # Define parameter combinations for dual SMA crossover strategy
    param_combinations = [(short, long) for short, long in product(sma_windows, sma_windows) if short < long]
    results = []

    # Strategy 1: Dual SMA Crossover
    for params in param_combinations:
        buy_signals, sell_signals = dual_sma_crossover(stock_data.copy(), *params)
        (final_value, profit, recent_buy_sell_log, num_wins, num_losses,
         avg_hold, avg_hold_wins, avg_hold_losses, profit_per_win, loss_per_loss) = backtest_strategy(stock_data, buy_signals, sell_signals)
        results.append({
            'Ticker': ticker,
            'Strategy': 'Dual SMA Crossover',
            'Params': str(params),
            'Final Portfolio Value': final_value,
            'Total Profit': profit,
            'Winning Trades': num_wins,
            'Losing Trades': num_losses,
            'Avg Holding Period': avg_hold,
            'Avg Holding Period Wins': avg_hold_wins,
            'Avg Holding Period Losses': avg_hold_losses,
            'Profit Percent per Win': profit_per_win,
            'Loss Percent per Loss': loss_per_loss,
            'Buy/Sell Log': recent_buy_sell_log
        })

    # Strategy 2: SMA Support & Resistance
    for window in sma_windows:
        buy_signals, sell_signals = sma_support_resistance(stock_data.copy(), window)
        (final_value, profit, recent_buy_sell_log, num_wins, num_losses,
         avg_hold, avg_hold_wins, avg_hold_losses, profit_per_win, loss_per_loss) = backtest_strategy(stock_data, buy_signals, sell_signals)
        results.append({
            'Ticker': ticker,
            'Strategy': 'SMA Support & Resistance',
            'Params': str((window,)),
            'Final Portfolio Value': final_value,
            'Total Profit': profit,
            'Winning Trades': num_wins,
            'Losing Trades': num_losses,
            'Avg Holding Period': avg_hold,
            'Avg Holding Period Wins': avg_hold_wins,
            'Avg Holding Period Losses': avg_hold_losses,
            'Profit Percent per Win': profit_per_win,
            'Loss Percent per Loss': loss_per_loss,
            'Buy/Sell Log': recent_buy_sell_log
        })

    # Strategy 3: SMA Filtered Breakout
    for window in sma_windows:
        buy_signals, sell_signals = sma_filtered_breakout(stock_data.copy(), window)
        (final_value, profit, recent_buy_sell_log, num_wins, num_losses,
         avg_hold, avg_hold_wins, avg_hold_losses, profit_per_win, loss_per_loss) = backtest_strategy(stock_data, buy_signals, sell_signals)
        results.append({
            'Ticker': ticker,
            'Strategy': 'SMA Filtered Breakout',
            'Params': str((window,)),
            'Final Portfolio Value': final_value,
            'Total Profit': profit,
            'Winning Trades': num_wins,
            'Losing Trades': num_losses,
            'Avg Holding Period': avg_hold,
            'Avg Holding Period Wins': avg_hold_wins,
            'Avg Holding Period Losses': avg_hold_losses,
            'Profit Percent per Win': profit_per_win,
            'Loss Percent per Loss': loss_per_loss,
            'Buy/Sell Log': recent_buy_sell_log
        })

    # Strategy 4: SMA Mean Reversion
    for window, thresh in product(sma_windows, thresholds):
        buy_signals, sell_signals = sma_mean_reversion(stock_data.copy(), window, thresh)
        (final_value, profit, recent_buy_sell_log, num_wins, num_losses,
         avg_hold, avg_hold_wins, avg_hold_losses, profit_per_win, loss_per_loss) = backtest_strategy(stock_data, buy_signals, sell_signals)
        results.append({
            'Ticker': ticker,
            'Strategy': 'SMA Mean Reversion',
            'Params': str((window, thresh)),
            'Final Portfolio Value': final_value,
            'Total Profit': profit,
            'Winning Trades': num_wins,
            'Losing Trades': num_losses,
            'Avg Holding Period': avg_hold,
            'Avg Holding Period Wins': avg_hold_wins,
            'Avg Holding Period Losses': avg_hold_losses,
            'Profit Percent per Win': profit_per_win,
            'Loss Percent per Loss': loss_per_loss,
            'Buy/Sell Log': recent_buy_sell_log
        })

    all_results.extend(results)
# ... [everything up through all_results.extend(results)] ...

# Convert results to DataFrame
results_df = pd.DataFrame(all_results)

# Calculate Win Trade Ratio
results_df['Win Trade Ratio'] = results_df.apply(
    lambda row: row['Winning Trades'] / (row['Winning Trades'] + row['Losing Trades'])
    if (row['Winning Trades'] + row['Losing Trades']) > 0 else 0,
    axis=1
)

# Calculate Composite Score (50% weight on Win Trade Ratio, 50% on Portfolio Growth)
results_df['Composite Score'] = 0.5 * results_df['Win Trade Ratio'] + \
                                0.5 * (results_df['Final Portfolio Value'] / initial_investment)

# Save results to an Excel file with two sheets
# Save results (Strategy Results + Buy/Sell Log + Top 3 Strategies) to a single Excel file
excel_filename = "SMA_Strategy_Results_All_Strategies_newest.xlsx"
with pd.ExcelWriter(excel_filename) as writer:
    # 1️⃣ Strategy Results sheet (exclude Buy/Sell Log)
    results_df.drop(columns=['Buy/Sell Log']).to_excel(writer, sheet_name='Strategy Results', index=False)

    # 2️⃣ Buy/Sell Log sheet
    logs = []
    for result in all_results:
        for log in result['Buy/Sell Log']:
            logs.append({
                'Ticker': result['Ticker'],
                'Strategy': result['Strategy'],
                'Params': result['Params'],
                'Date': log['Date'],
                'Action': log['Action'],
                'Price': log['Price'],
                'Shares': log['Shares']
            })
    pd.DataFrame(logs).to_excel(writer, sheet_name='Buy_Sell_Log', index=False)

    # 3️⃣ Top 3 Strategies sheet (by Composite Score per Ticker)
    top3_by_ticker = (
        results_df
        .sort_values(['Ticker','Composite Score'], ascending=[True, False])
        .groupby('Ticker')
        .head(3)
        .reset_index(drop=True)
    )
    top3_by_ticker.to_excel(writer, sheet_name='Top 3 Strategies', index=False)

print(f"Results (including Top 3 strategies) saved to '{excel_filename}'.")
