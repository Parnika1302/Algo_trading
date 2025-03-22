import pandas as pd
import yfinance as yf
import numpy as np
from itertools import product
from datetime import datetime, timedelta

# List of tickers to analyze
tickers = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", "HINDUNILVR.NS", "ITC.NS", "BHARTIARTL.NS", "SBIN.NS", "KOTAKBANK.NS",
           "LT.NS", "BAJFINANCE.NS", "HCLTECH.NS", "ASIANPAINT.NS", "MARUTI.NS", "AXISBANK.NS", "SUNPHARMA.NS", "NESTLEIND.NS", "TITAN.NS", "ULTRACEMCO.NS",
           "TATAMOTORS.NS", "WIPRO.NS", "POWERGRID.NS", "ADANIENT.NS", "ADANIGREEN.NS", "NTPC.NS", "JSWSTEEL.NS", "TECHM.NS", "GRASIM.NS", "TATASTEEL.NS",
           "INDUSINDBK.NS", "M&M.NS", "HDFCLIFE.NS", "BAJAJFINSV.NS", "EICHERMOT.NS", "COALINDIA.NS", "ONGC.NS", "DIVISLAB.NS", "BPCL.NS", "CIPLA.NS",
           "BRITANNIA.NS", "APOLLOHOSP.NS", "HEROMOTOCO.NS", "DRREDDY.NS", "HINDALCO.NS", "BAJAJ-AUTO.NS", "TATACONSUM.NS", "SBILIFE.NS", "UPL.NS", "IOC.NS"]

# Parameters
start_date = '2014-01-01'
end_date = datetime.now().strftime('%Y-%m-%d')  # Set end date to real-time
initial_investment = 10000  # INR per buy call
sma_windows = [9, 21, 50, 100, 200]  # Different SMA windows to test
thresholds = [0.02, 0.03, 0.05, 0.1]  # Different thresholds for mean reversion
lookback_period = timedelta(weeks=2)  # Past two weeks lookback period

# Backtesting Function with Corrected Metrics
def backtest_strategy(data, buy_signals, sell_signals, initial_investment=10000):
    investment = initial_investment
    position = 0  # Number of shares currently held
    cash = investment
    buy_sell_log = []  # Log to store buy and sell signals with dates
    trades = []  # Store trades (buy and sell info for analysis)

    for i in range(len(data)):
        if not np.isnan(buy_signals[i]):  # Check for NaN using np.isnan()
            price = data['Close'].iloc[i]
            position = cash // price  # Number of shares bought
            if position == 0:
                continue  # Skip if not enough cash to buy at least one share
            cash -= position * price
            buy_sell_log.append({
                'Date': data.index[i],
                'Action': 'Buy',
                'Price': price,
                'Shares': position
            })

        elif not np.isnan(sell_signals[i]) and position > 0 and len(buy_sell_log) > 0:  # Sell signal
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
                'Shares': position  # Add 'Shares' to the trade dictionary
            })
            buy_sell_log.append({
                'Date': data.index[i],
                'Action': 'Sell',
                'Price': price,
                'Shares': position
            })
            position = 0  # Reset position after sell

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
            'Shares': position  # Add 'Shares' to the trade dictionary
        })
        buy_sell_log.append({
            'Date': data.index[-1],
            'Action': 'Sell',
            'Price': price,
            'Shares': position
        })
        position = 0

    # Calculate trade metrics
    winning_trades = [trade for trade in trades if trade['Profit'] > 0]
    losing_trades = [trade for trade in trades if trade['Profit'] <= 0]

    num_winning_trades = len(winning_trades)
    num_losing_trades = len(losing_trades)

    avg_holding_period = np.mean([trade['Holding Period'] for trade in trades]) if trades else 0
    avg_holding_period_wins = np.mean([trade['Holding Period'] for trade in winning_trades]) if winning_trades else 0
    avg_holding_period_losses = np.mean([trade['Holding Period'] for trade in losing_trades]) if losing_trades else 0

    # Calculate Profit Percent per Win and Loss Percent per Loss
    profit_percent_per_win = np.mean([(trade['Profit'] / (trade['Buy Price'] * trade['Shares'])) * 100 for trade in winning_trades]) if winning_trades else 0
    loss_percent_per_loss = np.mean([(abs(trade['Profit']) / (trade['Buy Price'] * trade['Shares'])) * 100 for trade in losing_trades]) if losing_trades else 0

    final_portfolio_value = cash
    total_profit = final_portfolio_value - initial_investment

    # Filter buy/sell log for the past two weeks
    two_weeks_ago = datetime.now() - lookback_period
    recent_buy_sell_log = [log for log in buy_sell_log if log['Date'] >= two_weeks_ago]

    return (final_portfolio_value, total_profit, recent_buy_sell_log, num_winning_trades, num_losing_trades,
            avg_holding_period, avg_holding_period_wins, avg_holding_period_losses, profit_percent_per_win, loss_percent_per_loss)
            
# Strategy Functions
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
        if data['Close'].iloc[i] > data['Close'].iloc[i - 1] and data['Close'].iloc[i] > data[f'SMA{sma_window}'].iloc[i]:
            if not position:
                buy_signals[i] = data['Close'].iloc[i]
                position = True
        elif data['Close'].iloc[i] < data['Close'].iloc[i - 1] and data['Close'].iloc[i] < data[f'SMA{sma_window}'].iloc[i]:
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

# Optimizing strategies for each ticker
all_results = []

for ticker in tickers:
    print(f"Processing {ticker}...")
    stock_data = yf.download(ticker, start=start_date, end=end_date)

    if stock_data.empty:
        print(f"No data found for {ticker}. Skipping.")
        continue

    param_combinations = list(product(sma_windows, sma_windows))
    param_combinations = [(short, long) for short, long in param_combinations if short < long]

    results = []

    # 1. Dual SMA Crossover Strategy
    for params in param_combinations:
        buy_signals, sell_signals = dual_sma_crossover(stock_data, *params)
        (final_value, profit, recent_buy_sell_log, num_winning_trades, num_losing_trades,
         avg_holding_period, avg_holding_period_wins, avg_holding_period_losses, profit_percent_per_win, loss_percent_per_loss) = backtest_strategy(stock_data, buy_signals, sell_signals)
        results.append({
            'Ticker': ticker,
            'Strategy': 'Dual SMA Crossover',
            'Params': params,
            'Final Portfolio Value': final_value,
            'Total Profit': profit,
            'Winning Trades': num_winning_trades,
            'Losing Trades': num_losing_trades,
            'Avg Holding Period': avg_holding_period,
            'Avg Holding Period Wins': avg_holding_period_wins,
            'Avg Holding Period Losses': avg_holding_period_losses,
            'Profit Percent per Win': profit_percent_per_win,
            'Loss Percent per Loss': loss_percent_per_loss,
            'Buy/Sell Log': recent_buy_sell_log
        })

    # 2. SMA Support & Resistance Strategy
    for sma_window in sma_windows:
        buy_signals, sell_signals = sma_support_resistance(stock_data, sma_window)
        (final_value, profit, recent_buy_sell_log, num_winning_trades, num_losing_trades,
         avg_holding_period, avg_holding_period_wins, avg_holding_period_losses, profit_percent_per_win, loss_percent_per_loss) = backtest_strategy(stock_data, buy_signals, sell_signals)
        results.append({
            'Ticker': ticker,
            'Strategy': 'SMA Support & Resistance',
            'Params': (sma_window,),
            'Final Portfolio Value': final_value,
            'Total Profit': profit,
            'Winning Trades': num_winning_trades,
            'Losing Trades': num_losing_trades,
            'Avg Holding Period': avg_holding_period,
            'Avg Holding Period Wins': avg_holding_period_wins,
            'Avg Holding Period Losses': avg_holding_period_losses,
            'Profit Percent per Win': profit_percent_per_win,
            'Loss Percent per Loss': loss_percent_per_loss,
            'Buy/Sell Log': recent_buy_sell_log
        })

    # 3. SMA Filtered Breakout Strategy
    for sma_window in sma_windows:
        buy_signals, sell_signals = sma_filtered_breakout(stock_data, sma_window)
        (final_value, profit, recent_buy_sell_log, num_winning_trades, num_losing_trades,
         avg_holding_period, avg_holding_period_wins, avg_holding_period_losses, profit_percent_per_win, loss_percent_per_loss) = backtest_strategy(stock_data, buy_signals, sell_signals)
        results.append({
            'Ticker': ticker,
            'Strategy': 'SMA Filtered Breakout',
            'Params': (sma_window,),
            'Final Portfolio Value': final_value,
            'Total Profit': profit,
            'Winning Trades': num_winning_trades,
            'Losing Trades': num_losing_trades,
            'Avg Holding Period': avg_holding_period,
            'Avg Holding Period Wins': avg_holding_period_wins,
            'Avg Holding Period Losses': avg_holding_period_losses,
            'Profit Percent per Win': profit_percent_per_win,
            'Loss Percent per Loss': loss_percent_per_loss,
            'Buy/Sell Log': recent_buy_sell_log
        })

    # 4. SMA Mean Reversion Strategy
    for sma_window, threshold in product(sma_windows, thresholds):
        buy_signals, sell_signals = sma_mean_reversion(stock_data, sma_window, threshold)
        (final_value, profit, recent_buy_sell_log, num_winning_trades, num_losing_trades,
         avg_holding_period, avg_holding_period_wins, avg_holding_period_losses, profit_percent_per_win, loss_percent_per_loss) = backtest_strategy(stock_data, buy_signals, sell_signals)
        results.append({
            'Ticker': ticker,
            'Strategy': 'SMA Mean Reversion',
            'Params': (sma_window, threshold),
            'Final Portfolio Value': final_value,
            'Total Profit': profit,
            'Winning Trades': num_winning_trades,
            'Losing Trades': num_losing_trades,
            'Avg Holding Period': avg_holding_period,
            'Avg Holding Period Wins': avg_holding_period_wins,
            'Avg Holding Period Losses': avg_holding_period_losses,
            'Profit Percent per Win': profit_percent_per_win,
            'Loss Percent per Loss': loss_percent_per_loss,
            'Buy/Sell Log': recent_buy_sell_log
        })

    all_results.extend(results)

# Convert all results to a DataFrame
results_df = pd.DataFrame(all_results)

# Save to Excel file with two sheets: one for strategy results, one for buy/sell logs
excel_filename = "SMA_Strategy_Results_All_Strategies_newest.xlsx"
with pd.ExcelWriter(excel_filename) as writer:
    # Strategy results
    results_df.drop(columns=['Buy/Sell Log']).to_excel(writer, sheet_name='Strategy Results', index=False)

    # Buy/Sell Logs
    logs = []
    for result in all_results:
        for log in result['Buy/Sell Log']:
            log_entry = {
                'Ticker': result['Ticker'],
                'Strategy': result['Strategy'],
                'Params': result['Params'],
                'Date': log['Date'],
                'Action': log['Action'],
                'Price': log['Price'],
                'Shares': log['Shares']
            }
            logs.append(log_entry)
    logs_df = pd.DataFrame(logs)
    logs_df.to_excel(writer, sheet_name='Buy_Sell_Log', index=False)

print(f"Results have been saved to '{excel_filename}'.")