import os 
os.system('cls')

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sp500_table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
tickers = sp500_table[0]['Symbol'].tolist()  # full list

# Backtest parameters
start_date = '2022-01-01'
end_date = pd.Timestamp.today().strftime('%Y-%m-%d')
fast_window = 50
slow_window = 200

def backtest_stock(ticker, start_date, end_date, fast_window, slow_window):
    """
    Downloads historical data for a ticker, computes lagged SMAs, generates signals,
    logs trades (as dictionaries), and computes daily active returns when the stock is held.
    """
    try:
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    except Exception as e:
        print(f"Error downloading {ticker}: {e}")
        return None, []
    if df.empty:
        return None, []
    
    df = df[['Close']].rename(columns={'Close': 'price'})
    
    # Compute lagged SMAs (using only data up to yesterday)
    df['SMA_fast'] = df['price'].shift(1).rolling(window=fast_window, min_periods=fast_window).mean()
    df['SMA_slow'] = df['price'].shift(1).rolling(window=slow_window, min_periods=slow_window).mean()
    
    # Generate signal: 1 when lagged SMA_fast > lagged SMA_slow, else 0
    df['signal'] = 0
    df.loc[df['SMA_fast'] > df['SMA_slow'], 'signal'] = 1
    df['position'] = df['signal']
    
    # Calculate trade events: +1 is BUY, -1 is SELL
    df['trade_signal'] = df['position'].diff().fillna(0)
    trade_log = []
    for date, row in df.iterrows():
        ts_val = np.array(row['trade_signal']).item() if isinstance(row['trade_signal'], (pd.Series, np.ndarray)) else float(row['trade_signal'])
        price_val = np.array(row['price']).item() if isinstance(row['price'], (pd.Series, np.ndarray)) else float(row['price'])
        
        if ts_val == 1:
            trade_log.append({
                "Date": date,
                "Ticker": ticker,
                "Action": "BUY",
                "Price": price_val
            })
        elif ts_val == -1:
            trade_log.append({
                "Date": date,
                "Ticker": ticker,
                "Action": "SELL",
                "Price": price_val
            })
    
    df['market_return'] = df['price'].pct_change()
    df['active_return'] = np.where(df['position'] == 1, df['market_return'], np.nan)
    
    return df, trade_log

results = {}
all_trade_logs = []
active_returns_list = []

for ticker in tickers:
    df, trade_log = backtest_stock(ticker, start_date, end_date, fast_window, slow_window)
    if df is not None:
        results[ticker] = df
        all_trade_logs.extend(trade_log)
        active_returns_list.append(df['active_return'].rename(ticker))

portfolio_returns = pd.concat(active_returns_list, axis=1)
portfolio_daily_return = portfolio_returns.mean(axis=1, skipna=True)
portfolio_daily_return = portfolio_daily_return.fillna(0)
cum_portfolio = (1 + portfolio_daily_return).cumprod()

spy_data = yf.download('SPY', start=start_date, end=end_date, progress=False)
spy_data = spy_data[['Close']].rename(columns={'Close': 'price'})
spy_data['market_return'] = spy_data['price'].pct_change()
cum_spy = (1 + spy_data['market_return']).cumprod()

def annualized_return(cum_return, days):
    return cum_return ** (252 / days) - 1

def sharpe_ratio(daily_returns):
    daily_returns = daily_returns.dropna()
    return np.sqrt(252) * daily_returns.mean() / daily_returns.std()

def max_drawdown(cum_returns):
    running_max = cum_returns.cummax()
    drawdown = (cum_returns - running_max) / running_max
    return drawdown.min()

total_days = (cum_portfolio.index[-1] - cum_portfolio.index[0]).days
ann_ret_portfolio = annualized_return(cum_portfolio.iloc[-1], total_days)
total_return_portfolio = (cum_portfolio.iloc[-1] - 1) * 100
sharpe_portfolio = sharpe_ratio(portfolio_daily_return)
max_dd_portfolio = max_drawdown(cum_portfolio) * 100

total_days_spy = (cum_spy.index[-1] - cum_spy.index[0]).days
ann_ret_spy = annualized_return(cum_spy.iloc[-1], total_days_spy)
total_return_spy = (cum_spy.iloc[-1] - 1) * 100
sharpe_spy = sharpe_ratio(spy_data['market_return'])
max_dd_spy = max_drawdown(cum_spy) * 100

print("Algo Performance:")
print(f"  Total Return: {total_return_portfolio:.2f}%")
print(f"  Annualized Return: {ann_ret_portfolio:.2%}")
print(f"  Sharpe Ratio: {sharpe_portfolio:.2f}")
print(f"  Maximum Drawdown: {max_dd_portfolio:.2f}%\n")

print("Buy & Hold (SPY) Performance:")
print(f"  Total Return: {total_return_spy:.2f}%")
print(f"  Annualized Return: {ann_ret_spy:.2%}")
print(f"  Sharpe Ratio: {sharpe_spy:.2f}")
print(f"  Maximum Drawdown: {max_dd_spy:.2f}%\n")

trade_log_df = pd.DataFrame(all_trade_logs)
trade_log_df.sort_values(by="Date", inplace=True)
trade_log_df.to_excel(r"C:\Users\Nihal K\Documents\Trade_Log.xlsx", index=False)
print("Trade log has been saved to 'C:\\Users\\Nihal K\\Documents\\Trade_Log.xlsx'.")

plt.figure(figsize=(12, 6))
plt.plot(cum_spy.index, cum_spy, label='Buy & Hold (SPY)')
plt.plot(cum_portfolio.index, cum_portfolio, label='Algo Strategy')
plt.title(f'Portfolio vs. SPY Cumulative Returns from {start_date} to {end_date}')
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.legend()
plt.grid(True)
plt.show()