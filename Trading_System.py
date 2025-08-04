import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Function to compute MACD
def compute_macd(Stock_Data,use_sma,short_window=12, long_window=26, signal_window=9):
    if use_sma==True:
        # Using Simple Moving Average 
        Stock_Data['SMA'] = Stock_Data['Close'].rolling(window=short_window).mean()
        Stock_Data['LMA'] = Stock_Data['Close'].rolling(window=long_window).mean()
    else:
        # Using Exponential Moving Average
        Stock_Data['SMA'] = Stock_Data['Close'].ewm(span=short_window, adjust=True).mean()
        Stock_Data['LMA'] = Stock_Data['Close'].ewm(span=long_window, adjust=True).mean()
    
    Stock_Data['MACD'] = Stock_Data['SMA'] - Stock_Data['LMA']
    Stock_Data['Signal'] = Stock_Data['MACD'].ewm(span=signal_window, adjust=True).mean()
    return Stock_Data

#Function to generate signals  
def generate_signals(Stock_Data):
    Stock_Data['Signal_Indicator'] = np.where(Stock_Data['Signal'] < Stock_Data['MACD'], 1, 0)
    Stock_Data['Trade_Signal'] = Stock_Data['Signal_Indicator'].diff()
    buy_signals = Stock_Data[Stock_Data['Trade_Signal'] == 1].index
    sell_signals = Stock_Data[Stock_Data['Trade_Signal'] == -1].index
    return buy_signals, sell_signals

#Function to simulate trading
def simulate_trading(Stock_Data, buy_signals, sell_signals, initial_capital=100000, commission=0.00125):
    capital = initial_capital
    total_trades = 1
    returns = []
    
    for buy_date, sell_date in zip(buy_signals, sell_signals):
        if buy_date in Stock_Data.index and sell_date in Stock_Data.index:
            buy_price = Stock_Data.loc[buy_date, 'Close']
            sell_price = Stock_Data.loc[sell_date, 'Close']
            buy_c=Stock_Data.loc[buy_date, 'Close']+Stock_Data.loc[buy_date, 'Close']*commission
            sell_c=Stock_Data.loc[sell_date, 'Close']-Stock_Data.loc[sell_date, 'Close']*commission
            shares = capital/buy_c
            commission_paid_buy = shares * buy_price * commission
            trade_return = (sell_price - buy_price) / buy_price * 100
            commission_paid_sell = shares * sell_c * commission
            capital = shares * sell_c 
            returns.append(trade_return)
            total_trades += 1
    
    avg_return_per_trade = np.mean(returns) if returns else 0
    return capital, total_trades, avg_return_per_trade

#Function to simulate trading using buy hold sell strategy 
def buy_hold_sell_strategy(Stock_Data, initial_capital=100000, commission=0.00125):
    buy_price = Stock_Data['Close'].iloc[0]
    buy_price_with_commission = buy_price * (1 + commission)
    sell_price = Stock_Data['Close'].iloc[-1]
    sell_price_with_commission = sell_price * (1 - commission)
    shares = initial_capital / buy_price_with_commission
    final_capital = shares * sell_price_with_commission
    buy_commission = shares * (buy_price * commission)
    sell_commission = shares * (sell_price * commission)
    total_commission = buy_commission + sell_commission
    return final_capital

#Function to run trading model
def run_trading_model(file_path):
    choice = input("Choose moving average type (SMA/EMA): ").strip().upper()
    if choice=='SMA':
        use_sma=True
    else:
        use_sma=False

    # Loading the stock data
    Stock_Data = pd.read_excel(file_path)[['Date', 'Close']]
    Stock_Data['Date'] = pd.to_datetime(Stock_Data['Date'])
    Stock_Data.set_index('Date', inplace=True)
    
    # Computing the MACD
    Stock_Data = compute_macd(Stock_Data,use_sma)
    
    # Generating the trade signals
    buy_signals, sell_signals = generate_signals(Stock_Data)
    if len(buy_signals) == 0 or len(sell_signals) == 0:
        print("No trading signals generated.")
        return
    
    #Trading 
    macd_final_capital, total_trades, avg_return_per_trade = simulate_trading(Stock_Data, buy_signals, sell_signals)
    buy_hold_final_capital = buy_hold_sell_strategy(Stock_Data)
    relative_gain_loss = ((macd_final_capital / buy_hold_final_capital) - 1) * 100
        
    
    # Print Summary of the trading
    print()
    print("\033[1m---Trading-Summary---\033[0m")
    print(f"Initial Capital: \033[1m$100,000.00\033[0m")
    print(f"MACD Trading Final Capital: \033[1m${macd_final_capital:.2f}\033[0m")
    print(f"Buy-Hold-Sell Final Capital: \033[1m${buy_hold_final_capital:.2f}\033[0m")
    print(f"Total Trades Executed: {total_trades}")
    print("Number of buy signals: ",len(buy_signals))
    print("Number of sell signals: ",len(sell_signals))
    print(f"Average Return per Trade: {avg_return_per_trade:.2f}%")
    print(f"Relative Gain/Loss vs Buy-Hold: {relative_gain_loss:.2f}%")

    #Printing macd graph and histogram 
    plt.figure(figsize=(12, 5))
    plt.plot(Stock_Data.index, Stock_Data['MACD'], label='MACD', color='blue', linewidth=2)
    plt.plot(Stock_Data.index, Stock_Data['Signal'], label='Signal Line', color='orange', linewidth=2)
    macd_hist = Stock_Data['MACD'] - Stock_Data['Signal']
    plt.bar(Stock_Data.index, macd_hist, color=np.where(macd_hist > 0, 'green', 'red'), width=1.5, alpha=0.5, label='MACD Histogram')
    plt.scatter(buy_signals[::5], Stock_Data.loc[buy_signals[::5], 'MACD'], marker='^', color='green', label='Buy Signal', s=100, zorder=5)
    plt.scatter(sell_signals[::5], Stock_Data.loc[sell_signals[::5], 'MACD'], marker='v', color='red', label='Sell Signal', s=100, zorder=5)
    
    plt.title(f"MACD Trading Strategy: {choice} (Short-Term MA)", fontsize=18, fontweight='bold')
    plt.xlabel("Date", fontsize=14)
    plt.ylabel("MACD Value", fontsize=14)
    plt.legend(loc='best', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

# Main Program
print("Welcome to the MACD Trading Model!")
while True:
    file_path = input("\nEnter the path to the Excel file (or type 'Exit' to leave): ")
    if file_path.lower() == 'exit': 
        print("Exiting the program!")
        break
    try:
        run_trading_model(file_path)
    except FileNotFoundError:
        print("Error: File not found. Please enter a valid file path.")
