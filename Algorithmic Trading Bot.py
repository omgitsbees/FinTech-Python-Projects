import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Simulate price data for testing
def generate_simulated_data(num_points=100):
    """Generate simulated cryptocurrency price data."""
    np.random.seed(42)
    price = np.random.normal(loc=100, scale=2, size=num_points)  # Simulated price around $100 with some volatility
    time_index = pd.date_range(start='2023-01-01', periods=num_points, freq='T')
    
    df = pd.DataFrame({'time': time_index, 'close': price})
    return df

# Apply a simple Moving Average Crossover Strategy
def apply_moving_average_strategy(data):
    """Apply a Moving Average Crossover strategy."""
    data['SMA20'] = data['close'].rolling(window=20).mean()
    data['SMA50'] = data['close'].rolling(window=50).mean()
    
    # Generate trading signals
    data['signal'] = 0
    data['signal'][20:] = np.where(data['SMA20'][20:] > data['SMA50'][20:], 1, 0)  # Buy when SMA20 > SMA50
    data['position'] = data['signal'].diff()  # Position changes: Buy (1) or Sell (-1)
    
    return data

# Simulate trading based on signals
def simulate_trading(data, initial_cash=10000, crypto_amount=0, trade_amount=100):
    """Simulate trading based on buy/sell signals."""
    cash = initial_cash
    crypto = crypto_amount
    
    for index, row in data.iterrows():
        if row['position'] == 1:  # Buy signal
            if cash >= trade_amount:
                crypto += trade_amount / row['close']
                cash -= trade_amount
                print(f"Buy: {trade_amount}$ worth of crypto at {row['close']}$")
        elif row['position'] == -1:  # Sell signal
            if crypto > 0:
                cash += crypto * row['close']
                print(f"Sell: All crypto sold at {row['close']}$")
                crypto = 0
                
    return cash, crypto

# Plot buy and sell signals
def plot_signals(data):
    """Plot the moving averages and buy/sell signals."""
    plt.figure(figsize=(12,6))
    plt.plot(data['time'], data['close'], label='Close Price')
    plt.plot(data['time'], data['SMA20'], label='SMA 20')
    plt.plot(data['time'], data['SMA50'], label='SMA 50')
    
    # Plot buy/sell signals
    plt.plot(data[data['position'] == 1]['time'], data['SMA20'][data['position'] == 1], '^', markersize=10, color='g', label='Buy Signal')
    plt.plot(data[data['position'] == -1]['time'], data['SMA20'][data['position'] == -1], 'v', markersize=10, color='r', label='Sell Signal')
    
    plt.title('Simulated Moving Average Crossover Signals')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

# Main function to run the simulated trading bot
def run_simulated_trading_bot():
    # Generate simulated data
    data = generate_simulated_data()
    
    # Apply the Moving Average strategy
    data = apply_moving_average_strategy(data)
    
    # Simulate trading
    final_cash, final_crypto = simulate_trading(data)
    
    # Display the results
    print(f"Final Cash: {final_cash}$, Final Crypto Holdings: {final_crypto} units")
    
    # Plot the buy/sell signals
    plot_signals(data)

if __name__ == "__main__":
    run_simulated_trading_bot()
