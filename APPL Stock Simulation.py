import yfinance as yf
import schedule
import time
import plotly.graph_objects as go

# Set the initial amount of money
initial_amount = 10000

# Set the stock symbol
stock_symbol = 'AAPL'

# Set the buy/sell thresholds
buy_threshold = 0.05  # 5% drop
sell_threshold = 0.10  # 10% increase

# Create a function to fetch the current stock price
def get_current_price():
    return yf.Ticker(stock_symbol).info['currentPrice']

# Create a function to make buy/sell decisions
def make_decision(current_price, previous_price):
    if current_price < previous_price * (1 - buy_threshold):
        # Buy
        print('Buying at', current_price)
        return 'buy'
    elif current_price > previous_price * (1 + sell_threshold):
        # Sell
        print('Selling at', current_price)
        return 'sell'
    else:
        # Do nothing
        return 'hold'

# Create a function to update the simulation
class StockSimulation:
    def __init__(self):
        self.money = initial_amount
        self.shares = 0
        self.previous_price = None

    def update_simulation(self):
        current_price = get_current_price()
        if self.previous_price is None:
            self.previous_price = current_price
        else:
            decision = make_decision(current_price, self.previous_price)
            if decision == 'buy':
                if self.money >= current_price:
                    self.money -= current_price
                    self.shares += 1
                    print(f"Bought 1 share at {current_price}. Remaining money: {self.money}")
            elif decision == 'sell':
                if self.shares > 0:
                    self.money += current_price
                    self.shares -= 1
                    print(f"Sold 1 share at {current_price}. Remaining money: {self.money}")
            self.previous_price = current_price
        print('Current price:', current_price)
        print('Amount of money:', self.money)

# Create an instance of the StockSimulation class
simulation = StockSimulation()

# Schedule the simulation to run every minute
def run_simulation():
    simulation.update_simulation()

schedule.every(1).minutes.do(run_simulation)

# Run the simulation
while True:
    schedule.run_pending()
    time.sleep(1)