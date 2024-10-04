import yfinance as yf
import time

# Set the stock symbols
stock_symbols = ["AAPL", "META", "GOOG", "MSFT", "NVDA", "AMZN"]

# Set the initial amount of money
initial_amount = 10000

# Create a function to fetch the current stock price
def get_current_price(symbol):
    ticker = yf.Ticker(symbol)
    data = ticker.history(period="1d", interval="1m")
    return data.iloc[-1]['Close']

# Create a dictionary to store the stock units for each company
stock_units = {symbol: [{"purchase_price": get_current_price(symbol), "sold": False}] for symbol in stock_symbols}

# Subtract the initial stock prices from the initial amount
for symbol in stock_symbols:
    initial_amount -= stock_units[symbol][0]["purchase_price"]

# Create a dictionary to store the current prices and previous prices for each company
prices = {symbol: {"current": 0, "previous": 0} for symbol in stock_symbols}

# Create a function to calculate the percentage change
def calculate_percentage_change(current_price, previous_price):
    if previous_price == 0:
        return 0
    return ((current_price - previous_price) / previous_price) * 100

# Create a function to buy shares
def buy_shares(symbol, current_price):
    global initial_amount, stock_units
    if initial_amount > current_price and current_price < stock_units[symbol][-1]["purchase_price"]:
        initial_amount -= current_price
        stock_units[symbol].append({"purchase_price": current_price, "sold": False})
        print(f"Bought 1 share of {symbol} for ${current_price:.2f}")

# Create a function to sell shares
def sell_shares(symbol, current_price):
    global initial_amount, stock_units
    for unit in stock_units[symbol]:
        if not unit["sold"] and current_price > unit["purchase_price"]:
            initial_amount += current_price
            unit["sold"] = True
            print(f"Sold 1 share of {symbol} for ${current_price:.2f}")
            break

# Create a function to run the simulation
def run_simulation():
    global prices, stock_units, initial_amount
    print("Current Prices:")
    for symbol in stock_symbols:
        current_price = get_current_price(symbol)
        previous_price = prices[symbol]["current"]
        prices[symbol]["previous"] = previous_price
        prices[symbol]["current"] = current_price
        percentage_change = calculate_percentage_change(current_price, previous_price)
        print(f"{symbol}: ${current_price:.2f} ({percentage_change:.2f}%)")
        if percentage_change > 0.10:
            sell_shares(symbol, current_price)
            print(f"  Sold {symbol} due to {percentage_change:.2f}% increase")
        elif percentage_change < -0.10 and initial_amount > 0:
            buy_shares(symbol, current_price)
            print(f"  Bought {symbol} due to {percentage_change:.2f}% decrease")
    print(f"Stock Units: {stock_units}")
    print(f"Cash: ${initial_amount:.2f}")

# Run the simulation
while True:
    run_simulation()
    time.sleep(60)  # Sleep for 1 minute