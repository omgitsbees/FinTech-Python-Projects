import requests
import pandas as pd
import tkinter as tk
from tkinter import messagebox, ttk
import time
import json
import os

# API URL for real-time prices (CoinGecko)
API_URL = 'https://api.coingecko.com/api/v3/simple/price'

# File to save portfolio_data
DATA_FILE = 'portfolio_data.json'

# Default portfolio (if no saved data exists)
default_portfolio = {
    'bitcoin': 0.5,
    'ethereum': 2.0,
    'litecoin': 10.0
}

# Function to get real-time crypto prices
def get_crypto_prices(cryptos, vs_currency='usd'):
    ids = ','.join(cryptos.keys())
    params = {
        'ids': ids,
        'vs_currencies': vs_currency
    }
    response = requests.get(API_URL, params=params)
    return response.json()

# Function to calculate portfolio value and percentage changes
def calculate_portfolio_value(prices, holdings, old_prices):
    total_value = 0
    detailed_portfolio = []

    for crypto, amount in holdings.items():
        current_price = prices[crypto]['usd']
        old_price = old_prices.get(crypto, current_price)
        percentage_change = ((current_price - old_price) / old_price) * 100 if old_price else 0

        value = current_price * amount
        total_value += value
        detailed_portfolio.append({
            'Cryptocurrency': crypto.capitalize(),
            'Current Price (USD)': current_price,
            'Old Price (USD)': old_price,
            'Percentage Change (%)': percentage_change,
            'Amount Held': amount,
            'Value (USD)': value
        })

    df = pd.DataFrame(detailed_portfolio)
    return df, total_value

# Function to save portfolio data
def save_portfolio_data(portfolio):
    with open(DATA_FILE, 'w') as file:
        json.dump(portfolio, file)

# Function to load saved portfolio data
def load_portfolio_data():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'r') as file:
            return json.load(file)
    return default_portfolio

# GUI setup
def create_gui():
    window = tk.Tk()
    window.title("Cryptocurrency Portfolio Tracker")
    
    # Corrected geometry method
    window.geometry("600x400")

    def update_portfolio():
        try:
            # Fetch real-time prices
            crypto_prices = get_crypto_prices(portfolio)
            portfolio_df, total_value = calculate_portfolio_value(crypto_prices, portfolio, old_prices)

            # Display in Treeview
            for row in tree.get_children():
                tree.delete(row)
            for index, row in portfolio_df.iterrows():
                tree.insert("", "end", values=(row['Cryptocurrency'], f"${row['Current Price (USD)']:.2f}", 
                                               f"${row['Old Price (USD)']:.2f}", 
                                               f"{row['Percentage Change (%)']:.2f}%", 
                                               row['Amount Held'], f"${row['Value (USD)']:.2f}"))

            # Update total value
            total_value_label.config(text=f"Total Portfolio Value: ${total_value:,.2f}")

            # Save current prices as old prices
            for crypto in portfolio.keys():
                old_prices[crypto] = crypto_prices[crypto]['usd']

            # Save portfolio and prices
            save_portfolio_data({'portfolio': portfolio, 'old_prices': old_prices})

        except Exception as e:
            messagebox.showerror("Error", str(e))

    # Treeview for displaying portfolio data
    columns = ('Cryptocurrency', 'Current Price', 'Old Price', 'Percentage Change', 'Amount Held', 'Value')
    tree = ttk.Treeview(window, columns=columns, show='headings')
    for col in columns:
        tree.heading(col, text=col)
    tree.pack(pady=20)

    # Label to display total value
    total_value_label = tk.Label(window, text="Total Portfolio Value: $0.00", font=("Arial", 14))
    total_value_label.pack(pady=10)

    # Button to update portfolio
    update_button = tk.Button(window, text="Update Portfolio", command=update_portfolio)
    update_button.pack(pady=10)

    # Run initial update
    update_portfolio()

    window.mainloop()

# Load portfolio and old prices from file (if available)
data = load_portfolio_data()
portfolio = data.get('portfolio', default_portfolio)
old_prices = data.get('old_prices', {})

#Create and run the GUI
create_gui()