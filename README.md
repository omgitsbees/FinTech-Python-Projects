Stock Price Prediction Using Machine Learning

This project implements a machine learning model to predict future stock prices using technical indicators and a combination of Long Short-Term Memory (LSTM) and Gated Recurrent Units (GRU). The model is trained on historical stock data fetched using the yfinance library.
Project Overview

The goal of this project is to predict stock prices using various technical indicators such as Moving Averages, Bollinger Bands, Relative Strength Index (RSI), and Moving Average Convergence Divergence (MACD). The project also incorporates advanced machine learning techniques with LSTM and GRU to capture sequential patterns in stock price movements.
Key Features:

    Technical Indicators: Moving Average (MA), Bollinger Bands, RSI, and MACD.
    Model Architecture: Hybrid LSTM and GRU network for improved predictive accuracy.
    Data Source: Stock data fetched using the yfinance library.
    Visualization: Actual vs predicted stock prices plotted to compare the model's performance.

Getting Started
Prerequisites

To run this project, you need to have the following Python libraries installed:

bash

pip install yfinance numpy pandas matplotlib scikit-learn tensorflow

Installation

    Clone the repository:

    bash

git clone https://github.com/your-username/stock-price-prediction.git

Navigate to the project directory:

bash

cd stock-price-prediction

Run the Python script:

bash

    python stock_price_prediction.py

Data

The stock data is fetched from Yahoo Finance using the yfinance library. You can change the stock symbol and date range in the script to predict stock prices for different companies.

python

# Fetch historical stock price data
stock = yf.download('AAPL', start='2015-01-01', end='2023-12-31')

Technical Indicators Used

    MA50: 50-day moving average.
    RSI: Relative Strength Index to measure overbought/oversold conditions.
    MACD: Moving Average Convergence Divergence for trend-following.
    Bollinger Bands: Upper and lower bounds to gauge price volatility.

Model Architecture

The model uses a combination of LSTM and GRU layers to handle time-series data. The architecture is as follows:

    LSTM Layer: 50 units with dropout.
    GRU Layer: 50 units with dropout.
    Output Layer: Dense layer with 1 unit for the predicted stock price.

The model is trained using the Mean Squared Error (MSE) loss function and the Adam optimizer.
Model Performance

The performance of the model is evaluated by comparing the actual stock prices to the predicted stock prices using Mean Squared Error (MSE). A visualization is provided to compare the actual and predicted stock prices.

Example Output
The plot below shows the comparison between actual and predicted stock prices.
![Screenshot 2024-09-20 113327](https://github.com/user-attachments/assets/babc49e9-9a52-4886-8e3b-a5ddd30262c3)

-------------------------------------------------------------------------------------------------------------------

# Cryptocurrency Portfolio Tracker

A simple and interactive Python application for tracking your cryptocurrency portfolio. This tool allows users to monitor the real-time values of their cryptocurrency holdings, calculate total portfolio value, and track percentage changes over time.

## Features

- **Real-Time Price Updates**: Fetches current cryptocurrency prices using the CoinGecko API.
- **Portfolio Management**: Track multiple cryptocurrencies and their respective holdings.
- **Percentage Change Calculation**: Calculates and displays the percentage change in prices over time.
- **Data Persistence**: Saves portfolio and price data to a JSON file for easy access and tracking.
- **Graphical User Interface (GUI)**: Built with `tkinter` for a user-friendly experience.

## Requirements

- Python 3.x
- `requests` library
- `pandas` library
- `tkinter` (included with standard Python installations)

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/cryptocurrency-portfolio-tracker.git
   cd cryptocurrency-portfolio-tracker

    Install the required libraries:

    bash

pip install requests pandas

Run the application:

bash

    python cryptocurrency_portfolio_tracker.py

Usage

    Initial Setup: Modify the default_portfolio variable in the code to set your initial cryptocurrency holdings.
    Updating Portfolio: Click the "Update Portfolio" button to fetch the latest prices and update the displayed values.
    Data View: The application displays the cryptocurrency name, current price, old price, percentage change, amount held, and total value in a table format.
    Data Persistence: The portfolio data is saved to portfolio_data.json, which allows you to retain your settings across sessions.

API Used

    CoinGecko API

![Screenshot 2024-09-20 120129](https://github.com/user-attachments/assets/d26bad5f-ef85-478d-9568-2c569dee575a)

-------------------------------------------------------------------------------------------------------------------

Algorithmic Trading Bot (Simulated)

This project implements a simple algorithmic trading bot in Python using a Moving Average Crossover Strategy. The bot operates on simulated price data, generates buy/sell signals based on technical analysis, and simulates trading decisions. This project is a great starting point for understanding algorithmic trading and backtesting strategies.
Table of Contents

    Project Overview
    Features
    Requirements
    How It Works
    Setup and Installation
    Usage
    Example Output
    Next Steps
    Contributing
    License

Project Overview

The Algorithmic Trading Bot simulates trading a cryptocurrency using a moving average crossover strategy on randomly generated price data. It buys and sells based on crossover signals (buy when the short-term average crosses above the long-term average, sell when the opposite happens).
Moving Average Crossover Strategy

    SMA 20: Short-term simple moving average (20 periods).
    SMA 50: Long-term simple moving average (50 periods).
    Buy Signal: When the SMA 20 crosses above the SMA 50.
    Sell Signal: When the SMA 20 crosses below the SMA 50.

Features

    Simulated price data for testing and development.
    Simple moving average crossover trading strategy.
    Buy and sell signal generation based on moving averages.
    Simulated trading environment with initial cash balance.
    Detailed output of trading decisions (buy/sell) and final results.
    Visualize trading signals using Matplotlib.

Requirements

    Python 3.7+
    Libraries:
        pandas
        numpy
        matplotlib

Install the required libraries by running:

bash

pip install pandas numpy matplotlib

How It Works

    Simulated Data: We generate random price data to mimic the market's behavior.
    Moving Average Calculation: The bot calculates two moving averages:
        Short-term (20-period) SMA
        Long-term (50-period) SMA
    Buy/Sell Signals: The strategy generates buy signals when the short-term average crosses above the long-term average and sell signals when it crosses below.
    Simulated Trading: The bot "buys" and "sells" cryptocurrency using a predefined amount of cash, based on the generated signals.
    Visualization: Plots the price, moving averages, and buy/sell signals.

Setup and Installation

    Clone the repository:

bash

git clone https://github.com/your-username/algorithmic-trading-bot.git
cd algorithmic-trading-bot

    Install the required Python libraries:

bash

pip install -r requirements.txt

    Run the script:

bash

python algorithmic_trading_bot.py

Usage

Once the script is executed, it will generate the following:

    Console Output: Shows the buy/sell signals and trading transactions as they happen.
    Plot: Displays the simulated price data, moving averages, and buy/sell signals visually.

Example Usage

bash

python algorithmic_trading_bot.py

Example Console Output

ruby

Buy: 100$ worth of crypto at 98.74$
Sell: All crypto sold at 101.23$
Buy: 100$ worth of crypto at 97.61$
Sell: All crypto sold at 99.85$
Final Cash: 10050$, Final Crypto Holdings: 0 units

Example Plot

    Price Line: Simulated price over time.
    SMA 20 and SMA 50: Short-term and long-term moving averages.
    Buy/Sell Signals: Green arrows (buy), Red arrows (sell).

Example Output

    Initial Cash: The bot starts with $10,000 in cash.
    Trade Amount: Each buy action uses $100 to buy crypto.
    Trading Strategy: Sells all crypto when a sell signal occurs.

At the end of the simulation, the bot will display the final cash amount and remaining cryptocurrency holdings.
Next Steps

    Advanced Strategies: Implement more advanced trading strategies like RSI, MACD, or Bollinger Bands.
    Risk Management: Add stop-loss and take-profit features.
    Historical Data: Replace simulated data with historical cryptocurrency data for backtesting.
    Paper Trading: Connect the bot to a paper trading platform to test on live data without risking real money.

![Screenshot 2024-09-20 123512](https://github.com/user-attachments/assets/589dc2e5-b6f8-4fe4-9865-8616227c1f41)
