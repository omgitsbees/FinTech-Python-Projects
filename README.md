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

-------------------------------------------------------------------------------------------------------------------

Credit Scoring System using Logistic Regression

This project simulates a credit scoring system, where a logistic regression model is trained to predict whether a customer will default on a loan based on several features such as income, age, loan amount, and credit history. The project demonstrates how to simulate data, train a logistic regression model, and evaluate its performance using metrics like accuracy, precision, recall, F1-score, and ROC-AUC.

Table of Contents

	•	Overview
	•	Features
	•	Data Simulation
	•	Model Training
	•	Evaluation
	•	Installation
	•	Usage
	•	Contributing
	•	License

Overview

Credit scoring is the process of determining the creditworthiness of a borrower by assessing various factors such as income, age, credit history, and loan amount. Logistic regression is a commonly used algorithm for binary classification problems like this one, where the goal is to predict whether or not a customer will default on a loan.

This project simulates customer data, trains a logistic regression model, and evaluates its performance using various evaluation metrics.

Features

	•	Data Simulation: Synthetic data is generated for income, age, loan_amount, and credit_history.
	•	Logistic Regression: A logistic regression model is trained on the simulated data to predict loan default.
	•	Model Evaluation:
	•	Confusion Matrix: Provides a detailed view of True Positives, False Positives, True Negatives, and False Negatives.
	•	Precision, Recall, and F1-score: To understand the trade-offs between different evaluation metrics.
	•	ROC-AUC: Measures the model’s ability to distinguish between classes.
	•	ROC Curve: A visual representation of the True Positive Rate vs. False Positive Rate.

Data Simulation

The dataset is simulated using the following features:

	•	Income: Normally distributed around 50,000 with a standard deviation of 15,000.
	•	Age: Normally distributed around 40 years with a standard deviation of 10 years.
	•	Loan Amount: Normally distributed around 15,000 with a standard deviation of 5,000.
	•	Credit History: Random integers from 0 to 9.

The binary target variable default is generated based on a logistic function applied to these features.

Model Training

The logistic regression model is trained on a subset of the simulated data. The features are standardized, and the data is split into training and testing sets. The model is trained to predict whether a customer will default on their loan (default = 1) or not (default = 0).

Evaluation

After training, the model’s performance is evaluated using:

	•	Accuracy: Measures the overall correctness of the model.
	•	Confusion Matrix: Shows the distribution of predictions in terms of True Positives, False Positives, True Negatives, and False Negatives.
	•	Classification Report: Provides precision, recall, and F1-score for each class (default, no default).
	•	ROC-AUC: The area under the ROC curve, which summarizes the model’s ability to distinguish between the two classes.
	•	ROC Curve: A plot of the True Positive Rate (TPR) against the False Positive Rate (FPR).

<img width="646" alt="Screenshot 2024-09-21 at 1 13 01 PM" src="https://github.com/user-attachments/assets/26341bc1-19c7-4603-a9c7-381e0751c5ff">

<img width="645" alt="Screenshot 2024-09-21 at 1 13 20 PM" src="https://github.com/user-attachments/assets/cde28209-84af-4d86-b0b8-f4bb54d2a679">

-------------------------------------------------------------------------------------------------------------------

Dynamic Pricing Model
A machine learning-based dynamic pricing model for banking and finance applications.

Overview

This project implements a dynamic pricing model that uses machine learning algorithms to predict optimal interest rates for loans based on various factors such as customer segment, loan type, collateral type, and market conditions.

Features

Uses machine learning algorithms (Random Forest, Neural Networks, and ARIMA) to predict optimal interest rates

Incorporates regulatory requirements (Basel III, Dodd-Frank Act)

Uses banking-specific data sources (Federal Reserve Economic Data, Bank of International Settlements)

Includes model interpretability techniques (feature importance, partial dependence plots)

Requirements
Python 3.x
scikit-learn
pandas
numpy
yfinance
requests


Usage
Clone the repository: git clone https://github.com/your-username/Dynamic-Pricing-Model.git
Install the required packages: pip install -r requirements.txt
Run the model: python dynamic_pricing_model.py

Example Use Cases
Predicting optimal interest rates for loans based on customer segment, loan type, and collateral type
Analyzing the impact of regulatory requirements on interest rates
Visualizing the relationships between interest rates and market conditions

-------------------------------------------------------------------------------------------------------------------

# Budgeting App

A simple budgeting app built with Python and Tkinter.

## Features

* User registration and login functionality
* Ability to add transactions, budgets, and investments
* View budget and investment data

## Requirements

* Python 3.x
* Tkinter
* SQLite3

## Installation

1. Clone the repository: `git clone https://github.com/your-username/budgeting-app.git`
2. Navigate to the project directory: `cd budgeting-app`
3. Run the app: `python budgeting_app.py`

## Usage

1. Register a new user or login with existing credentials
2. Add transactions, budgets, and investments
3. View budget and investment data

## Contributing

Contributions are welcome! Please submit a pull request with your changes.

## License

This project is licensed under the MIT License. See LICENSE for details.

## Acknowledgments

* Tkinter for the GUI framework
* SQLite3 for the database

-------------------------------------------------------------------------------------------------------------------

Apple Stock Candlestick Chart

A Python script that uses the mplfinance library to create a candlestick chart of Apple's stock price history.

Features

Loads Apple's stock price history from a CSV file

Creates a candlestick chart using the mplfinance library

Displays the chart in a Tkinter window

Requirements

Python 3.x

pandas library

mplfinance library

tkinter library

Installation

Clone the repository: git clone https://github.com/your-username/apple-stock-candlestick-chart.git

Install the required libraries: pip install pandas mplfinance tkinter

Run the script: python apple_stock_candlestick_chart.py

Usage

Load the CSV file containing Apple's stock price history into the script.

Run the script to create the candlestick chart.

The chart will be displayed in a Tkinter window.

Contributing

Contributions are welcome! If you'd like to contribute to this project, please fork the repository and submit a pull request.

License

This project is licensed under the MIT License. See the LICENSE file for more information.

Acknowledgments

Thanks to the mplfinance library for providing an easy-to-use interface for creating candlestick charts.



Thanks to the tkinter library for providing a simple way to create GUI windows in Python.

![Screenshot 2024-10-03 142635](https://github.com/user-attachments/assets/8c388590-9f34-4f61-b9be-73811c0efda0)

-------------------------------------------------------------------------------------------------------------------

Loan Default Prediction

Overview

This project uses machine learning to predict whether a loan will default or not. The model is trained on a dataset of loan information and uses a logistic regression algorithm to make predictions.

Dataset

The dataset used for this project is a CSV file containing information on loans, including:

Loan ID

Age

Income

Loan Amount

Credit Score

Months Employed

Num Credit Lines

Interest Rate
Loan Term

DTI Ratio

Education

Employment Type

Marital Status

Has Mortgage

Has Dependents

Loan Purpose

Has Co-Signer

Default (target variable)

Model

The model used for this project is a logistic regression model, which is a type of supervised learning algorithm that is well-suited for binary classification problems like this one. The model is trained on the dataset using the scikit-learn library in Python.

Results

The model achieves an accuracy of 88.51% on the test set, with a precision of 0.89 and a recall of 1.00 for non-defaults, and a precision of 0.59 and a recall of 0.03 for defaults. The F1-score for non-defaults is 0.94, while the F1-score for defaults is 0.07.

Code

The code for this project is written in Python and uses the following libraries:

pandas for data manipulation and analysis

scikit-learn for machine learning

tkinter for the GUI

The code is organized into the following files:

loan_default_prediction.py: This file contains the code for the model and the GUI.

data.csv: This file contains the dataset used for the project.

Usage

To use this project, simply clone the repository and run the loan_default_prediction.py file. This will launch the GUI, where you can select a CSV file containing loan information and click the "Predict" button to make predictions.

Contributing

If you'd like to contribute to this project, please fork the repository and submit a pull request. I'd be happy to review your changes and merge them into the main branch.

License

This project is licensed under the MIT License. See the LICENSE file for details.

Acknowledgments

This project was built using the following resources:

scikit-learn documentation: https://scikit-learn.org/stable/
pandas documentation: https://pandas.pydata.org/docs/
tkinter documentation: https://docs.python.org/3/library/tk.html

![Screenshot 2024-10-03 145018](https://github.com/user-attachments/assets/b734aba1-52a9-49c3-8c6e-5f31f9a443ff)

-------------------------------------------------------------------------------------------------------------------

Stock Trading Simulation with yFinance and Schedule

This Python project simulates stock trading for a specified stock symbol using real-time price data fetched from Yahoo Finance. The simulation makes automated buy/sell decisions based on pre-defined price thresholds and runs periodically, allowing you to track and simulate the growth or decline of your investment over time.
Features

    Real-time stock price fetching: Uses yfinance to retrieve the current stock price.
    Automated buy/sell decisions: Buys when the price drops by a set percentage and sells when the price increases by another set percentage.
    Continuous simulation: The simulation runs every minute using the schedule library.
    Portfolio management: Keeps track of the amount of money and the number of shares owned.
    Plotting: Uses plotly.graph_objects for visualizing data (placeholder for future implementation).

Requirements

    Python 3.x
    Required libraries:
        yfinance
        schedule
        plotly

You can install the dependencies by running:

bash

pip install yfinance schedule plotly

How it Works

    Initial Setup: The simulation starts with a fixed amount of money (e.g., $10,000) and no shares of the stock.
    Stock Price Fetching: The script fetches the real-time stock price for the specified stock symbol (default: AAPL).
    Buy/Sell Logic:
        Buy when the price drops by 5% from the previous fetched price.
        Sell when the price increases by 10% from the previous fetched price.
        Hold if neither condition is met.
    Continuous Execution: The simulation checks the stock price every minute and updates your portfolio accordingly.

Code Overview

python

import yfinance as yf
import schedule
import time
import plotly.graph_objects as go

Parameters:

    initial_amount: The starting money for the simulation (default: $10,000).
    stock_symbol: The stock symbol to track (default: 'AAPL').
    buy_threshold: Percentage drop in price to trigger a buy (default: 5%).
    sell_threshold: Percentage increase in price to trigger a sell (default: 10%).

Key Functions:

    get_current_price(): Fetches the real-time stock price using yfinance.
    make_decision(current_price, previous_price): Decides whether to buy, sell, or hold based on current and previous stock prices.
    StockSimulation: The class that manages the portfolio and simulates stock trades.

Running the Simulation:

The simulation is scheduled to run every minute using the schedule library. It continuously updates based on the latest stock price.

python

while True:
    schedule.run_pending()
    time.sleep(1)

Running the Project

    Clone this repository or download the script.
    Install the required libraries using pip.
    Run the script:

bash

python stock_simulation.py

The script will start running, fetching the stock price every minute and making buy/sell decisions based on the defined thresholds.
Future Improvements

    Data visualization: Implement real-time stock price and portfolio value plotting using plotly.
    More advanced strategies: Introduce more complex trading strategies, such as using moving averages or other technical indicators.
    Multiple stocks: Expand the simulation to track and trade multiple stocks simultaneously.

License

This project is licensed under the MIT License.

-------------------------------------------------------------------------------------------------------------------

Stock Price Simulation Bot

This Python script simulates buying and selling shares of major tech companies based on stock price fluctuations. It uses the yfinance library to fetch real-time stock data, calculates percentage price changes, and performs buy/sell operations depending on certain thresholds.
Features

    Tracks stock prices of major companies (e.g., AAPL, META, GOOG, MSFT, NVDA, AMZN).
    Buys shares when the stock price decreases by more than 0.10%.
    Sells shares when the stock price increases by more than 0.10%.
    Simulates the trading environment by continuously checking prices every minute.

Prerequisites

Ensure you have the following installed:

    Python 3.x
    yfinance library

You can install the required yfinance library using pip:

bash

pip install yfinance

How It Works

    The script starts with an initial amount of $10,000 and buys one share of each listed company.
    It tracks the stock prices every minute.
    If a stock's price increases by more than 0.10%, the bot sells one share of that stock.
    If a stock's price decreases by more than 0.10%, the bot buys one additional share (provided there is enough cash).
    The simulation runs indefinitely, with updates printed to the console each minute.

Stock Symbols

The following stock symbols are being tracked:

    AAPL (Apple)
    META (Meta)
    GOOG (Google)
    MSFT (Microsoft)
    NVDA (NVIDIA)
    AMZN (Amazon)

Functions Overview

    get_current_price(symbol): Fetches the latest stock price for a given symbol.
    calculate_percentage_change(current_price, previous_price): Calculates the percentage change between the current and previous prices.
    buy_shares(symbol, current_price): Buys a share of the specified stock if conditions are met.
    sell_shares(symbol, current_price): Sells a share of the specified stock if conditions are met.
    run_simulation(): Runs the stock tracking and trading simulation, printing results to the console.

Running the Simulation

To run the script, simply execute it using Python:

bash

python stock_simulation.py

The script will fetch stock data and make buy/sell decisions every minute based on the stock price movements.
License

This project is open-source under the MIT License.
