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
