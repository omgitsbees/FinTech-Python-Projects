import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout

# Fetch historical stock price data
stock = yf.download('AAPL', start='2015-01-01', end='2023-12-31')

# Adding more technical indicators
def get_technical_indicators(data):
    # Moving Average (MA50)
    data['MA50'] = data['Close'].rolling(window=50).mean()

    # Daily Returns
    data['Return'] = data['Close'].pct_change() * 100

    # Relative Strength Index (RSI)
    delta = data['Close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))

    # Moving Average Convergence Divergence (MACD)
    data['EMA12'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['EMA26'] = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = data['EMA12'] - data['EMA26']
    
    # Bollinger Bands
    data['20SMA'] = data['Close'].rolling(window=20).mean()
    data['STD20'] = data['Close'].rolling(window=20).std()
    data['Upper_BB'] = data['20SMA'] + (2 * data['STD20'])
    data['Lower_BB'] = data['20SMA'] - (2 * data['STD20'])
    
    return data

# Apply the technical indicators function
stock = get_technical_indicators(stock)

# Drop missing values
stock = stock.dropna()

# Feature Scaling (Normalization) for LSTM
scaler = MinMaxScaler(feature_range=(0, 1))

# Ensure MA50, Return, RSI, MACD, Upper_BB, Lower_BB are included in scaling
scaled_data = scaler.fit_transform(stock[['Close', 'MA50', 'Return', 'RSI', 'MACD', 'Upper_BB', 'Lower_BB']])

# Prepare training data for LSTM
X = []
y = []
sequence_length = 90  # Extended sequence length to capture longer trends

for i in range(sequence_length, len(scaled_data)):
    X.append(scaled_data[i-sequence_length:i])
    y.append(scaled_data[i, 0])  # Predicting the Close price

X = np.array(X)
y = np.array(y)

# Split data into training and testing sets (80% training, 20% testing)
split_ratio = 0.8
split = int(split_ratio * len(X))

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Reshape data for LSTM/GRU (samples, time steps, features)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2]))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_test.shape[2]))

# Build the hybrid LSTM/GRU model
model = Sequential()

# LSTM layer
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))

# GRU layer
model.add(GRU(units=50, return_sequences=False))
model.add(Dropout(0.2))

# Output layer (1 unit for the predicted price)
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model with more epochs (50)
model.fit(X_train, y_train, batch_size=32, epochs=50)

# Predict stock prices
y_pred = model.predict(X_test)

# Rescale the predictions and actual values back to the original scale
y_pred_rescaled = scaler.inverse_transform(np.concatenate([y_pred, np.zeros((y_pred.shape[0], 6))], axis=1))[:, 0]
y_test_rescaled = scaler.inverse_transform(np.concatenate([y_test.reshape(-1, 1), np.zeros((y_test.shape[0], 6))], axis=1))[:, 0]

# Model evaluation
mse = mean_squared_error(y_test_rescaled, y_pred_rescaled)
print(f'Mean Squared Error (LSTM + GRU): {mse}')

# Plotting actual vs predicted stock prices
plt.figure(figsize=(10, 6))
plt.plot(stock.index[split+sequence_length:], y_test_rescaled, label="Actual Price")
plt.plot(stock.index[split+sequence_length:], y_pred_rescaled, label="Predicted Price")
plt.title("Actual vs Predicted Stock Prices (LSTM + GRU)")
plt.legend()
plt.show()
