import pandas as pd
import mplfinance as mpf
import tkinter as tk

# Load the CSV file
df = pd.read_csv('C:\\Users\\kyleh\\Downloads\\AAPL Historical Data.csv')

# Convert the date column to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Set the date column as the index
df.set_index('Date', inplace=True)

# Remove any non-numeric characters from the "Vol." column
df['Vol.'] = df['Vol.'].apply(lambda x: float(x.replace('M', '').replace('K', '')) * (1000000 if 'M' in x else 1000))

# Rename the columns to match what mpf.plot is expecting
df = df.rename(columns={'Price': 'Close', 'Vol.': 'Volume'})

# Create the main window
root = tk.Tk()
root.title("Apple Stock Candlestick Chart")

# Create the figure and axis
mpf.plot(df, type='candle', title='AAPL Candlestick Chart', ylabel='Price (USD)', ylabel_lower='', volume=True, style='yahoo')

# Show the window
root.mainloop()