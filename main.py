#!/usr/bin/env python
# coding: utf-8

# install and import necessary libraries for data manipulation, modeling, and visualization.
get_ipython().system('pip install yfinance')  # Install yfinance library
get_ipython().system('pip install pyfolio')  # Install pyfolio library
import pandas as pd  # Data manipulation and analysis
import numpy as np  # Numerical operations
from sklearn import preprocessing  # Import preprocessing module from scikit-learn
from sklearn.model_selection import train_test_split  # Import train_test_split for splitting data
import time  # Import time module
import pandas_datareader as pdr  # Import pandas_datareader for financial data
import keras  # Import Keras for neural network modeling
from keras.layers import LSTM  # Import LSTM layer
from keras.models import Sequential  # Import Sequential model
from keras.layers.core import Dense, Activation, Dropout  # Import core layers
from sklearn.preprocessing import MinMaxScaler  # Import MinMaxScaler for data scaling
import seaborn as sns  # Import Seaborn for statistical data visualization
import matplotlib.pyplot as plt  # Import matplotlib for plotting
import yfinance as yf  # Import yfinance for fetching financial data
import pyfolio as pf  # Import pyfolio for performance analysis
import datetime as dt  # Import datetime for date operations
import pandas_datareader.data as web  # Import datareader for financial data
import os  # Import os for operating system operations
import warnings  # Import warnings for handling warnings


# Load dataset
df = pd.read_csv('/content/sample_data/XAUUSD60.csv')  # Load the dataset from a CSV file

# Modify column names for clarity
df.rename(columns={"2020.01.02": "Date"}, inplace=True)  # Rename the '2020.01.02' column to 'Date'
df.rename(columns={"06:00": "Time"}, inplace=True)  # Rename the '06:00' column to 'Time'
df.rename(columns={"1520.26": "Open"}, inplace=True)  # Rename the '1520.26' column to 'Open'
df.rename(columns={"1520.36": "High"}, inplace=True)  # Rename the '1520.36' column to 'High'
df.rename(columns={"1519.39": "Low"}, inplace=True)  # Rename the '1519.39' column to 'Low'
df.rename(columns={"1519.39.1": "Close"}, inplace=True)  # Rename the '1519.39.1' column to 'Close'
df.rename(columns={"756": "Volume"}, inplace=True)  # Rename the '756' column to 'Volume'

# Display the first 30 rows of the dataframe to verify changes
df.head(30)  

# Feature selection from Dataset
X = df.iloc[:, 2:5]  # Select feature columns (Open, High, Low) for model input
Y = df.iloc[:, 5]  # Select target column (Close) for model output

# Display the target variable
Y  




# Preprocessing Data to train and test Model
min_max_scaler = preprocessing.MinMaxScaler()  # Initialize Min-Max Scaler to normalize features
X_scale = min_max_scaler.fit_transform(X)  # Fit and transform the feature data

# Display scaled feature data
X_scale  

# Convert 'Date' column to datetime format and display the first few rows of the dataframe to verify changes
df["Date"] = pd.to_datetime(df["Date"])  
df.head()  

# Set 'Date' column as the index of the dataframe and display the first few rows to verify changes
df.set_index("Date", inplace=True)  
df.head()  

# Display the minimum and maximum values of 'Volume' column
df["Volume"].min(), df["Volume"].max()  

# Plot 'High', 'Low', 'Open', and 'Close' prices
df[["High", "Low", "Open", "Close"]].plot(figsize=(10,7))  
plt.legend(loc="best")  # Add legend to the plot
plt.xlabel("Date")  # Set x-axis label
plt.ylabel("Price")  # Set y-axis label
plt.title("Predicting stock prices with date and range of Price")  # Set plot title

# Create a new dataframe with only 'Close' prices
new_df = pd.DataFrame(df["Close"].copy(), columns=["Close"]) 
new_df.head()  # Display the first few rows


train_size = int(len(new_df) * 0.8)  # Define the training set size as 80% of the data

train = new_df.iloc[0:train_size]  # Split data into training set
test = new_df.iloc[train_size:len(new_df)]  # Split data into test set
test.head(15)  # Display the first 15 rows of the test set
len(train), len(test)  # Display the number of rows in the training and test sets
train.head(10)  # Display the first 10 rows of the training set

def create_dataset(X, y, lag=1):
    """
    Function to create dataset with lag features for time series prediction.
    
    Parameters:
    X (DataFrame): Input features.
    y (Series): Target variable.
    lag (int): Number of previous time steps to include as features.
    
    Returns:
    np.array: Features with lag.
    np.array: Target variable.
    """
    xs, ys = [], []
    for i in range(len(X) - lag):
        tmp = X.iloc[i: i + lag].values  # Extract features with lag
        xs.append(tmp)
        ys.append(y.iloc[i + lag])  # Extract target variable
    return np.array(xs), np.array(ys)  # Return features and target arrays


xtrain, ytrain = create_dataset(train, train["Close"], 10)  # Create training dataset with a lag of 10
xtest, ytest = create_dataset(test, test["Close"], 10)  # Create test dataset with a lag of 10

# Sequential Model for Neural Network
model = Sequential()  # Initialize Sequential model
model.add(LSTM(50, activation='relu', input_shape=(xtrain.shape[1], xtrain.shape[2])))  # Add LSTM layer
model.add(Dense(25))  # Add Dense layer
model.add(Dense(1))  # Add output layer
model.compile(loss="mean_squared_error", optimizer="adam")  # Compile the model with mean squared error loss and Adam optimizer
model.fit(xtrain, ytrain,  # Train the model
          epochs=10,  # Number of epochs
          batch_size=10,  # Batch size
          verbose=1,  # Print progress
          shuffle=False)  # Do not shuffle data

ypred = model.predict(xtest)  # Make predictions on the test set
ytest  # Display the actual target values for test set

plt.figure(figsize=(12,7))  # Create a new figure
plt.plot(np.arange(0, len(xtrain)), ytrain, 'g', label="history")  # Plot training data
plt.plot(np.arange(len(xtrain), len(xtrain) + len(xtest)), ypred, 'b', label="predictions")  # Plot predictions
plt.plot(np.arange(len(xtrain), len(xtrain) + len(xtest)), ytest, 'b', label="Close")  # Plot actual close prices
plt.xlabel("Predictions")  # Set x-axis label
plt.ylabel("Price")  # Set y-axis label
plt.title("Prices prediction with High ")  # Set plot title

# print all outputs
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"  # Set interactive shell to print all outputs

# Backtest result for Stock Market
df['Buy and Hold Returns'] = np.log(df['Close'] / df['Close'].shift(1))  # Calculate buy and hold returns
df.head(3)  # Display the first 3 rows

# Strategy-Based Columns
df['mean'] = df['Close'].rolling(window=20).mean()  # Calculate 20-day rolling mean
df['standarddeviation'] = df['Close'].rolling(window=20).std()  # Calculate 20-day rolling standard deviation
df['upper'] = df['mean'] + (2 * df['standarddeviation'])  # Calculate upper Bollinger Band
df['lower'] = df['mean'] - (2 * df['standarddeviation'])  # Calculate lower Bollinger Band
df.drop(['Open', 'High', 'Low'], axis=1, inplace=True, errors='ignore')  # Drop unnecessary columns
df.tail(5)  # Display the last 5 rows

# Buying or Selling Stock 
df['stock prediction'] = np.where((df['Close'] < df['lower']) &
                        (df['Close'].shift(1) >= df['lower']), 1, 0)  # Buy signal

# SELL condition
df['stock prediction'] = np.where((df['Close'] > df['upper']) &
                          (df['Close'].shift(1) <= df['upper']), -1, df['stock prediction'])  # Sell signal

# Creating long and short positions
df['sl position'] = df['stock prediction'].replace(to_replace=0, method='ffill')  # Forward fill positions
df['sl position'] = df['sl position'].shift(1)  # Shift positions by 1 day

# Calculating strategy returns
df['strategy_returns'] = df['Buy and Hold Returns'] * df['sl position']  # Calculate strategy returns

df.tail(5)  # Display the last 5 rows

df["stock prediction"].min(), df["stock prediction"].max()  # Display minimum and maximum values of stock predictions

print("Buy and hold returns:", df['Buy and Hold Returns'].cumsum()[-1])  # Print cumulative buy and hold returns
print("Strategy returns:", df['strategy_returns'].cumsum()[-1])  # Print cumulative strategy returns

# Plotting strategy historical performance over time
df[['Buy and Hold Return', 'strategy_returns']] = df[['Buy and Hold Returns', 'strategy_returns']].cumsum()  # Cumulative sum of returns
df[['Buy and Hold Return', 'strategy_returns']].plot(grid=True, figsize=(12, 8))  # Plot cumulative returns

# In[122]:
pf.create_simple_tear_sheet(df['strategy_returns'].diff())  # Create a simple tear sheet for strategy returns
