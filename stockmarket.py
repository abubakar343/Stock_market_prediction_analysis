#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Python Libraries


# In[89]:


import pandas as pd
import numpy as np


# In[ ]:


#Dataset


# In[90]:


df=pd.read_csv('/content/sample_data/XAUUSD60.csv')


# In[ ]:


#Modify Columns


# In[91]:



df.rename(columns={"2020.01.02": "Date"}, inplace=True)
df.columns

df.rename(columns={"06:00": "Time"}, inplace=True)
df.columns

df.rename(columns={"1520.26": "Open"}, inplace=True)
df.columns

df.rename(columns={"1520.36": "High"}, inplace=True)
df.columns


df.rename(columns={"1519.39": "Low"}, inplace=True)
df.columns


df.rename(columns={"1519.39.1": "Close"}, inplace=True)
df.columns


df.rename(columns={"756": "Volume"}, inplace=True)
df.columns


# In[92]:


df.head(30)


# In[ ]:


#Feature selection from Dataset


# In[93]:


X = df.iloc[:,2:5]
Y = df.iloc[:,5]


# In[94]:


Y


# In[96]:


from sklearn import preprocessing


# In[ ]:


#Preprocessing Data to train and test Model


# In[95]:


min_max_scaler = preprocessing.MinMaxScaler()
X_scale = min_max_scaler.fit_transform(X)


# In[97]:


X_scale


# In[ ]:


# Data Splitting to train and test model


# In[98]:


from sklearn.model_selection import train_test_split


# In[ ]:


#LSTM Model for neural network algorithm


# In[99]:


import time
import pandas_datareader as pdr

import keras
from keras.layers import LSTM
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout

from sklearn.preprocessing import MinMaxScaler

import seaborn as sns
import matplotlib.pyplot as plt


# In[100]:


df["Date"] = pd.to_datetime(df["Date"])
df.head()


# In[101]:


df.set_index("Date",inplace=True)
df.head()


# In[102]:


df["Volume"].min(), df["Volume"].max()


# In[103]:


df[["High", "Low", "Open","Close"]].plot(figsize=(10,7))
plt.legend(loc="best")
plt.xlabel("Date")
plt.ylabel("Price")
plt.title("Predicting stock prices with date and range of Price")


# In[104]:


new_df = pd.DataFrame(df["Close"].copy(), columns=["Close"])
new_df.head()


# In[105]:


train_size = int(len(new_df)*0.8)

train = new_df.iloc[0:train_size]
test = new_df.iloc[train_size:len(new_df)]
test.head(15)


# In[106]:


len(train), len(test)


# In[107]:


train.head(10)


# In[108]:


def create_dataset(X, y, lag=1):
    xs,ys = [], []
    
    for i in range(len(X) - lag):
        tmp = X.iloc[i: i+lag].values
        xs.append(tmp)
        ys.append(y.iloc[i+lag])
    
    return np.array(xs), np.array(ys)


# In[109]:


xtrain, ytrain = create_dataset(train, train["Close"],10)
xtest, ytest = create_dataset(test, test["Close"],10)


# In[ ]:


#Sequential Model for Neural Network


# In[110]:


model = Sequential()
model.add(LSTM(50,activation='relu', input_shape=(xtrain.shape[1], xtrain.shape[2])))
model.add(Dense(25))
model.add(Dense(1))
model.compile(loss="mean_squared_error", optimizer="adam")


# In[111]:


model.fit(xtrain, ytrain,
         epochs=10,
         batch_size=10,
         verbose=1,
         shuffle=False
         )


# In[112]:


ypred = model.predict(xtest)


# In[113]:


ytest


# In[114]:


plt.figure(figsize=(12,7))
plt.plot(np.arange(0, len(xtrain)), ytrain, 'g', label="history")
plt.plot(np.arange(len(xtrain), len(xtrain) + len(xtest)), ypred, 'b', label="predictions")
plt.plot(np.arange(len(xtrain), len(xtrain) + len(xtest)), ytest, 'b', label="Close")
plt.xlabel("Predictions")
plt.ylabel("Price")
plt.title("Prices prediction with High ")


# In[115]:


import yfinance as yf
import pyfolio as pf
import datetime as dt
import pandas_datareader.data as web
import os
import warnings

# print all outputs
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# In[ ]:


get_ipython().system('pip install yfinance')


# In[116]:


get_ipython().system('pip install pyfolio')


# In[ ]:


#Backtest result for Stock Market


# In[117]:


df['Buy and Hold Returns'] = np.log(df['Close']/df['Close'].shift(1))
df.head(3)


# In[ ]:


#Strategy Based Columns
#Working with Close Price Column


# In[118]:


df['mean'] = df['Close'].rolling(window=20).mean()
df['standarddeviation'] = df['Close'].rolling(window=20).std()
df['upper'] = df['mean'] + (2 * df['standarddeviation'])
df['lower'] = df['mean'] - (2 * df['standarddeviation'])
df.drop(['Open','High','Low'],axis=1,inplace=True,errors='ignore')
df.tail(5)


# In[ ]:


#Buying or Selling Stock 


# In[119]:


df['stock prediction'] = np.where((df['Close'] < df['lower']) &
                        (df['Close'].shift(1) >=       df['lower']),1,0)

# SELL condition
df['stock prediction'] = np.where( (df['Close'] > df['upper']) &
                          (df['Close'].shift(1) <= df['upper']),-1,df['stock prediction'])
# creating long and short positions 
df['sl position'] = df['stock prediction'].replace(to_replace=0, method='ffill')

# shifting by 1, to account of close price return calculations
df['sl position'] = df['sl position'].shift(1)

# calculating stretegy returns
df['strategy_returns'] = df['Buy and Hold Returns'] * (df['sl position'])

df.tail(5)


# In[120]:


df["stock prediction"].min(), df["stock prediction"].max()


# In[121]:


print("Buy and hold returns:",df['Buy and Hold Returns'].cumsum()[-1])
print("Strategy returns:",df['strategy_returns'].cumsum()[-1])

# plotting strategy historical performance over time
df[['Buy and Hold Return','strategy_returns']] = df[['Buy and Hold Returns','strategy_returns']].cumsum()
df[['Buy and Hold Return','strategy_returns']].plot(grid=True, figsize=(12, 8))


# In[122]:


pf.create_simple_tear_sheet(df['strategy_returns'].diff())

