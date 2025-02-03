### 5. Stock Market Prediction (LSTM)

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# Generate synthetic stock price data
data = np.sin(np.linspace(0, 100, 1000)) + np.random.normal(0, 0.1, 1000)
df = pd.DataFrame({'StockPrice': data})

# Preprocess data
scaler = MinMaxScaler()
df['StockPrice'] = scaler.fit_transform(df[['StockPrice']])

X, y = [], []
for i in range(len(df) - 10):
    X.append(df.iloc[i:i+10].values)
    y.append(df.iloc[i+10].values)
X, y = np.array(X), np.array(y)

# Define model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(10, 1)),
    LSTM(50),
    Dense(1)
])

# Compile and train model
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=10, batch_size=16)
