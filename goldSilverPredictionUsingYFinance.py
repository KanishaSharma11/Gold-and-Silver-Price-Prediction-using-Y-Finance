# Required Libraries
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM

# Function to download and prepare data
def load_data(ticker, start='2010-01-01', end='2024-12-31'):
    df = yf.download(ticker, start=start, end=end)
    df = df[['Close']]
    df.dropna(inplace=True)
    return df

# Function to scale and prepare dataset for LSTM
def preprocess_data(df, time_step=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)

    X, y = [], []
    for i in range(time_step, len(scaled_data)):
        X.append(scaled_data[i-time_step:i, 0])
        y.append(scaled_data[i, 0])

    X = np.array(X)
    y = np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    return X, y, scaler

# Function to create LSTM model
def build_model(X):
    input_shape = (X.shape[1], X.shape[2])  # (timesteps, features)
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Function to predict and plot results
def predict_and_plot(model, df, scaler, time_step=60, label='Asset'):
    full_data = scaler.transform(df)
    X_test = []
    for i in range(time_step, len(full_data)):
        X_test.append(full_data[i-time_step:i, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    predicted = model.predict(X_test)
    predicted = scaler.inverse_transform(predicted)

    actual = df[time_step:].values

    # Return values for RMSE
    return actual, predicted

# GOLD Prediction
print("Processing GOLD...")
gold_df = load_data('GC=F')  # Gold Futures
X_gold, y_gold, scaler_gold = preprocess_data(gold_df)
model_gold = build_model(X_gold)
model_gold.fit(X_gold, y_gold, epochs=10, batch_size=64, verbose=1)
predict_and_plot(model_gold, gold_df, scaler_gold, label='Gold')

# SILVER Prediction
print("Processing SILVER...")
silver_df = load_data('SI=F')  # Silver Futures
X_silver, y_silver, scaler_silver = preprocess_data(silver_df)
model_silver = build_model(X_silver)
model_silver.fit(X_silver, y_silver, epochs=10, batch_size=64, verbose=1)
predict_and_plot(model_silver, silver_df, scaler_silver, label='Silver')

from sklearn.metrics import mean_squared_error
import math

actual, predicted = predict_and_plot(model_gold, gold_df, scaler_gold, label='Gold')
rmse = math.sqrt(mean_squared_error(actual, predicted))
print(f"Gold RMSE: {rmse:.2f}")

actual, predicted = predict_and_plot(model_silver, silver_df, scaler_silver, label='Silver')
rmse = math.sqrt(mean_squared_error(actual, predicted))
print(f"Silver RMSE: {rmse:.2f}")

#Forecast Future Prices
def forecast_future(model, df, scaler, days=30, time_step=60):
    input_data = scaler.transform(df)[-time_step:]
    predictions = []

    for _ in range(days):
        input_seq = input_data[-time_step:]
        input_seq = input_seq.reshape((1, time_step, 1))
        pred = model.predict(input_seq)[0][0]
        predictions.append(pred)
        input_data = np.append(input_data, [[pred]], axis=0)

    predicted_prices = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    return predicted_prices

future_prices = forecast_future(model_gold, gold_df, scaler_gold)
