import yfinance as yf
import pandas as pd
import numpy as np
from pytickersymbols import PyTickerSymbols
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
import datetime

start_date = "2015-01-01"
end_date = pd.Timestamp.today().strftime('%Y-%m-%d')

def downloadTicker(ticker):
    data = yf.download(ticker, start=start_date, end=end_date)
    datasheet = pd.DataFrame(data)

    if datasheet.index.name == "Date":
        datasheet = datasheet.reset_index()
    datasheet = datasheet[['Date', 'Close', 'Volume']]
    datasheet.columns = datasheet.columns.droplevel('Ticker')
    datasheet.columns.name = None
    return datasheet

def stock_choice(ticker_name, best_date):
    ticker_data = downloadTicker(ticker_name)  # This fetches the actual stock data

    # Now apply the filter on the stock data
    stock_data = ticker_data[ticker_data['Date'] > best_date].copy()
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])
    stock_data = stock_data.set_index("Date")

    # Moving Averages (Short-term & Long-term trends)
    stock_data["MA_5"] = stock_data["Close"].rolling(window=5).mean()  # Short-term
    stock_data["MA_20"] = stock_data["Close"].rolling(window=20).mean()  # Long-term
    stock_data["EMA_10"] = stock_data["Close"].ewm(span=10, adjust=False).mean()

    # Volatility: Rolling standard deviation of returns
    stock_data["Volatility"] = stock_data["Close"].pct_change().rolling(window=10).std()

    # Momentum Indicator: Percentage change
    stock_data["Pct_Change"] = stock_data["Close"].pct_change()

    # Relative Strength Index (RSI) - Measures momentum
    window_length = 14
    delta = stock_data[f"Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window_length).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window_length).mean()
    rs = gain / loss
    stock_data["RSI"] = 100 - (100 / (1 + rs))

    # Bollinger Bands (Upper and Lower bands)
    rolling_mean = stock_data["Close"].rolling(window=20).mean()
    rolling_std = stock_data["Close"].rolling(window=20).std()
    stock_data["BB_Upper"] = rolling_mean + (rolling_std * 2)
    stock_data["BB_Lower"] = rolling_mean - (rolling_std * 2)

    # On-Balance Volume (OBV) - Measures buying/selling pressure
    stock_data["OBV"] = (np.sign(stock_data["Close"].diff()) * stock_data["Volume"]).fillna(0).cumsum()

    # Lag Features
    stock_data["Close_Lag1"] = stock_data["Close"].shift(1)
    stock_data["Close_Lag2"] = stock_data["Close"].shift(2)
    stock_data["Volume_Lag1"] = stock_data["Volume"].shift(1)
    stock_data = stock_data.dropna()  # Drop NaN rows created by lagging

    X = stock_data[[
        "Close_Lag1", "Close_Lag2", "Volume_Lag1",
        "MA_5", "MA_20", "EMA_10", "Volatility", "Pct_Change", "RSI",
        "BB_Upper", "BB_Lower", "OBV"
    ]]
    y = stock_data["Close"]

    return X, y

#Program that checks all possible combinations of time frames for the best one that predicts prices
def best_date(ticker_name):
    lowest_rmse = float("inf")  # Initialize to infinity for finding the lowest RMSE
    best_start_date = None
    
    today = pd.to_datetime("today")
    for years in np.arange(3, 10.5, 0.5):  # Test different time windows
        start_date = (today - pd.DateOffset(months=int(years * 12))).strftime('%Y-%m-%d')
        X, y = stock_choice(ticker_name, start_date)  # Get stock data
        
        if len(X) < 100:  # Ensure enough data points
            continue

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100, learning_rate=0.1)
        model.fit(X_train, y_train)  # Fit model

        # Predict on test set
        y_pred = model.predict(X_test)

        # Calculate RMSE
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        if rmse < lowest_rmse:  # Check if this RMSE is the lowest
            lowest_rmse = rmse
            best_start_date = start_date

        # Stop if RMSE < 0.03
        if rmse < 0.03:
            break

    return best_start_date

def reshape_data(X, y):
    X_reshaped, y_reshaped = [], []
    for i in range(len(X) - 10):
        X_reshaped.append(X.iloc[i:i + 10].values)
        y_reshaped.append(y.iloc[i + 10])
    return np.array(X_reshaped), np.array(y_reshaped)

def predictTomorrow(ticker_name):
    X, y = stock_choice(ticker_name, best_date(ticker_name))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    #XGB Model
    xgb_model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=500,
        learning_rate=0.1,
        max_depth=5,
        subsample=0.9,
        colsample_bytree=0.9, 
        eval_metric="rmse"
    )

    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],  
        verbose=True
    )

    y_pred_xgb = xgb_model.predict(X_test)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train_tf, X_test_tf, y_train_tf, y_test_tf = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)

    # TensorFlow Dense Model
    tf_model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train_tf.shape[1],)),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(1)
    ])
    
    tf_model.compile(optimizer=Adam(learning_rate=0.005), loss='mse')
    early_stopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)

    history = tf_model.fit(
        X_train_tf, y_train_tf,
        epochs=1000,
        batch_size=16,
        validation_data=(X_test_tf, y_test_tf),
        callbacks=[early_stopping],
        verbose=1
    )

    y_pred_tf = tf_model.predict(X_test_tf).flatten()
    
    #rmse = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
    #print(f"XGBoost Test RMSE: {rmse}")
    #rmse_tf = np.sqrt(mean_squared_error(y_test_tf, y_pred_tf))
    #print(f"TensorFlow Model RMSE: {rmse_tf}")

    latest_data = X.iloc[-1].values.reshape(1, -1)
    today_price = y.iloc[-1]
    #print(f"Actual {ticker} Close Price Today: {today_price}")
    #tomorrow_priceXGB = xgb_model.predict(latest_data)[0]
    #print(f"Predicted XGBoost {ticker} Close Price for Tomorrow: {tomorrow_priceXGB}")
    latest_data_scaled = scaler.transform(pd.DataFrame([X.iloc[-1]], columns=X.columns))
    tomorrow_pricetf = tf_model.predict(latest_data_scaled)[0][0]
    #print(f"Predicted TensorFlow {ticker} Close Price for Tomorrow: {tomorrow_pricetf}")

    # Plot actual vs predicted values
    # plt.figure(figsize=(12, 6))
    # plt.plot(y_test.index, y_test, label=f"Actual {ticker} Close Price", color="blue")
    # plt.plot(y_test.index, y_pred_xgb, label=f"XGB Predicted {ticker} Close Price", color="purple", linestyle="dashed")
    # plt.plot(y_test.index, y_pred_tf, label=f"TensorFlow Predicted {ticker} Close Price", color="green", linestyle="dashed")
    # plt.xlabel("Date")
    # plt.ylabel("Close Price")
    # plt.title(f"{ticker} Actual vs Predicted Prices (XGB & TensorFlow)")
    # plt.legend()
    # plt.show()

    # #Check percentage error over time
    # plt.figure(figsize=(12, 6))
    # plt.plot(y_test.index, abs(y_test - y_pred_xgb), label="XGBoost Prediction Error", color="purple")
    # plt.plot(y_test.index, abs(y_test_tf - y_pred_tf), label="Tensorflow Prediction Error", color="green")
    # plt.xlabel("Date")
    # plt.ylabel("Absolute Error")
    # plt.title(f"{ticker} Prediction Error Over Time (XGB & TensorFlow)")
    # plt.legend()
    # plt.show()

    return tomorrow_pricetf
