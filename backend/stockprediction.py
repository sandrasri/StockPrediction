import pandas as pd
import numpy as np
from pytickersymbols import PyTickerSymbols
import yfinance as yf
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

#Set up start and end dates
start_date = "2015-01-01"
end_date = pd.Timestamp.today().strftime('%Y-%m-%d')

#Download the top 100 tickers from sp500
# first_step = PyTickerSymbols()
# sp500_tickers = list(first_step.get_stocks_by_index('S&P 500'))
# tickers = [stock['symbol'] for stock in sp500_tickers[:100]]
# top_100 = yf.download(tickers=' '.join(tickers), start=start_date, end=end_date, auto_adjust=True, threads=True)
# top_100_data = pd.DataFrame(top_100)

#Download of any ticker required
def downloadTicker(ticker):
    data = yf.download(ticker, start=start_date, end=end_date)
    datasheet = pd.DataFrame(data)

    if datasheet.index.name == "Date":
        datasheet = datasheet.reset_index()
    datasheet = datasheet[['Date', 'Close', 'Volume']]
    datasheet.columns = datasheet.columns.droplevel('Ticker')
    datasheet.columns.name = None
    return datasheet

#Define static variables --> Need to get from User
ticker = 'AAPL'
stock_name = downloadTicker(ticker)

#Define the columns and X,y for the ML application
def stock_choice(ticker_name, best_date):
    #stock_data = top_100_data.loc[:, ["Date_", f"Close_{ticker_name}", f"Volume_{ticker_name}"]]
    stock_data = ticker_name[ticker_name['Date'] > best_date].copy()
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
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
    print(f"XGBoost Test RMSE: {rmse}")
    rmse_tf = np.sqrt(mean_squared_error(y_test_tf, y_pred_tf))
    print(f"TensorFlow Model RMSE: {rmse_tf}")

    latest_data = X.iloc[-1].values.reshape(1, -1)
    today_price = y.iloc[-1]
    print(f"Actual {ticker} Close Price Today: {today_price}")
    tomorrow_priceXGB = xgb_model.predict(latest_data)[0]
    print(f"Predicted XGBoost {ticker} Close Price for Tomorrow: {tomorrow_priceXGB}")
    latest_data_scaled = scaler.transform(pd.DataFrame([X.iloc[-1]], columns=X.columns))
    tomorrow_pricetf = tf_model.predict(latest_data_scaled)[0][0]
    print(f"Predicted TensorFlow {ticker} Close Price for Tomorrow: {tomorrow_pricetf}")

    # Plot actual vs predicted values
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.index, y_test, label=f"Actual {ticker} Close Price", color="blue")
    plt.plot(y_test.index, y_pred_xgb, label=f"XGB Predicted {ticker} Close Price", color="purple", linestyle="dashed")
    plt.plot(y_test.index, y_pred_tf, label=f"TensorFlow Predicted {ticker} Close Price", color="green", linestyle="dashed")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.title(f"{ticker} Actual vs Predicted Prices (XGB & TensorFlow)")
    plt.legend()
    plt.show()

    #Check percentage error over time
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.index, abs(y_test - y_pred_xgb), label="XGBoost Prediction Error", color="purple")
    plt.plot(y_test.index, abs(y_test_tf - y_pred_tf), label="Tensorflow Prediction Error", color="green")
    plt.xlabel("Date")
    plt.ylabel("Absolute Error")
    plt.title(f"{ticker} Prediction Error Over Time (XGB & TensorFlow)")
    plt.legend()
    plt.show()

    return tomorrow_pricetf

#reshape data for tensorflow LSTM
def reshape_data_shortlstm(X, y, time_steps=10):
    X_reshaped = []
    y_reshaped = []
    
    for i in range(len(X) - time_steps):
        X_reshaped.append(X.iloc[i:i + time_steps].values)
        y_reshaped.append(y[i + time_steps])
    
    return np.array(X_reshaped), np.array(y_reshaped)

def predictShortTerm(ticker_name):
    X, y = stock_choice(ticker_name, best_date(ticker_name))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # XGBoost Model
    xgb_model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100, learning_rate=0.1)
    xgb_model.fit(X_train, y_train)

    y_pred_xgb = xgb_model.predict(X_test)

    #XGB Data Scaling
    scaler = StandardScaler()
    
    # Data Scaling - Scale the features but not the target variable
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    
    # Use a separate scaler for the target variable (Close price)
    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))  # Ensure y is in the correct shape
    
    # Reshape data for LSTM input
    X_reshaped, y_reshaped = reshape_data_shortlstm(pd.DataFrame(X_scaled), y_scaled)  # Ensure reshape_data is properly defined
    X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm = train_test_split(X_reshaped, y_reshaped, test_size=0.2, shuffle=False)

    # LSTM Model
    lstm_model = Sequential([
        LSTM(64, activation='relu', input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2]), return_sequences=True),
        Dropout(0.3),
        LSTM(32, activation='relu', return_sequences=False),
        Dropout(0.3),
        Dense(1)
    ])

    lstm_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    early_stopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)
    
    lstm_model.fit(
        X_train_lstm, y_train_lstm,
        epochs=1000,
        batch_size=16,
        validation_data=(X_test_lstm, y_test_lstm),
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Predict the Next 14 Days' Prices
    last_data = X_scaled[-10:].reshape(1, X_reshaped.shape[1], X_reshaped.shape[2])  # Ensure correct shape
    short_term_preds_lstm = []

    for i in range(1,15):
        next_day_price_scaled = lstm_model.predict(last_data)[0][0]
        short_term_preds_lstm.append(next_day_price_scaled)
        
        # Update the input data for the next prediction
        last_data = np.roll(last_data, shift=-1, axis=1)
        last_data[0, -1, 0] = next_day_price_scaled  # Add the predicted price to the last time step
    
    # Inverse transform the predictions to the original scale
    short_term_preds_lstm = scaler_y.inverse_transform(np.array(short_term_preds_lstm).reshape(-1, 1)).flatten()

    # Print predictions for the next 14 days
    print(f"Predicted {ticker_name} Close Prices for the Next 14 Days (LSTM):")
    for i, lstm_price in enumerate(short_term_preds_lstm, 1):
        print(f"Day {i}: {lstm_price:.2f}")
        return lstm_price
        
    # Plot actual vs predicted values for short-term predictions
    plt.figure(figsize=(12, 6))
    predicted_dates = pd.date_range(y_test.index[-1] + pd.Timedelta(days=1), periods=14, freq="D")
    plt.plot(y_test.index, y_test, label=f"Actual {ticker_name} Close Price", color="blue")
    plt.plot(predicted_dates, short_term_preds_lstm, label=f"TensorFlow LSTM Predicted {ticker_name} Close Price", color="green", linestyle="dashed")
    plt.xlabel("Days")
    plt.ylabel("Close Price")
    plt.title(f"{ticker_name} Actual vs Predicted Prices (XGB & LSTM) - Next 14 Days")
    plt.show()

def reshape_data_long_lstm(X, y, time_steps=10):
    X_reshaped = []
    y_reshaped = []
    
    for i in range(len(X) - time_steps):
        X_reshaped.append(X.iloc[i:i + time_steps].values)  # This is fine as X is a DataFrame
        y_reshaped.append(y[i + time_steps])  # Use numpy array indexing here
    
    return np.array(X_reshaped), np.array(y_reshaped)

def predictLongTerm(ticker_name):
    X, y = stock_choice(ticker_name, best_date(ticker_name))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Data Scaling - Scale the features but not the target variable
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    
    # Use a separate scaler for the target variable (Close price)
    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))  # Ensure y is in the correct shape
    
    # Reshape data for LSTM input
    X_reshaped, y_reshaped = reshape_data_long_lstm(pd.DataFrame(X_scaled), y_scaled)  # Ensure reshape_data is properly defined
    X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm = train_test_split(X_reshaped, y_reshaped, test_size=0.2, shuffle=False)

    # LSTM Model
    lstm_model = Sequential([
        LSTM(64, activation='relu', input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2]), return_sequences=True),
        Dropout(0.3),
        LSTM(32, activation='relu', return_sequences=False),
        Dropout(0.3),
        Dense(1)
    ])
    
    lstm_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    early_stopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)
    
    lstm_model.fit(
        X_train_lstm, y_train_lstm,
        epochs=1000,
        batch_size=16,
        validation_data=(X_test_lstm, y_test_lstm),
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Predict the Next 365 Days' Prices (Approximating for the next year)
    last_data = X_scaled[-10:].reshape(1, X_reshaped.shape[1], X_reshaped.shape[2])  # Ensure correct shape
    
    # Prepare an array to hold predictions for the next year (365 days)
    long_term_pred_lstm = []
    
    for _ in range(365):  # Predict for the next year (365 days)
        next_day_price_scaled = lstm_model.predict(last_data)[0][0]
        long_term_pred_lstm.append(next_day_price_scaled)
        
        # Update the input data for the next prediction
        last_data = np.roll(last_data, shift=-1, axis=1)
        last_data[0, -1, 0] = next_day_price_scaled  # Add the predicted price to the last time step
    
    # Inverse transform the predictions to the original scale
    long_term_pred_lstm = scaler_y.inverse_transform(np.array(long_term_pred_lstm).reshape(-1, 1)).flatten()

    # Print predictions for the next 365 Days
    print(f"Predicted {ticker_name} Close Prices for the Next 1 Year (XGBoost & LSTM):")
    for i, lstm_price in enumerate(long_term_pred_lstm, 1):
        print(f"Day {i}: {lstm_price:.2f}")
        return lstm_price
    
    # Plot actual vs predicted values for monthly averages
    plt.figure(figsize=(12, 6))
    predicted_dates = pd.date_range(y_test.index[-1] + pd.Timedelta(days=1), periods=365, freq="D")
    plt.plot(y_test.index, y_test, label=f"Actual {ticker_name} Close Price", color="blue")
    plt.plot(predicted_dates, long_term_pred_lstm, label=f"TensorFlow LSTM Predicted {ticker_name} Close Price", color="green", linestyle="dashed")
    plt.xlabel("Days")
    plt.ylabel("Close Price")
    plt.title(f"{ticker_name} Actual vs Predicted Prices (XGB & LSTM) - Next Year in Monthly Averages")
    plt.show()

def predictShortorLongTerm(ticker_name, horizon_type):
    X, y = stock_choice(ticker_name, best_date(ticker_name))
    
    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # XGBoost Model
    xgb_model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100, learning_rate=0.1)
    xgb_model.fit(X_train, y_train)

    y_pred_xgb = xgb_model.predict(X_test)
    
    # Prediction for tomorrow's price - LSTM (more accurate)
    if horizon_type == "tomorrow":
        predictTomorrow(ticker_name)

    # Prediction for 14 days - XGB & LSTM
    elif horizon_type == "short-term":
        predictShortTerm(ticker_name)

    elif horizon_type == "long-term":
        predictLongTerm(stock_name)       

    else:
        print("Invalid horizon type")

#predictShortorLongTerm(stock_name, "short-term") --> Also User Input

