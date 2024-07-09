# Import the libraries
import datetime  # For date and time operations
from math import sqrt  # For mathematical operations
import numpy as np  # For numerical operations
from sklearn.preprocessing import MinMaxScaler  # For data scaling
from keras.models import Sequential  # For building sequential models
from keras.layers import Dense, LSTM, Dropout  # For model layers
import tensorflow as tf  # For deep learning models
keras = tf.keras  # Keras interface within TensorFlow
import yfinance as yf  # For fetching stock data

import warnings 
warnings.filterwarnings("ignore")  # To ignore warnings

# Set the date range for stock data
start_date = datetime.date(2013,1,1)
end_date = datetime.datetime.today()

tickerSymbol = 'INFY'  # Set ticker symbol

# Download stock data for the given ticker symbol and date range
stock = yf.download(tickerSymbol, start=start_date, end=end_date, progress=False)
df = stock.reset_index()['Close']  # Reset index and get closing prices

scaler = MinMaxScaler(feature_range=(0,1))  # Initialize scaler
df1 = scaler.fit_transform(np.array(df).reshape(-1,1))  # Scale data

# Split data into training and testing sets
train_size = int(len(df)*0.70)
test_size = len(df) - train_size
train_data, test_data = df1[0:train_size,:], df1[train_size:len(df),:1]

# Convert an array of values into a dataset matrix
def create_dataset(dataset, timestamp=1):
    X, y = [], []
    for i in range(len(dataset) - timestamp - 1):
        a = dataset[i:(i + timestamp), 0]
        X.append(a)
        y.append(dataset[i + timestamp, 0])
    return np.array(X), np.array(y)

timestamp = 100  # Set the timestamp (look-back period)
X_train, y_train = create_dataset(train_data, timestamp)  # Create training data
X_test, y_test = create_dataset(test_data, timestamp)  # Create testing data

# Reshape input to be [samples, time steps, features] required for LSTM
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Build the LSTM model
model = Sequential()
model.add(LSTM(units=50, activation='relu', return_sequences=True, input_shape=(100, 1)))  # First LSTM layer
model.add(Dropout(0.2))  # Dropout layer to prevent overfitting
model.add(LSTM(units=60, activation='relu', return_sequences=True))  # Second LSTM layer
model.add(Dropout(0.3))  # Dropout layer
model.add(LSTM(units=80, activation='relu', return_sequences=True))  # Third LSTM layer
model.add(Dropout(0.4))  # Dropout layer
model.add(LSTM(units=120, activation='relu'))  # Fourth LSTM layer
model.add(Dropout(0.5))  # Dropout layer
model.add(Dense(units=1))  # Output layer
model.compile(optimizer='adam', loss='mean_squared_error')  # Compile the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=64, verbose=2)  # Train the model

model.save('lstm_model.h5')  # Save the trained model
