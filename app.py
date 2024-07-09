# IMPORT THE LIBRARIES
import streamlit as st  # For creating the web app
import numpy as np  # For numerical operations
import datetime  # For date and time operations
import pandas as pd  # For data manipulation
import matplotlib.pyplot as plt  # For plotting graphs
import plotly.express as px  # For interactive plots
import yfinance as yf  # For fetching stock data
from sklearn.preprocessing import MinMaxScaler  # For data scaling
import plotly.graph_objects as go  # For creating more complex plots
import tensorflow as tf  # For deep learning models
keras = tf.keras  # Keras interface within TensorFlow
from keras.models import load_model  # For loading pre-trained models

import warnings 
warnings.filterwarnings("ignore")  # To ignore warnings

st.sidebar.title("Stock Price Prediction System")  # Sidebar title for the web app

print('\n')  # Print a newline for better readability in console

# Set the date range for stock data
start_date = datetime.date(2018,1,1)
end_date = datetime.datetime.today()

tickerSymbol = ' '  # Initialize ticker symbol variable

# Get user input for the ticker symbol
tickerSymbol = st.sidebar.text_input("Enter Ticker Symbol:")

try:
    # Download stock data for the given ticker symbol and date range
    stock = yf.download(tickerSymbol, start=start_date, end=end_date, progress=False)

    st.subheader(f'{tickerSymbol} STOCK PRICE DATA')  # Display the ticker symbol
    st.write(stock)  # Display the stock data

    # Data Visualization options
    chart = st.sidebar.selectbox("Graph:", ("Select", "Line", "Area", "Candle Sticks"))

    st.subheader('Data Visualisation')
    if chart == 'Line':
        st.markdown('Line Chart')
        st.line_chart(stock['Close'], use_container_width=True)  # Line chart for stock closing prices
    elif chart == 'Area':
        st.markdown('Area Chart')
        st.area_chart(stock['Close'], use_container_width=True)  # Area chart for stock closing prices
    elif chart == 'Candle Sticks':
        st.markdown('Candle Stick')
        figure = go.Figure(data=[go.Candlestick(x=stock.index,
                                                open=stock["Open"], high=stock["High"],
                                                low=stock["Low"], close=stock["Close"])])
        st.plotly_chart(figure)  # Candle stick chart for stock prices
    else:
        st.write('CHOOSE AN APPROPRIATE OPTION')  # Prompt user to choose an option

    # Forecasting
    no_days = st.sidebar.number_input('No. of days for Forecasting:')  # Get user input for number of days to forecast

    st.subheader(f'Forecasting for {no_days} days')  # Display forecasting subheader

    df = stock.reset_index()['Close']  # Reset index and get closing prices
    model = load_model('lstm_model.h5')  # Load pre-trained LSTM model

    # Function to perform forecasting
    def fun(no_days):
        scaler = MinMaxScaler(feature_range=(0,1))  # Initialize scaler
        df1 = scaler.fit_transform(np.array(df).reshape(-1,1))  # Scale data

        # Split data into training and testing sets
        train_size = int(len(df)*0.70)
        test_size = len(df) - train_size
        train_data, test_data = df1[0:train_size,:], df1[train_size:len(df),:1]
        x_input = test_data[test_size-100:].reshape(1,-1)
        temp_input = list(x_input)
        temp_input = temp_input[0].tolist()

        lst_output = []
        n_steps = 100
        i = 0
        while(i <= no_days):
            if(len(temp_input) > 100):
                x_input = np.array(temp_input[1:])
                x_input = x_input.reshape(1, -1)
                x_input = x_input.reshape((1, n_steps, 1))
                yhat = model.predict(x_input, verbose=0)
                temp_input.extend(yhat[0].tolist())
                temp_input = temp_input[1:]
                lst_output.extend(yhat.tolist())
                i = i + 1
            else:
                x_input = x_input.reshape((1, n_steps, 1))
                yhat = model.predict(x_input, verbose=0)
                temp_input.extend(yhat[0].tolist())
                lst_output.extend(yhat.tolist())
                i = i + 1

        pred = scaler.inverse_transform(lst_output)  # Inverse transform to get actual values
        output = [element for innerList in pred for element in innerList] 
        day_new = np.arange(1,101)
        day_pred = np.arange(101,131)
        return output

    if tickerSymbol != ' ':
        output = fun(no_days)  # Get forecasted values
        output = pd.DataFrame(output)  # Convert to DataFrame
        df2 = pd.concat([df, output], ignore_index=True)  # Concatenate original and forecasted data
        df2.rename(columns = {0:'Close'}, inplace = True)  # Rename column to 'Close'
        df3 = df2[1200:]  # Get subset of data for plotting
        st.line_chart(df3, use_container_width=True)  # Plot forecasted data
        st.write(output)  # Display forecasted values
    else:
        pass  # Do nothing if ticker symbol is not provided
except:
    pass  # Ignore errors silently
