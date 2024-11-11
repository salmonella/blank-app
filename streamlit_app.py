#Importing Libraries
import streamlit as st
import pandas as pd
import base64
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import date
from plotly import graph_objs as go
import pandas_datareader.data as web
from PIL import Image
import datetime as dt
import math
import time
import numpy as np
from bs4 import BeautifulSoup
import requests
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout



#Page Settings
st.set_page_config(page_title="Wall-Street Exchange",page_icon="\N{Cyclone}",layout = "wide")
show_spinner=False

def local_css(file_name):
    f = open("style.css")
    st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)
local_css("style.css")

#MAIN TITLE
st.title("\N{Cyclone}WALL-STREET EXCHANGE")

#MAIN IMG
file = open("5.2.gif", "rb")
contents = file.read()
data_url = base64.b64encode(contents).decode("utf-8")
file.close()
st.markdown(f'<img src="data:image/gif;base64,{data_url}" alt="cat gif">',unsafe_allow_html=True)


#To Hide Hamburger Menu
hide_streamlit_style = """
    <style>
    #download:hover{font-size:1.1rem;  transition: 0.3s;}
    /* This is to hide hamburger menu completely */
    #MainMenu {visibility: hidden;}
    /* This is to hide Streamlit footer */
    footer {visibility: hidden;}
    /*
    If you did not hide the hamburger menu completely,
    you can use the following styles to control which items on the menu to hide.
    */
    ul[data-testid=main-menu-list] > li:nth-of-type(4), /* Documentation */
    ul[data-testid=main-menu-list] > li:nth-of-type(5), /* Ask a question */
    ul[data-testid=main-menu-list] > li:nth-of-type(6), /* Report a bug */
    ul[data-testid=main-menu-list] > li:nth-of-type(7), /* Streamlit for Teams */
    ul[data-testid=main-menu-list] > div:nth-of-type(2) /* 2nd divider */
        {display: none;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
st.markdown("")

#About
about_more = st.expander("\N{newspaper}About")
if about_more:
    about_more.info("""
        • \N{CIRCLED INFORMATION SOURCE}Wall-Street Exchange is a Web-app that collects the data of Stocks & Cryptocurrencies and helps in Analyzing & Price Prediction.
        \n• \N{envelope}Leave a Message : https://www.linkedin.com/in/karansuneja/
    """)


st.divider()




   
    

st.header("\N{speech balloon}Stock/Crypto Data")
plpl1, plpl2 = st.columns(2)
with plpl1:
    #Parameter Section
    st.subheader("Parameters")
    user_inp = st.text_input("\N{small blue diamond}Type Ticker of Stock/Crypto", "MSFT", help="Eg : Tesla = TSLA")
     #Start Date
    start_date = st.date_input("\N{calendar}Select Start Date", dt.date(2021, 1, 1), help = "Should Not Exceed Today's Date")
    #End Date
    end_date = st.date_input("\N{tear-off calendar}Select End Date", help = "Should Not Exceed Today's Date")

# Collecting Data From The Internet
tickerData = yf.Ticker(user_inp)
df = pd.DataFrame(tickerData.history(start=start_date, end=end_date))
df.reset_index(inplace=True)
df['Date'] = pd.to_datetime(df['Date']).dt.date

#Button To Download The Stock Data in A CSV Fromat
def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f"<a href = 'data:file/csv;base64, {b64}' download = 'Data.csv'><div><span class='highlight blue' id=download style=font-weight:700;>Download Data File</span></div></a>"
    return href

with plpl2:
    #Data Section
    st.dataframe(df)
    def V_SPACE(lines):
        for _ in range(lines):
            st.write('&nbsp;')
  


#t = f"<div>Name : <span class='highlight blue'>{tickerData.info['shortName']}</span></div>"
#plpl1.markdown(t, unsafe_allow_html=True)
plpl1.markdown(filedownload(df), unsafe_allow_html=True)

st.divider()

#Analysis Section
st.title('\N{office building}Data Analysis')

st.markdown("")
form1 = st.form('Form1')

#Plotting Graphs For Open & Close Price
form1.header("Stock/Crypto Price Graph")
t = f"<div>This feature shows the Graphical Representation of the <span class='highlight blue'>Open & Close Price</span>  Of the selected Stock/Crypto.</div>"
form1.markdown(t, unsafe_allow_html=True)
form1.markdown("")


submitted1 = form1.form_submit_button('\N{gear}Show Graph')
if submitted1:
    #Plotting Data
    
    def plot_data():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["Date"], y=df["Open"], name="stock_open", line_color = "blue"))
        fig.add_trace(go.Scatter(x=df["Date"], y=df["Close"], name="stock_close", line_color ="green"))
        fig.update_layout(legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1))
        fig.layout.update(xaxis_rangeslider_visible=True, xaxis_showgrid=False, yaxis_showgrid=False)
        fig.update_layout(xaxis=dict(rangeselector=dict(buttons=list([
                    dict(count=1,
                         label="1m",
                         step="month",
                         stepmode="backward"),
                    dict(count=6,
                         label="6m",
                         step="month",
                         stepmode="backward"),
                    dict(count=1,
                         label="1y",
                         step="year",
                         stepmode="backward"),
                    dict(count=1,
                         label="YTD",
                         step="year",
                         stepmode="todate"),
                    dict(step="all")]), bgcolor = '#0078FF'), rangeslider=dict(visible=True),type="date"), plot_bgcolor="#000000", width = 850,height = 550, hovermode="x unified")
        config={'modeBarButtonsToAdd':['drawline','drawopenpath','drawcircle','drawrect','eraseshape'], 'displayModeBar': True, 'displaylogo': False}
        form1.plotly_chart(fig, use_container_width=True, config=config)

    form1.markdown("""<style>.stProgress > div > div > div > div {background-color: #00a3ff;}</style>""",unsafe_allow_html=True)
    plot_data()



#FORM 3
form3 = st.form("Form 3")
#Buy or Sell Indicators
form3.header("Stock/Crypto Buy/Sell Strategy")
t = f"<div>This feature helps to determine when to Buy Or Sell a Stock/Crypto using a <span class='highlight blue'>DEMA Model</span></div>"
form3.markdown(t, unsafe_allow_html=True)
form3.markdown("")

#Learn More
learn_more = form3.expander("How It Works : ")
if learn_more :
    learn_more_text = 'This feature uses Double Exponential Moving Average(DEMA) to determine when to Buy or Sell a Stock/Crypto. It Calculates 2 DEMA values (DEMA_short = 20 Days, DEMA_long = 50 Days) \n- If DEMA_short is more than DEMA_long then, The Buy Signal will be Flagged.\n- If DEMA_short is less than DEMA_long then, The Sell Signal will be Flagged.' 
    learn_more.info(learn_more_text)
    form3.markdown(" ")

submitted3 = form3.form_submit_button('\N{gear}Run Strategy')
if submitted3:
    form3.markdown("""<style>.stProgress > div > div > div > div {background-color: #0078FF;}</style>""",unsafe_allow_html=True)
        
        
    def bssignal(com):
        #Getting Data
        df = tickerData.history(com, start = "2020-1-1", end=end_date)
        #create a function to calculate DEMA
        def DEMA(data, time, column):
            #calculate EMA for some time period
            EMA = data[column].ewm(span=time, adjust=False).mean()
            #calculate the DEMA
            DEMA = 2 * EMA - EMA.ewm(span=time, adjust=False).mean()
            return DEMA

        #store the short term DEMA(20days) and the long term DEMA(50days) into dataset
        df["DEMA_short"] = DEMA(df, 20, "Close")
        df["DEMA_long"] = DEMA(df, 50, "Close")

        #create a function to buy or sell the stock
        def DEMA_strategy(data):
            buy_list = []
            sell_list = []
            flag = False
            #loop through the data
            for i in range(0, len(data)):
                if data["DEMA_short"][i] > data["DEMA_long"][i] and flag == False:
                    buy_list.append(data["Close"][i])
                    sell_list.append(np.nan)
                    flag = True

                elif data["DEMA_short"][i] < data["DEMA_long"][i] and flag == True:
                    buy_list.append(np.nan)
                    sell_list.append(data["Close"][i])
                    flag = False
                else:
                    buy_list.append(np.nan)
                    sell_list.append(np.nan)

            #store the buy and sell signals into dataset
            data["Buy"] = buy_list
            data["Sell"] = sell_list
            

        #run strategy to get buy and sell signals
        DEMA_strategy(df)

        #Editind Data
        df.reset_index(inplace=True)
        df['Date'] = pd.to_datetime(df['Date']).dt.date

        #visually show stock buy and sell signals
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["Date"],y=df["Close"], name="Close Price",line_color = "#0078FF", opacity = 0.40))
        fig.add_trace(go.Scatter(x=df["Date"],y=df["DEMA_short"], name="DEMA_short",line_color = "white"))
        fig.add_trace(go.Scatter(x=df["Date"],y=df["DEMA_long"], name="DEMA_long",line_color = "white"))
        fig.add_trace(go.Scatter(x=df["Date"],y=df["Sell"], name="Buy Signal", line_color = "Green", mode='markers',marker=dict(size=10)))
        fig.add_trace(go.Scatter(x=df["Date"],y=df["Buy"], name="Sell Signal", line_color ="Red", mode = 'markers', marker=dict(size=10)))
        fig.update_layout(autosize=False,width=850,height=500)
        fig.layout.update(plot_bgcolor="#0E1117", xaxis_rangeslider_visible=True)
        fig.update_layout(xaxis_showgrid=False, yaxis_showgrid=False)
        fig.update_layout(xaxis=dict(rangeselector=dict(buttons=list([dict(count=1,label="1m",step="month",stepmode="backward"),
        dict(count=6,
             label="6m",
             step="month",
             stepmode="backward"),
        dict(count=1,
             label="1y",
             step="year",
             stepmode="backward"),
        dict(count=1,
             label="YTD",
             step="year",
             stepmode="todate"),
        dict(step="all")]), bgcolor = '#0078FF'), rangeslider=dict(visible=True),type="date"), plot_bgcolor="#000000", width = 900, hovermode="x unified")
        config={'modeBarButtonsToAdd':['drawline','drawopenpath','drawcircle','drawrect','eraseshape'], 'displayModeBar': True, 'displaylogo': False}
    

        form3.plotly_chart(fig, use_container_width=True, config=config)

        buy_list = df["Buy"]
        sell_list = df["Sell"]
        buy_list = [x for x in buy_list if x == x]
        sell_list = [x for x in sell_list if x == x]
        if buy_list[:-2:-1] > sell_list[:-2:-1]:
            form3.info("The Model has Predicted that You should Sell the Stock/Crypto")
        elif buy_list[:-2:-1] < sell_list[:-2:-1]:
            form3.info("The Model has Predicted that You should Buy the Stocks/Crypto")
            
    bssignal(user_inp)
    form3.warning("WARNING : Generated Predictions shouldn't Considered to be 100% Correct, This App only works on currently available data of the Stock/Crypto.")


#FORM 4
form4 = st.form("Form 4")

#OHLC Charts
form4.header("Stock/Crypto Candlestick Charts")
t = f"<div>This feature shows the <span class='highlight blue'>OHLC Candlestick Charts</span> of the selected Stock/Crypto.</div>"
form4.markdown(t, unsafe_allow_html=True)
form4.markdown("")

#Learn More
learn_more = form4.expander("How It Works : ")
if learn_more :
    learn_more_text = "This graph shows the Open-High-Low-Close(OHLC) Candlestick Charts which is used to illustrate movements in the price of a financial instrument over time. \n- Select the Data Duration From Below to set the Data length. \n- Select the Interval From Below to set the Time Range."
    learn_more.info(learn_more_text)
    form4.markdown(" ")

q1111, q1112 = form4.columns(2)
with q1111:
    p = st.selectbox("Data Duration (in Days)",("1", "7","31", "365"))
with q1112:
    i = st.selectbox("Interval (in Minutes)",("1m","5m", "15m", "60m","1mo", "3mo"))

submitted4 = form4.form_submit_button('\N{gear}Submit')
if submitted4:
            
    form4.markdown("""<style>.stProgress > div > div > div > div {background-color: #0078FF;}</style>""",unsafe_allow_html=True)


    #Collecting Data
    history_data = tickerData.history(interval = i, period = str(p) + "d")
    history_data.reset_index(inplace=True)
    
    if "Datetime" not in history_data:
        history_data = history_data.set_index((history_data["Date"]))
        history_data.rename(columns = {'Datetime' : 'Date'}, inplace=True)
                            
    else:
        history_data = history_data.set_index((history_data["Datetime"]))
        history_data.rename(columns = {'Datetime' : 'Date'}, inplace=True)
        

    #Plotting OHLC CHarts
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=history_data['Date'],
                    open=history_data['Open'],
                    high=history_data['High'],
                    low=history_data['Low'],
                    close=history_data['Close'],
                    yaxis= "y2",increasing_line_color= '#0078FF', decreasing_line_color= 'gray', name= "Gain/Loss"  ))
    fig.update_layout(width=800,height=500, margin=dict(l=10, r=10, t=50, b=50), yaxis2 = dict(title = "Price",side="right"))
    fig.update_layout(showlegend=True, legend=dict(yanchor="bottom",y=1,xanchor="left",x=0.01))
    fig.update_yaxes(nticks=20, showgrid=False)
    fig.update_layout(plot_bgcolor="#000000", xaxis_showgrid=False, yaxis_showgrid=False, hovermode="x unified")
    config={'modeBarButtonsToAdd':['drawline','drawopenpath','drawcircle','drawrect','eraseshape'], 'displayModeBar': True, 'displaylogo': False}
    

    form4.plotly_chart(fig, use_container_width=True, config=config)

exp = st.expander("Experimental Feature")
if exp :
    #Stock Predictor
    exp.header("Stock/Crypto Price Prediction")
    t = f"<div>This feature predicts the Price of the Stock/Crypto using a <span class='highlight blue'>Machine Learning Model</span></div>"
    exp.markdown(t, unsafe_allow_html=True)
    exp.markdown("")
  
    submitted2 = exp.button('\N{gear}Run Model')
    if submitted2:
        exp.text("[This may take a while]")
            
        def pred(user_inp):
            #getting previous data
            start = dt.datetime(2012,1,1)
            end = dt.datetime(2020,1,1)
            data = tickerData.history(user_inp,start = start, end=end)
            data.reset_index(inplace=True)
            df['Date'] = pd.to_datetime(df['Date']).dt.date
    
            #scale the data
            scaler = MinMaxScaler(feature_range=(0,1))
            scaled_data = scaler.fit_transform(data["Close"].values.reshape(-1,1))
            prediction_days = 60
            #create the training data set
            #split the data into x_train and y_train data sets
            x_train = []
            y_train = []
            for x in range(prediction_days, len(scaled_data)):
                x_train.append(scaled_data[x-prediction_days:x, 0])
                y_train.append(scaled_data[x, 0])
            
            #convert the x_train and y_train to numpy arrays
            x_train, y_train = np.array(x_train), np.array(y_train)
            #reshape the data as lstm models expects 3-dimensional data
            x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    
            #build the lstm model
            model = Sequential()
            model.add(LSTM(units=50, return_sequences=True, input_shape = (x_train.shape[1], 1)))
            model.add(Dropout(0.2))
            model.add(LSTM(50, return_sequences=True))
            model.add(Dropout(0.2))
            model.add(LSTM(units=50))
            model.add(Dropout(0.2))
            model.add(Dense(1)) #prediction of the next closing value
    
            #compile the model
            model.compile(optimizer ='adam', loss='mean_squared_error')
            
            #train the model
            model.fit(x_train, y_train, epochs=25, batch_size=32)
            
            #TESTING THE MODEL ON EXISTING DATA
            #create the testing dataset
            test_start = dt.datetime(2020,1,1)
            test_end = dt.datetime.now()
            test_data = tickerData.history(user_inp,start=test_start,end=test_end)
            test_data.reset_index(inplace=True)
            test_data['Date'] = pd.to_datetime(test_data['Date']).dt.date
            
            actual_prices = test_data["Close"].values
            total_dataset = pd.concat((data["Close"], test_data["Close"]), axis=0)
    
            model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
            model_inputs = model_inputs.reshape(-1,1)
            model_inputs = scaler.transform(model_inputs)
    
            #create the datasets x_test and y_test
            x_test = []
            for i in range(prediction_days, len(model_inputs)):
                 x_test.append(model_inputs[i-prediction_days:i, 0])
    
            #convert the data to numpy
            x_test = np.array(x_test)
    
            #reshape the data
            x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    
            #get the model's predicted price values
            predicted_prices = model.predict(x_test)
            predicted_prices = scaler.inverse_transform(predicted_prices)
    
            #Editing test_data dataframe
            test_data["Predicted"] = predicted_prices
            
            #Plotting Data
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=test_data["Date"], y=test_data["Close"], name="actual_price", line_color= "cyan"))
            fig.add_trace(go.Scatter(x=test_data["Date"], y=test_data["Predicted"], name="predicted_price", line_color= "blue"))
            fig.update_layout(legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1))
            fig.layout.update(xaxis_rangeslider_visible=True, xaxis_showgrid=False, yaxis_showgrid=False)
            fig.update_layout(xaxis=dict(rangeselector=dict(buttons=list([
                        dict(count=1,
                             label="1m",
                             step="month",
                             stepmode="backward"),
                        dict(count=6,
                             label="6m",
                             step="month",
                             stepmode="backward"),
                        dict(count=1,
                             label="1y",
                             step="year",
                             stepmode="backward"),
                        dict(count=1,
                             label="YTD",
                             step="year",
                             stepmode="todate"),
                        dict(step="all")]), bgcolor = '#0078FF'), rangeslider=dict(visible=True),type="date"), plot_bgcolor="#000000", width = 850, height = 550, hovermode="x unified")
            config={'modeBarButtonsToAdd':['drawline','drawopenpath','drawcircle','drawrect','eraseshape'], 'displayModeBar': True, 'displaylogo': False}
            exp.plotly_chart(fig, use_container_width=True, config=config)
    
            #Printing Prediced Data
            real_data = [model_inputs[len(model_inputs) + 1 - prediction_days:len(model_inputs) + 1, 0]]
            real_data = np.array(real_data)
            real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))
            prediction = model.predict(real_data)
            prediction = scaler.inverse_transform(prediction)
            comp_close = []
            comp_close = test_data["Close"].tolist()
            col1, col2 = exp.columns(2)
            if comp_close[:-2:-1] < prediction:
                col1.info("The Model Has Predicted That the Close Price Will Go Higher")
            else:
                col1.info("The Model Has Predicted That the Close Price Will Go Lower")
            col2.info(f"Prediciton Of Closing Price of Stock/Crypto : {prediction}")
            

            
        pred(user_inp)
        exp.warning("WARNING : Generated Predictions shouldn't Considered to be 100% Correct, This Web-App only works on Currently Availbale Data of the Stock/Crypto.")



st.caption("Designed & Developed by Karan Suneja.")
st.caption("©All rights reserved.")