import pandas as pd
import yfinance as yf
import datetime
import plotly.graph_objects as go
from autots import AutoTS
from datetime import datetime, timedelta
today = datetime.today()

d1 = today.strftime("%Y-%m-%d")
end_date = d1
d2 = datetime.today() - timedelta(days=1095)
d2 = d2.strftime("%Y-%m-%d")
start_date = d2

data = yf.download('AMZN', 
                      start=start_date, 
                      end=end_date, 
                      progress=False)
data["Date"] = data.index
data = data[["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]]
data.reset_index(drop=True, inplace=True)
print(data.head())

data.shape

figure = go.Figure(data=[go.Candlestick(x=data["Date"],
                                        open=data["Open"], 
                                        high=data["High"],
                                        low=data["Low"], 
                                        close=data["Close"])])
figure.update_layout(title = "Bitcoin Price Analysis", 
                     xaxis_rangeslider_visible=False)
figure.show()

correlation = data.corr()
print(correlation["Close"].sort_values(ascending=False))


model = AutoTS(forecast_length=30, frequency='infer', ensemble='simple')
model = model.fit(data, date_col='Date', value_col='Close', id_col=None)
prediction = model.predict()
forecast = prediction.forecast
print(forecast)