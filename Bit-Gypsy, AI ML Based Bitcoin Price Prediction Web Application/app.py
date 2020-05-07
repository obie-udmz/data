# to resolve the problem of ModuleNotFoundError: No module named 'binance.client'; 'binance' is not a package #72
# or any similar challemge in the future, consider an earlier version of the library, in this case the latest 
# binance lib version 0.7.5 had client in a different location so the binance.client command was not working 
# I downgraded to an earlier versison using pip install python-binance==0.7.4 which had the client located where 
# I expected. Also the prophet library didn't not complete installation because i had a new and unstable version of
# python (v.3.8) at the time , so i downgraded to v3.7.7
# conda install -c conda-forge fbprophet
# use above instead of pip to install fbprophet
from flask import Flask, request, render_template
import datetime
from binance.client import Client
#import binance as bi
import pandas as pd
from fbprophet import Prophet
import numpy as np
from datetime import date
from datetime import datetime, timedelta


app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method =="POST":
        '''
        For rendering results on HTML GUI
        '''

        feature5=request.form["market_dt"]
        feature6=request.form["market_dt2"]
        weekDays = ("Monday", "Tuesday", "Wednesday",
                    "Thursday", "Friday", "Saturday", "Sunday")
        YrMonth = ("Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")        

        #Client = bi.client.Client

        client = Client('5oJKxLS9xif8zp5bbsjRr67s5h4oWvmCh3D6ZOlbdFNwYDvDkLNXPLfyVjJ53vUT','kxpoCJI9la6MQTOJlJzgNLPy4X4XdRlNkX8QmZJCjPnPDf9vgu3ib46x1veWv9fU')



        symbol = 'BTCUSDT'
        BTC = client.get_historical_klines(symbol=symbol, interval = Client.KLINE_INTERVAL_1DAY  , start_str = '1 year ago UTC')
        # See Binance API support doc > Binance Constants for intervals and 
        # See Binance API support doc >MArket Data Endpoints > Aggregate Trade Iterator for start_str examples
        # KLINE_INTERVAL_1DAY
        # KLINE_INTERVAL_30MINUTE



        # See Binance API support doc > Binance API > client module for column names
        BTC = pd.DataFrame(BTC, columns=['Open time','Open','High','Low','Close','Volume','Close time','Quote asset volume','Number of trades','Taker buy base asset volume','Taker buy quote asset volume','ignored'])

        BTC['Open time'] = pd.to_datetime(BTC['Open time'], unit='ms')

        df_new=BTC[['Open time','Close']]


        df_new=df_new.rename(columns={'Open time':'ds','Close':'y'})


        df_new['y'] = df_new['y'].astype(float)


        df_old = df_new.copy()

        df_old.head()


        df_new['y'] = np.log(df_new['y'])



        m=Prophet(interval_width=0.95, yearly_seasonality=True, weekly_seasonality=True,daily_seasonality=True, changepoint_prior_scale=2)
        m.add_seasonality(name='monthly', period=30.5, fourier_order=5, prior_scale=0.02)
        m.fit(df_new)
        future = m.make_future_dataframe(periods = 365,freq='D')


        forecast = m.predict(future)
        forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]


        all_pred = forecast[['ds','yhat']]


        all_pred['yhat'] = np.exp(all_pred['yhat'])


        #all_pred.set_index('ds',inplace=True)

        result = all_pred.loc[all_pred['ds']==feature5]

        output = result['yhat']


        mkt_dt = datetime.strptime(feature5, '%Y-%m-%d').date()
        dow = mkt_dt.weekday()
        mow = mkt_dt.month
        mkt_dy = weekDays[dow]+' ' + YrMonth[mow-1] + \
            ' ' + str(mkt_dt.day)+', ' + str(mkt_dt.year)

        mkt_dt2 = mkt_dt + timedelta(days=1)
        dow2 = mkt_dt2.weekday()
        mow2 = mkt_dt2.month
        mkt_dy2 = weekDays[dow2]+' ' + YrMonth[mow2-1] + \
            ' ' + str(mkt_dt2.day)+', ' + str(mkt_dt2.year)

        mkt_dt3 = mkt_dt + timedelta(days=2)
        dow3 = mkt_dt3.weekday()
        mow3 = mkt_dt3.month
        mkt_dy3 = weekDays[dow3]+' ' + YrMonth[mow3-1] + \
            ' ' + str(mkt_dt3.day)+', ' + str(mkt_dt3.year)


        result2 = all_pred.loc[all_pred['ds']==str(mkt_dt2)]

        output2 = result2['yhat']

        result3 = all_pred.loc[all_pred['ds']==str(mkt_dt3)]

        output3 = result3['yhat']


    #return render_template('index.html', prediction_text='Predicted Closing Price: ${}'.format(round(output.values[0]),2,2))
    return render_template('index.html', prediction_text='Predicted Closing Price for {} '.format(mkt_dy) + ' is: ${}'.format(round(output.values[0]),2,2),
                            prediction_text2='Predicted Closing Price for {} '.format(mkt_dy2) + ' is: ${}'.format(round(output2.values[0]),2,2),
                            prediction_text3='Predicted Closing Price for {} '.format(mkt_dy3) + ' is: ${}'.format(round(output3.values[0]),2,2))

if __name__ == "__main__":
    app.run(debug=True)

    