!pip install fbprophet
!pip install pytrends
!pip install pytrends

import quandl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import fbprophet
import pytrends
from pytrends.request import TrendReq

from keras.models import Sequential
from keras.layers import LSTM,Dense
from sklearn.preprocessing import MinMaxScaler

class Stock(object):
  # initialize the stock data
  def __init__(self,ticker):
    
    ticker = ticker.upper()
    
    self.symbol = ticker
    
    quandl.ApiConfig.api_key = 'saU3Kye_asAxvpXoxUC_'
    # Retrieval the financial data
    try:
      stock = quandl.get('WIKI/%s'%ticker)
    except Exception as e:
      print('Error Retrieving Data')
      print(e)
      return
    # Set the index to a column called Date
    stock = stock.reset_index(level=0)
    # Columns required for prophet
    stock['ds'] = stock['Date']
    stock['y'] = stock['Adj. Close']
    # Data assigned as class attribute
    self.stock = stock
    # Minimum and maximum date in range
    self.min_date = np.min(stock['ds'])
    self.max_date = np.max(stock['ds'])
    # This can be changed by user
    self.changepoint_prior_scale = 0.2
    
    print('{} Stocker initialized, Data covers from {} to {}'.format(self.symbol,self.min_date.date(),self.max_date.date()))

  # Basic Historical Plot
  def plot(self,start_date = None, end_date = None):
    
    if not start_date:
      start_date = self.min_date
    if not end_date:
      end_date = self.max_date
    
    # Convert to pandas date time
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    # Check to make sure dates are in the data
    if start_date.date() > end_date.date():
      print('Start date must be later than end date')
      return
    
    if (start_date not in list(self.stock['ds'])):
      print('Start Date not in data (either out of range or not a trading day.)')
      return
    elif (end_date not in list(self.stock['ds'])):
      print('End Date not in data (either out of range or not a trading day.)')
      return
    
    stock_plot = self.stock[(self.stock['ds'] >= start_date.date())&(self.stock['ds'] <= end_date.date())]
    
    plt.style.use('seaborn-notebook')
    plt.plot(stock_plot['ds'], stock_plot['y'],color='b',linewidth = 1)
    plt.title("{} Stock Price from {} to {}".format(self.symbol,start_date.date(),end_date.date()))
    plt.ylabel('Closing Price($)')
  
  # Create a prophet model without training
  def __create_model(self,**kwargs):
    # Make the model
    model = fbprophet.Prophet(daily_seasonality = False, weekly_seasonality = False,
                              changepoint_prior_scale = self.changepoint_prior_scale,
                              **kwargs)
    model.add_seasonality(name = 'monthly', period = 30.5, fourier_order = 5)
    
    return model
  
  # Method to linearly interpolate prices on the weekends
  def __resampling(self, dataframe):
    # Change the index and resample at daily level
    dataframe = dataframe.reset_index('ds')
    dataframe = dataframe.resample('D')
    # Reset the index and interpolate nan values
    dataframe = dataframe.reset_index(level = 0)
    dataframe = dataframe.interpolate()
    
    return dataframe
  
  # Remove weekends from a dataframe
  def __remove_weekends(self, dataframe):
    # Reset index to use ix
    dataframe = dataframe.reset_index(drop=True)
        
    weekends = []
        
    # Find all of the weekends
    for i, date in enumerate(dataframe['ds']):
      if (date.weekday()) == 5 | (date.weekday() == 6):
        weekends.append(i)
            
    # Drop the weekends
    dataframe = dataframe.drop(weekends, axis=0)
        
    return dataframe
  
  # Train Prophet model and plot the fitted values
  def create_prophet_model(self, years = 5, resample = False):
    
    # Set how many years of historical data used to train model 
    self.years = years
    
    stock_history = self.stock[self.stock['ds'] > (self.max_date - pd.DateOffset(years = self.years)).date()]
    
    model = self.__create_model()
    
    if resample:
      stock_history = self.__resampling(stock_history)
    # Train model and fit for observations
    model.fit(stock_history)
    future = model.make_future_dataframe(periods=0, freq='D')
    predictions = future.copy()
    mp = model.predict(future)
    # Design result dataframe
    predictions['yhat_upper'] = mp['yhat_upper']
    predictions['yhat_lower'] = mp['yhat_lower']
    predictions['yhat'] = mp['yhat']
    # Plot the fitted results
    plt.style.use('seaborn-notebook')
    fig,ax = plt.subplots(1,1)
    ax.plot(stock_history['ds'],stock_history['y'],color='b',label = 'Observations')
    ax.plot(predictions['ds'],predictions['yhat'],linewidth = 1,label='Fitted',color="red")
    ax.fill_between(predictions['ds'].dt.to_pydatetime(), predictions['yhat_upper'],predictions['yhat_lower'],alpha = 0.3, facecolor='red', edgecolor = 'k', linewidth = 0.6)
    plt.legend(loc='best')
    plt.ylabel('Closing Price($)')
    plt.title('{} Stock Price'.format(self.symbol))
    plt.show()
    return model, predictions  
  
  # Using trained model to predict future and analysis results
  def predict_feature(self, model, days = 365, resample = False):
    
    # Used for plot
    stock_history = self.stock[self.stock['ds'] > (self.max_date - pd.DateOffset(years = self.years)).date()]
    # Making predictions
    future = model.make_future_dataframe(periods = days, freq = 'D')
    future = model.predict(future)
    
    future_ = future[future['ds'] > self.max_date.date()]
    # Remove weekends
    future_ = self.__remove_weekends(future_)

    trading_days = future_.shape[0]
    # Analysis results
    future_['diff'] = future_['yhat'].diff()
    
    future_ = future_.dropna()
    
    future_['direction'] = (future_['diff']>0)*1
    
    future_ = future_.rename(columns = {'ds':'Date','yhat':'Estimate','diff':'change','yhat_upper':'upper','yhat_lower':'lower'})
    
    num_day_increase = future_[future_['direction'] == 1].shape[0]
    num_day_decrease = trading_days - num_day_increase
    
    print("Among {} days, {} days are trading days".format(days,trading_days))
    print("Among {} trading days, {} days are predicted increasing, {} days are predicted decreasing".format(trading_days,num_day_increase,num_day_decrease))
    
    # Plot future prediction with confidence interval
    plt.style.use('seaborn-notebook')
    fig,ax = plt.subplots(1,1)
    ax.plot(stock_history['ds'],stock_history['y'],color='b',label = 'Observations')
    ax.plot(future['ds'],future['yhat'],linewidth = 1,label='Fitted',color="red")
    ax.fill_between(future['ds'].dt.to_pydatetime(), future['yhat_upper'],future['yhat_lower'],alpha = 0.3, facecolor='red', edgecolor = 'k', linewidth = 0.6,label = 'Uncertainty')
    plt.vlines(x=self.max_date.date(), ymin=min(future['yhat_lower']), ymax=max(future['yhat_upper']), colors = 'r',
                   linestyles='dashed', label = 'Prediction Start')
    plt.legend(loc='best')
    plt.ylabel('Closing Price($)')
    plt.title('{} Stock Price'.format(self.symbol))
    plt.show()
    
    return future_
  
  # Split data into training and test, train model with training data and evaluate result on test data
  def train_prophet_model(self,years,metrics = 'rmse'):
    
    start_date = self.max_date - pd.DateOffset(years = 1)
    # Obtain training and test day. Default we use last 1 year as test data, last 3 years as training data.
    train = self.stock[(self.stock['ds'] < start_date.date()) &(self.stock['ds'] > (start_date - pd.DateOffset(years = years)).date())]
    test = self.stock[self.stock['ds'] >= start_date.date()]
    
    eval_days = (max(test['ds'])-min(test['ds'])).days
    # Build model and fit
    model = self.__create_model()
    model.fit(train)
    # Making prediction on test data
    future = model.make_future_dataframe(periods = eval_days,freq = 'D')
    future = model.predict(future)
    # Evaluate results
    train_results = pd.merge(train,future[['ds','yhat','yhat_upper','yhat_lower']],on='ds',how='inner')
    if metrics == 'rmse':
      ave_train_error = np.sqrt(np.mean(np.square(train_results['y'] - train_results['yhat'])))
    elif metrics == 'mae':
      ave_train_error = np.mean(abs(train_results['y'] - train_results['yhat']))
    
    test_results = pd.merge(test,future[['ds','yhat','yhat_upper','yhat_lower']],on='ds',how='inner')
    if metrics == 'rmse':
      ave_test_error = np.sqrt(np.mean(np.square(test_results['y'] - test_results['yhat'])))
    elif metrics == 'mae':
      ave_test_error = np.mean(abs(test_results['y'] - test_results['yhat']))
    
    print("Under the metrics {}, the training error: {}, the test error: {}".format(metrics, round(ave_train_error,4),round(ave_test_error,4)))    
    # Plot the reuslts
    plt.style.use('seaborn-notebook')
    fig,ax = plt.subplots(1,1)
    ax.plot(train['ds'],train['y'],color='b',label = 'Observations')
    ax.plot(test['ds'],test['y'],color='b') 
    ax.plot(future['ds'],future['yhat'],linewidth = 1,label='Fitted',color="red")
    ax.fill_between(future['ds'].dt.to_pydatetime(), future['yhat_upper'],future['yhat_lower'],alpha = 0.3, facecolor='yellow', edgecolor = 'k', linewidth = 0.6,label = 'Uncertainty')
    plt.vlines(x=start_date.date(), ymin=min(future['yhat_lower']), ymax=max(future['yhat_upper']), colors = 'r',
                   linestyles='dashed', label = 'Prediction Start')
    plt.legend(loc='best')
    plt.ylabel('Closing Price($)')
    plt.title('{} Stock Price'.format(self.symbol))
    plt.show()
  
  # Tune changepoint_prior_scale by validation method
  def changepoint_prior_scale_validation(self,changepoint_prior_scale = [0.01,0.1,0.3,0.5],metrics = 'rmse'):
    
    start_date = self.max_date - pd.DateOffset(years = 1)
    
    train = self.stock[(self.stock['ds'] < start_date.date()) &(self.stock['ds'] > (start_date - pd.DateOffset(years = 3)).date())]
    test = self.stock[self.stock['ds'] >= start_date.date()]
    
    eval_days = (max(test['ds'])-min(test['ds'])).days
    
    results = pd.DataFrame(0,index=list(range(len(changepoint_prior_scale))), columns = ['cps', 'train_err','test_err'])
    
    for i,prior in enumerate(changepoint_prior_scale):
      results.ix[i, 'cps'] = prior
      self.changepoint_prior_scale = prior
      model = self.__create_model()
      model.fit(train)
    
      future = model.make_future_dataframe(periods = eval_days,freq = 'D')
      future = model.predict(future)
    
      train_results = pd.merge(train,future[['ds','yhat','yhat_upper','yhat_lower']],on='ds',how='inner')
      if metrics == 'rmse':
        ave_train_error = np.sqrt(np.mean(np.square(train_results['y'] - train_results['yhat'])))
      elif metrics == 'mae':
        ave_train_error = np.mean(abs(train_results['y'] - train_results['yhat']))
    
      test_results = pd.merge(test,future[['ds','yhat','yhat_upper','yhat_lower']],on='ds',how='inner')
      if metrics == 'rmse':
        ave_test_error = np.sqrt(np.mean(np.square(test_results['y'] - test_results['yhat'])))
      elif metrics == 'mae':
        ave_test_error = np.mean(abs(test_results['y'] - test_results['yhat']))
      
      results.ix[i,'train_err'] = ave_train_error
      results.ix[i,'test_err'] = ave_test_error
    print(results)
    
    plt.style.use('seaborn-notebook')
    plt.plot(results['cps'],results['train_err'],'.-',color='b',label = "Train Error")
    plt.plot(results['cps'],results['test_err'],'.-',color='orange',label = "Test Error")
    plt.title('Training and Testing Curves as Function of CPS')
    plt.xticks(results['cps'], results['cps'])
    plt.legend(loc='best')
    plt.show()
  
  # Retrieval search frequency from google trends
  def retrieve_google_trends(self,term,date_range):
    # Set up the trend fetching object
    pytrends = TrendReq(hl='en-US', tz=360)
    kw_list = [term]
    self.term = term
    try:
      # Create the search object
      pytrends.build_payload(kw_list, cat=0, timeframe=date_range, geo='', gprop='')
            
      # Retrieve the interest over time
      trends = pytrends.interest_over_time()
    except Exception as e:
      print('\nGoogle Search Trend retrieval failed.')
      print(e)
      return
    
    trends = trends.resample('D')
    trends = trends.reset_index(level=0)
    trends = trends.rename(columns={'date':'ds',term:'freq'})
    trends['freq'] = trends['freq'].interpolate()
        
    return trends
  
  # Analysis chagepoint based on google trends
  def changepoint_analysis(self,model,trends):
    
    stock_history = self.stock[self.stock['ds'] > (self.max_date - pd.DateOffset(years = self.years)).date()]
    
    cp = model.changepoints
    # Create dataframe of only changepoints
    change_indices = []
    for changepoint in (cp):
      change_indices.append(self.stock[self.stock['ds'] == changepoint.date()].index[0])
  
    c_data = self.stock.ix[change_indices, :]
    deltas = model.params['delta'][0]
    c_data['delta'] = deltas
    c_data['abs_delta'] = abs(c_data['delta'])
    # Sort the values by maximum change
    c_data = c_data.sort_values(by='abs_delta', ascending=False)
#     print('\nChangepoints sorted by slope rate of change (2nd derivative):\n')
#     print(c_data.ix[:, ['Date', 'Adj. Close', 'delta']][:5])
    
    plt.style.use('seaborn-notebook')
    data_merge = pd.merge(stock_history, trends, on = 'ds', how = 'inner')
    # Normalize values
    data_merge['y_norm'] = data_merge['y'] / max(data_merge['y'])
    data_merge['freq_norm'] = data_merge['freq'] / max(data_merge['freq'])
    plt.plot(data_merge['ds'], data_merge['y_norm'], 'k-', label = 'Stock Price')
    plt.plot(data_merge['ds'], data_merge['freq_norm'], color='forestgreen', label = 'Search Frequency')

    c_data_top10 = c_data[:10]

    plt.vlines(c_data_top10['ds'].dt.to_pydatetime(), ymin=0, ymax=1,
                   linewidth = 1.2, label='Changepoints', linestyles='dashed', color = 'r')
       
    # Plot formatting
    plt.legend(prop={'size': 10})
    plt.xlabel('Date'); plt.ylabel('Normalized Values'); plt.title('Stock Price and Search Frequency for the term %s' %self.term)
    plt.show()