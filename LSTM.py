import quandl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

from keras.models import Sequential
from keras.layers import LSTM,Dense
from sklearn.preprocessing import MinMaxScaler

class Stock_LSTM(object):
  # initialize the stock data
  def __init__(self,ticker):
    
    ticker = ticker.upper()
    
    self.symbol = ticker
    
    quandl.ApiConfig.api_key = 'saU3Kye_asAxvpXoxUC_'
    
    try:
      stock = quandl.get('WIKI/%s'%ticker)
    except Exception as e:
      print('Error Retrieving Data')
      print(e)
      return
    
    stock = stock.reset_index(level=0)
    
    stock['ds'] = stock['Date']
    stock['y'] = stock['Adj. Close']
    
    self.stock = stock
    
    self.min_date = np.min(stock['ds'])
    self.max_date = np.max(stock['ds'])
    
    self.lb = 7
    
    print('{} Stocker initialized, Data covers from {} to {}'.format(self.symbol,self.min_date.date(),self.max_date.date()))

  def plot(self,start_date = None, end_date = None):
    
    if not start_date:
      start_date = self.min_date
    if not end_date:
      end_date = self.max_date
      
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
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
    
  def create_LSTM_model(self,units = 64):
    
    model = Sequential()
    model.add(LSTM(units,input_shape = (self.lb,1)))
    model.add(Dense(1))
    
    model.compile(optimizer = 'adam', loss = 'mse')
    return model
  
  def __processData(self,data):
    
    X,y = [],[]
    
    for i in range(len(data) - self.lb -1):
      X.append(data[i:(i+self.lb),0])
      y.append(data[(i+self.lb),0])
      
    return np.array(X),np.array(y)
    
    
  def train_LSTM_model(self,model,epochs = 300):
    stock_history = self.stock[self.stock['ds'] > (self.max_date - pd.DateOffset(years = 4)).date()]
    
    data = stock_history['y'].values
    
    scl = MinMaxScaler()
    data = data.reshape(data.shape[0],1)
    data = scl.fit_transform(data)
    
    X, y = self.__processData(data)
    
    X_train, X_test = X[:int(X.shape[0]*0.75)], X[int(X.shape[0]*0.75):]
    y_train, y_test = y[:int(y.shape[0]*0.75)], y[int(y.shape[0]*0.75):]
    
    X_train = X_train.reshape((X_train.shape[0],X_train.shape[1],1))
    X_test = X_test.reshape((X_test.shape[0],X_test.shape[1],1))
    
    print("Start training LSTM model...")
    history = model.fit(X_train,y_train,epochs = epochs,verbose = 0, validation_data = (X_test,y_test), shuffle = False)
    print("Training process is done.")
    
    plt.style.use('seaborn-notebook')
    plt.figure(figsize = (16,10))
    plt.subplot(2,2,1)
    plt.plot(history.history['loss'], label = "Training Loss")
    plt.plot(history.history['val_loss'], label = "Validation_Loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title("Loss function")
    plt.legend(loc='best')
    
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    train_pred  = scl.inverse_transform(train_pred)
    test_pred = scl.inverse_transform(test_pred)
    
    plt.subplot(2,2,2)
    Xt = model.predict(X_test)
    plt.plot(scl.inverse_transform(y_test.reshape(-1,1)),color='k',label="original series")
    plt.plot(scl.inverse_transform(Xt),color='r',label="predicted series")
    plt.title("Prediction of Stock Price on Test Set")
    plt.legend(loc='best')
    
    plt.subplot(2,2,3)
    plt.plot(scl.inverse_transform(y.reshape(-1,1)), color='k')
    
    split_pt = int(X.shape[0]*0.75) # window_size
    plt.plot(np.arange(0,split_pt,1), train_pred, color='b')
    plt.plot(np.arange(split_pt,split_pt+len(test_pred),1), test_pred, color='r')

    plt.xlabel('day')
    plt.ylabel('Stock Prics($)')
    plt.title('Stock Price of {}'.format(self.symbol))
    plt.legend(['original series','training fit','testing fit'],loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()
    
    train_mse = np.sqrt(np.mean(np.square(train_pred - y_train)))
    test_mse = np.sqrt(np.mean(np.square(train_pred - y_test)))
         
    print("Training rmse error: {}, test rmse error: {}".format(train_mse,test_mse))
    
    return model
  
  def prediction_future(self,model,days = 180):
    stock_history = self.stock[self.stock['ds'] > (self.max_date - pd.DateOffset(years = 4)).date()]
    
    data = stock_history['y'].values
    
    scl = MinMaxScaler()
    data = data.reshape(data.shape[0],1)
    data = scl.fit_transform(data)
    
    x, y = self.__processData(data)
    
    X = data[-self.lb:]
    X = X.reshape((1,X.shape[0],1))
    
    prediction = []
    
    for i in range(days): 
      pred = model.predict(X)
      X = np.concatenate((X, pred), axis=None)
      X = X[1:]
      X = np.reshape(X,(-1, 1))
      X = X.reshape((1,X.shape[0],1))
      pred = scl.inverse_transform(pred.reshape(1,-1))
      prediction = np.concatenate((prediction, pred), axis=None)
      
    
    plt.style.use('seaborn-notebook')
    split_pt = x.shape[0]
    plt.plot(np.arange(0,split_pt,1),scl.inverse_transform(y.reshape(-1,1)), color='k',label="Observations")
    plt.plot(np.arange(split_pt,split_pt+days,1),prediction,color = 'r',label="Predict future")
    plt.title("Predict Future Stock Price by Trained Model")
    plt.xlabel('days')
    plt.ylabel('Stock Price($)')
    plt.legend(loc="best")
