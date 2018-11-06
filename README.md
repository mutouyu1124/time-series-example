# Stock Prediction

I perform a time series prediction using [Prophet](https://facebook.github.io/prophet/docs/quick_start.html) and [LSTM](https://en.wikipedia.org/wiki/Long_short-term_memory) model to predict the stock price of an individual company. For this example, I will predict the stock price of JPMorgan Chase & Co.

The Facebook Prophet package was released in 2017 for Python and R. Prophet is designed for analyzing time series with daily observations that display patterns on different time scales. It also has advanced capabilities for modeling the effects of holidays on a time-series and implementing custom changepoints, here we will stick to the basic functions to get a model up.

Recently, LSTM models are used to do time series prediction. This is based on the gated architecture of LSTM’s that has an ability to manipulate its memory state, they are ideal for time series problems. Further, LSTMs can almost seamlessly model problems with multiple input variables. This adds a great benefit in time series forecasting, where classical linear methods can be difficult to adapt to multivariate or multiple input forecasting problems.

In this project, the model Stock() is a class in [Prophet.py](https://github.com/mutouyu1124/time-series-example/blob/master/Prophet.py), which implements prediction by Prophet model. The model Stock_LSTM() is another class in [LSTM.py](https://github.com/mutouyu1124/time-series-example/blob/master/LSTM.py), which implements prediction by LSTM model. We have a demo of the model's prediction on stock price of JPMorgan Chase & Co in the stock_price_demo.ipynb. 

## Data

Usually, about 80% of the time spent on a data science project is getting and cleaning data. Thanks to the [Quandl financial library](https://www.quandl.com/tools/python),which can be installed with pip from the command line, lets you access thousands of financial indicators with a single line of Python. Quandl automatically puts our data into a pandas dataframe, the data structure of choice for data science. For example, if you want to access the stock price of JPMorgan Chase & Co, just replace the 'JMP' with the stock ticker.

## Installation
Install the prophet library:
```
pip install fbprophet
```
Install quandl library to access stock data:
```
pip install quandl
```
Install pytrends library to access google search trend data:
```
pip install pytrends
```

## Stock() Methods

*  `__init__(self,ticker)` : Initialize and access the stock data.

    Parameter:
    *  string for company ticker, i.e. 'JMP' (required)
    
*  `plot(self,start_date = None, end_date = None)`: Display the stock price 

    Parameter:
    *  start_date: the start date to display stock price. If start_date = None, the default date is the earliest date in records.
    *  end_date: the end date to display stock price. If end_date = None, the default date is the lastest date in records.
    
*  `__create_model(self,**kwargs)`: Initialize the prophet model.
    
    Parameter:
    *  `**kwargs`: other arguments of prophet model.
    
    Return:
    * model: a prophet model without training.

*  `__resampling(self, dataframe)`:  Method to linearly interpolate prices on the weekends

    Parameter:
    *  dataframe: the data need to be interpolated.
    
    Return:
    *  dataframe: the data with interpolated prices on the weekends
    
*  `__remove_weekends(self, dataframe)`: Remove weekends from a dataframe

    Parameter:
    *  dataframe: the data needs to remove weekends.
    
    Return:
    *  dataframe: the data with weekends being removed.
    
*  `create_prophet_model(self, years = 5, resample = False)`: 

    Parameter:
    *  years: int for using last n years as training data.
    *  resample: boolean, optional, default False. When True, the data used to train prophet model with interpolated prices on the weekends.
    
    Return:
    * model: the prophet model after training
    * predictions: the fitted value of training data.
    
    
