# Stock Prediction

I perform a time series prediction using [Prophet](https://facebook.github.io/prophet/docs/quick_start.html) and [LSTM](https://en.wikipedia.org/wiki/Long_short-term_memory) model to predict the stock price of an individual company. For this example, I will predict the stock price of JPMorgan Chase & Co.

The Facebook Prophet package was released in 2017 for Python and R. Prophet is designed for analyzing time series with daily observations that display patterns on different time scales. It also has advanced capabilities for modeling the effects of holidays on a time-series and implementing custom changepoints, here we will stick to the basic functions to get a model up.

Recently, LSTM models are used to do time series prediction. This is based on the gated architecture of LSTM’s that has an ability to manipulate its memory state, they are ideal for time series problems. LSTM models are pretty good at extracting patterns in input feature space, where the input data spans over long sequence. Further, LSTMs can almost seamlessly model problems with multiple input variables. This adds a great benefit in time series forecasting, where classical linear methods can be difficult to adapt to multivariate or multiple input forecasting problems.

The model Stock() is a class in [Prophet.py](https://github.com/mutouyu1124/time-series-example/blob/master/Prophet.py), which implements prediction by Prophet model. The model Stock_LSTM() is another class in [LSTM.py](https://github.com/mutouyu1124/time-series-example/blob/master/LSTM.py), which implements prediction by LSTM model. We have a demo of the model's prediction on stock price of JPMorgan Chase & Co in the stock_price_demo.ipynb. 

## Data

Usually, about 80% of the time spent on a data science project is getting and cleaning data. Thanks to the [Quandl financial library](https://www.quandl.com/tools/python),which can be installed with pip from the command line, lets you access thousands of financial indicators with a single line of Python. Quandl automatically puts our data into a pandas dataframe, the data structure of choice for data science. For example, if you want to access the stock price of JPMorgan Chase & Co, just replace the 'JMP' with the stock ticker.


