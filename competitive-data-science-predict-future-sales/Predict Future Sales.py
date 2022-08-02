#!/usr/bin/env python
# coding: utf-8

# # Final project for "How to win a data science competition" Coursera course

# ## Importing Libraries

# In[96]:


import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings 
warnings.filterwarnings('ignore')

import random
random.seed(0)

STEPS = 26


# ## Loading Data

# ### File descriptions
# * sales_train.csv - the training set. Daily historical data from January 2013 to October 2015.
# * test.csv - the test set. You need to forecast the sales for these shops and products for November 2015.
# * sample_submission.csv - a sample submission file in the correct format.
# * items.csv - supplemental information about the items/products.
# * item_categories.csv  - supplemental information about the items categories.
# * shops.csv- supplemental information about the shops.

# In[2]:


trainData = pd.read_csv('./sales_train.csv', parse_dates=['date'])


# ### Data fields
# * ID - an Id that represents a (Shop, Item) tuple within the test set
# * shop_id - unique identifier of a shop
# * item_id - unique identifier of a product
# * item_category_id - unique identifier of item category
# * item_cnt_day - number of products sold. You are predicting a monthly amount of this measure
# * item_price - current price of an item
# * date - date in format dd/mm/yyyy
# * date_block_num - a consecutive month number, used for convenience. January 2013 is 0, February 2013 is 1,..., October 2015 is 33
# * item_name - name of item
# * shop_name - name of shop
# * item_category_name - name of item category

# In[3]:


trainData.info()


# In[4]:


trainData.describe()


# In[5]:


trainData.hist(figsize=(20, 20))


# ## Data Preprocessing

# In[113]:


trainData = trainData[(trainData['item_price'] > 0) & (trainData['item_price'] < 10000)]
trainData = trainData[(trainData['item_cnt_day']>0) & (trainData['item_cnt_day'] < 1000)]


# In[115]:


fig, ax = plt.subplots(2, 2, figsize=(15, 8))
sns.boxplot(trainData['item_price'], palette='PRGn', ax = ax[0, 0])
sns.distplot(trainData['item_price'], ax = ax[1, 0])
sns.boxplot(trainData['item_cnt_day'], palette='PRGn', ax = ax[0, 1])
sns.distplot(trainData['item_cnt_day'], ax = ax[1, 1])


# ## Time Series Analysis

# ### Rolling mean and standard deviation graphs

# In[116]:


from statsmodels.tsa.stattools import adfuller
def test_stationarity(df, period = 12, dft=False):
    movingAVG = df.rolling(window=period).mean()
    movingSTD = df.rolling(window=period).std()
    #plot
    plt.figure(figsize=(15, 10))
    df.plot(label='Original', alpha=0.3)
    mean = plt.plot(movingAVG, label='Rolling Mean')
    std = plt.plot(movingSTD, label='Rolling  Standard Deviation', alpha=0.5)
    plt.title('Rolling Mean & Standard Deviation')
    plt.legend()
    if dft:
        print('Results of Dickey-Fuller Test:')
        dftest = adfuller(df, autolag='AIC')
        dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
        for key, value in dftest[4].items():
            dfoutput['\nCritical Value (%s)' % key] = value
        print(dfoutput)


# In[117]:


test_stationarity(trainData['item_cnt_day'], period = 30)


# ### Per month

# In[110]:


trainData['Month'] = trainData['date'].dt.to_period('M')
trainData


# In[111]:


trainDataPerMonth = trainData.groupby(['Month']).agg({'item_cnt_day' : 'sum'})
trainDataPerMonth.reset_index(inplace=True)
trainDataPerMonth = trainDataPerMonth.set_index('Month')
trainDataPerMonth.rename(columns = {'item_cnt_day':'item_cnt_month'}, inplace = True)
trainDataPerMonth.plot()


# ### Decomposition Type and data transformation

# In[97]:


#import the required modules for TimeSeries data generation:
import statsmodels.api as sm
def tsdisplay(y, figsize = (15, 10), title = "", lags = 12):
    tmp_data = y
    fig = plt.figure(figsize = figsize)
    #Plot the time series
    tmp_data.plot(ax = fig.add_subplot(311), title = "$Time\ Series\ " + title + "$", legend = False)
    #Plot the ACF:
    sm.graphics.tsa.plot_acf(tmp_data, lags = lags, zero = False, ax = fig.add_subplot(323))
    plt.xticks(np.arange(1,  lags + 1, 1.0))
    #Plot the PACF:
    sm.graphics.tsa.plot_pacf(tmp_data, lags = lags, zero = False, ax = fig.add_subplot(324))
    plt.xticks(np.arange(1,  lags + 1, 1.0))
    #Plot the QQ plot of the data:
    sm.qqplot(tmp_data, line='s', ax = fig.add_subplot(325)) 
    plt.title("QQ Plot")
    #Plot the residual histogram:
    fig.add_subplot(326).hist(tmp_data, bins = 40)
    plt.title("Histogram")
    #Fix the layout of the plots:
    plt.tight_layout()
    plt.show()


# In[94]:


#Import the required modules for model estimation:
import statsmodels.tsa as smt
def decomposition(df):
    log_passengers = np.log(df)
    decomposition_1  = smt.seasonal.seasonal_decompose(log_passengers, model = "additive", freq = 12, two_sided = True)
    fig, ax = plt.subplots(4, 1, figsize=(15, 8))
    # Plot the series
    decomposition_1.observed.plot(ax = ax[0])
    decomposition_1.trend.plot(ax = ax[1])
    decomposition_1.seasonal.plot(ax = ax[2])
    decomposition_1.resid.plot(ax = ax[3])
    # Add the labels to the Y-axis
    ax[0].set_ylabel('Observed')
    ax[1].set_ylabel('Trend')
    ax[2].set_ylabel('Seasonal')
    ax[3].set_ylabel('Residual')
    # Fix layout
    plt.tight_layout()
    plt.show()


# In[80]:


decomposition(trainDataPerMonth)


# In[89]:


test_stationarity(trainDataPerMonth['item_cnt_month'], 6, True)


# The p-value is greater than the critical value of 0.05. The series is not stationary

# ## Forecasting

# In[82]:


log_passengers = np.log(trainDataPerMonth)
trainDataPerMonthShift = log_passengers - log_passengers.shift()
trainDataPerMonthShift.dropna(inplace=True)
trainDataPerMonthShift.plot()


# In[88]:


test_stationarity(trainDataPerMonthShift['item_cnt_month'], 6, True)


# In[98]:


tsdisplay(trainDataPerMonthShift)


# ### Train Test Split

# In[99]:


from sklearn.linear_model import LinearRegression

train = trainDataPerMonthShift[:STEPS]
test = trainDataPerMonthShift[STEPS:]


print(f"Train dates : {train.index.min()} --- {train.index.max()}  (n={len(train)})")
print(f"Test dates  : {test.index.min()} --- {test.index.max()}  (n={len(test)})")

fig, ax = plt.subplots(figsize=(9, 4))
train['item_cnt_month'].plot(ax=ax, label='train')
test['item_cnt_month'].plot(ax=ax, label='test')
ax.legend()


# In[100]:


from statsmodels.tsa.arima_model import ARIMA

model = ARIMA(train, order=(1,1,0))
results_ARIMA = model.fit(disp=-1)
train.plot()
plt.plot(results_ARIMA.fittedvalues, color='red')
print('RSS: %.4f' %sum((results_ARIMA.fittedvalues-train['item_cnt_month'])[1:]**2))
print("plotting ARIMA model")


# In[101]:


results_ARIMA.plot_predict(1,40)


# In[102]:


x = results_ARIMA.forecast(steps=9)[0]
x 


# In[103]:



from sklearn.metrics import mean_squared_error
mean_squared_error(x,test['item_cnt_month'])


# In[104]:


pred = pd.Series(x, index = test.index)


# In[105]:


fig, ax = plt.subplots(figsize=(9, 4))
train['item_cnt_month'].plot(ax=ax, label='train')
test['item_cnt_month'].plot(ax=ax, label='test')
pred.plot(label='pred')
ax.legend()


# New

# In[129]:


trainDataPerMonthDiff = trainDataPerMonth - trainDataPerMonth.shift()
trainDataPerMonthDiff = trainDataPerMonthDiff.dropna()
test_stationarity(trainDataPerMonthDiff['item_cnt_month'], 12, True)


# In[138]:


import pmdarima as pm
from pmdarima.arima import auto_arima
model = auto_arima(
    y=trainDataPerMonthDiff[:STEPS],
    seasonal=True,
    start_p = 1, max_p =5,
    start_q =1, max_q =5,
    d = None,
    start_P = 1, max_P =5,
    start_Q =1, max_Q =5,
    D = None,
    m=12,)


# In[139]:


model.summary()


# In[167]:


prediction, confint = model.predict(n_periods=12, return_conf_int=True)
confint_df = pd.DataFrame(confint)
prediction


# In[168]:


period_index = pd.period_range(
    start = trainDataPerMonthDiff[:STEPS].index[-1],
    periods = 12,
    freq='M'
)
predicted_df = pd.DataFrame({'value':prediction}, index=period_index)
predicted_df


# In[171]:


plt.figure(figsize=(10, 8))
plt.plot(trainDataPerMonthDiff[:STEPS].to_timestamp(), label='Actual data')
plt.plot(predicted_df.to_timestamp(), color='orange', label='Predicted data')
plt.plot(trainDataPerMonthDiff[STEPS:], label='Test data')
plt.fill_between(period_index.to_timestamp(), confint_df[0], confint_df[1],color='grey',alpha=.2, label='Confidence Intervals Area')
plt.legend()
plt.show()


# In[ ]:




