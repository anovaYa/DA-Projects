#!/usr/bin/env python
# coding: utf-8

# <a href="https://www.kaggle.com/code/anovayana/online-retail-ii-uci-analytics?scriptVersionId=101245814" target="_blank"><img align="left" alt="Kaggle" title="Open in Kaggle" src="https://kaggle.com/static/images/open-in-kaggle.svg"></a>

# # Online Retail II UCI

# ## Loading the required libraries

# In[1]:


import warnings 
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import seaborn as sns
from datetime import datetime as dt, timedelta
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Load the data

# In[8]:


# data = pd.read_csv('../input/online-retail-ii-uci/online_retail_II.csv', on_bad_lines = 'skip', parse_dates=['InvoiceDate'])
data = pd.read_csv('../input/online_retail_II.csv', on_bad_lines = 'skip', parse_dates=['InvoiceDate'])


# ## Attribute Information:
# * InvoiceNo: Invoice number. Nominal. A 6-digit integral number uniquely assigned to each transaction. If this code starts with the letter 'c', it indicates a cancellation.
# * StockCode: Product (item) code. Nominal. A 5-digit integral number uniquely assigned to each distinct product.
# * Description: Product (item) name. Nominal.
# * Quantity: The quantities of each product (item) per transaction. Numeric.
# * InvoiceDate: Invice date and time. Numeric. The day and time when a transaction was generated.
# * UnitPrice: Unit price. Numeric. Product price per unit in sterling (Â£).
# * CustomerID: Customer number. Nominal. A 5-digit integral number uniquely assigned to each customer.
# * Country: Country name. Nominal. The name of the country where a customer resides.

# ## Checking the data

# In[3]:


data.head()


# In[4]:


data.info()


# Four of the variables are 'object' while three are numerical, and one is dt.

# In[5]:


data.describe(include='all')


# From the output, we can infer that the median (50%) Price is 2,1 and the median Quantity is 3, when the average Price is 4.65 and the average Quantity is 9.94. There is a difference between the mean and the median values of these variables, which is because of the distribution of the data. 
# Also, we can infer that the min Price and Quantity are negative. 

# In[6]:


plt.figure(figsize=(15,7))
plt.scatter(data['Price'], data['Quantity'])
plt.xlabel('Price')
plt.ylabel('Quantity')
plt.grid(True)
plt.colorbar()


# In[7]:


fig, ax = plt.subplots(2, 1)
sns.boxplot(data['Price'], palette='PRGn', ax = ax[0])
sns.distplot(data['Price'], ax = ax[1])


# In[8]:


fig, ax = plt.subplots(2, 1)
sns.boxplot(data['Quantity'], palette='PRGn', ax = ax[0])
sns.distplot(data['Quantity'], ax = ax[1])


# ## Data cleaning

# In[9]:


data = data[(data['Price']>0) & (data['Price'] < 20000)]
data = data[(data['Quantity']>0) & (data['Quantity']< 20000)]


# In[10]:


fig, ax = plt.subplots(2, 1)
sns.boxplot(data['Quantity'][data['Quantity']<100], palette='PRGn', ax = ax[0])
sns.distplot(data['Quantity'][data['Quantity']<100], ax = ax[1])


# In[11]:


fig, ax = plt.subplots(2, 1)
sns.boxplot(data['Price'][data['Price']<30], palette='PRGn', ax = ax[0])
sns.distplot(data['Price'][data['Price']<30], ax = ax[1])


# In[12]:


fig,ax = plt.subplots(1, 2, figsize=(15, 10))
fig.suptitle('Histograms')
sns.distplot(data['Price'][data['Price']<30], ax=ax[0])
sns.distplot(data['Quantity'][data['Quantity']<100], ax=ax[1])


# In[13]:


data['Total'] = data['Quantity'] * data['Price']


# In[14]:


#Customers with Max Total Purchase Amount
data.groupby(['Customer ID', 'Country'], as_index=False)['Total'].agg('sum').sort_values('Total', ascending=False).head()


# In[15]:


#Countries With Max Total Purchase Amount
data.groupby(['Country'], as_index=False)['Total'].agg('sum').sort_values('Total', ascending=False).head()


# # RFM

# In[16]:


data = data.groupby('Customer ID', as_index=False).agg({'Total':'mean', 'InvoiceDate':'max', 'Invoice':'nunique'})


# In[17]:


obs_date = (max(data['InvoiceDate']) + timedelta(days=1))
data['days_since_lats_purch'] = data['InvoiceDate'].apply(lambda x: obs_date - x)
data['days_since_lats_purch'] = data['days_since_lats_purch'].dt.days.astype(int)


# In[18]:


sns.distplot(data['days_since_lats_purch'])


# # # recent

# In[19]:


def recent_score(r):
    if r <= 70:
        return 3
    elif r > 70 and r <= 500:
        return 2
    else:
        return 1

data['recent'] = data['days_since_lats_purch'].apply(recent_score)
sns.countplot(data['recent'])


# In[20]:


sns.distplot(data[data['Invoice']<25]['Invoice'])


# # # frequency

# In[21]:


def frequency_score(r):
    if r <= 3:
        return 3
    elif r > 3 and r <= 7:
        return 2
    else:
        return 1

data['frequency'] = data['Invoice'].apply(frequency_score)
sns.countplot(data['frequency'])


# In[22]:


sns.distplot(data[data['Total']<100]['Total'])


# # # monetary

# In[23]:


def monetary_score(r):
    if r <= 30:
        return 3
    elif r > 30 and r <= 60:
        return 2
    else:
        return 1

data['monetary'] = data['Total'].apply(monetary_score)
sns.countplot(data['monetary'])


# In[24]:


data['rfm'] = data.apply(lambda x: str(x['recent']) + str(x['frequency']) + str(x['monetary']), axis=1)
data.head()


# In[25]:


plt.figure(figsize=(15, 10))
sns.countplot(y=data['rfm'])

