#!/usr/bin/env python
# coding: utf-8

# # Evaluating language knowledge of ELL students from grades 8-12

# ## Import Libraries

# In[24]:


import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings 
warnings.filterwarnings('ignore')


# ## Loading The Dataset

# *Using Pandas Library, we’ll load the CSV file. Named it with ellTrainData for the dataset.*

# In[43]:


ellTrainData = pd.read_csv('input/train.csv')


# In[44]:


ellTrainData.head()


# ## Data Profiling & Cleaning

# *Get the number of columns and rows*

# In[45]:


ellTrainData.shape


# In[46]:


ellTrainData.info()


# *From the info, we know that there are 3911 entries and 8 columns.*

# In[47]:


ellTrainData.isnull().sum()


# *There are no null entries.*

# In[48]:


ellTrainData.describe()


# In[49]:


ellTrainData['full_text'] = ellTrainData['full_text'].replace(r'[^A-Za-z0-9.,!?\'\"]+', ' ', regex=True)


# In[56]:


num = 0
for i in range(3911):
    num += ellTrainData['full_text'][i].count('\"')
num


# ## Exploratory Analysis and Visualization

# In[50]:


ellTrainData['text_len'] = ellTrainData['full_text'].apply(lambda x: len(x))
ellTrainData['words_num'] = ellTrainData['full_text'].apply(lambda x: len(x.split()))


# In[51]:


ellTrainData


# In[52]:


# Length of full_text and words num
fig, ax = plt.subplots(2, 2, figsize=(15, 8))
sns.boxplot(ellTrainData['text_len'], palette='PRGn', ax = ax[0, 0])
sns.distplot(ellTrainData['text_len'], ax = ax[1, 0])
sns.boxplot(ellTrainData['words_num'], palette='PRGn', ax = ax[0, 1])
sns.distplot(ellTrainData['words_num'], ax = ax[1, 1])


# In[ ]:




