#!/usr/bin/env python
# coding: utf-8

# # Evaluating language knowledge of ELL students from grades 8-12

# ## Import Libraries

# In[12]:


import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings 
warnings.filterwarnings('ignore')

import re


# ## Loading The Dataset

# *Using Pandas Library, we’ll load the CSV file. Named it with ellTrainData for the dataset.*

# In[2]:


ellTrainData = pd.read_csv('input/train.csv')


# In[3]:


ellTrainData.head()


# ## Data Profiling & Cleaning

# *Get the number of columns and rows*

# In[4]:


ellTrainData.shape


# In[5]:


ellTrainData.info()


# *From the info, we know that there are 3911 entries and 8 columns.*

# In[6]:


ellTrainData.isnull().sum()


# *There are no null entries.*

# In[7]:


ellTrainData.describe()


# In[8]:


ellTrainData['full_text'] = ellTrainData['full_text'].astype(str)


# In[9]:


ellTrainData['full_text'] = ellTrainData['full_text'].replace(r'[^A-Za-z0-9.,!?\']+', ' ', regex=True)


# In[10]:


def remove_extra_whitespaces_func(text):
    return re.sub(r'^\s*|\s\s*', ' ', text).strip()

def add_whitespaces_func(text):
    return re.sub('(?<![A-Z])([.,-])(?![A-Z]+)', r' \g<1> ', text)


# In[13]:


ellTrainData['full_text'] = ellTrainData['full_text'].apply(add_whitespaces_func)
ellTrainData['full_text'] = ellTrainData['full_text'].apply(remove_extra_whitespaces_func)


# ## Exploratory Analysis and Visualization

# In[15]:


ellTrainData['text_len'] = ellTrainData['full_text'].apply(lambda x: len(x))
ellTrainData['words_num'] = ellTrainData['full_text'].apply(lambda x: len(x.split()))


# In[16]:


ellTrainData.head()


# In[17]:


# Length of full_text and words num
fig, ax = plt.subplots(2, 2, figsize=(15, 8))
sns.boxplot(ellTrainData['text_len'], palette='PRGn', ax = ax[0, 0])
sns.distplot(ellTrainData['text_len'], ax = ax[1, 0])
sns.boxplot(ellTrainData['words_num'], palette='PRGn', ax = ax[0, 1])
sns.distplot(ellTrainData['words_num'], ax = ax[1, 1])


# In[38]:


from wordcloud import WordCloud
from wordcloud import STOPWORDS


# In[39]:


text = " ".join(i for i in ellTrainData['full_text'])
stopwords = set(STOPWORDS)
wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)
plt.figure( figsize=(15,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[80]:


wordsList = [i for i in wordcloud.words_]
def words_score(text, wordsList):
    score = 0
    for i in wordsList:
        score += text.count(i)*wordcloud.words_[i]
        print(text.count(i), wordcloud.words_[i])
    return score


# In[74]:


ellTrainData['words_score'] = ellTrainData['full_text'].apply(words_score, wordsList=wordsList)


# In[78]:


plt.figure(figsize=(15,10))
heatmap = sns.heatmap(ellTrainData.corr(), cmap = "Blues", annot=True, linewidth=3)
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':12}, pad=12)


# In[77]:


ellTrainData['total'] = ellTrainData[['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']].sum(axis=1)


# In[79]:


ellTrainData


# In[ ]:




