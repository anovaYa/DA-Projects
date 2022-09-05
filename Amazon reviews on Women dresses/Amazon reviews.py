#!/usr/bin/env python
# coding: utf-8

# # Amazon reviews on Women dresses 
# ### (23K Datapoints)
# 

# ## Importing Libraries

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings 
warnings.filterwarnings('ignore')


# In[153]:


SNS_PALETTE='magma'
sns.set(context='notebook', style='dark', palette=SNS_PALETTE, font='sans-serif', font_scale=1, color_codes=False, rc=None)


# ## Loading Data

# In[345]:


dressesData = pd.read_csv('../input/women_dresses_data.csv')


# In[346]:


dressesData.head()


# In[394]:


dressesData.columns = [i.split(' ')[0] for i in dressesData.columns]


# ### About this file
# 
# * s.no: Index
# 
# * age: Age of the customer.
# 
# * division_name: Division of the Cloth customer has bought
# 
# * department_name: Department of the Cloth
# 
# * class_name: Class of the Cloth in particular.
# 
# * clothing_id: Clothing ID (Unique to a type of product)
# 
# * title: Title customers write on their feedback/review text
# 
# * review_text: Customer's Review Text
# 
# * alikefeedbackcount: Number of other customers who agrees with the given feedback (as their experience is quite the same)
# 
# * rating: Rating or stars they've given to the product
# 
# * recommend_index: Whether they'll recommend someone to buy the product or not (0: NO, 1:YES)
# 
# ***thx https://www.kaggle.com/code/flaviocavalcante/amazon-reviews-eda***

# ## Data Preprocessing

# In[347]:


dressesData.info()


# In[348]:


# unique value 
for i in dressesData.columns:
    print(i, len(dressesData[i].unique()))


# In[349]:


# Nan value
dressesData.isna().sum()


# In[350]:


dressesData = dressesData.dropna(subset=['division_name', 'department_name', 'class_name'])


# In[351]:


dressesData['division_name'] = dressesData['division_name'].astype('category')
dressesData['department_name'] = dressesData['department_name'].astype('category')
dressesData['class_name'] = dressesData['class_name'].astype('category')


# In[352]:


dressesData.describe()


# ## Data Exploration and Analysis

# In[353]:


fig, ax = plt.subplots(1, 2, figsize=(15,5))
sns.boxplot(dressesData['age'], palette=SNS_PALETTE, ax=ax[0])
sns.distplot(dressesData['age'], ax=ax[1])
plt.show()


# In[354]:


plt.figure(figsize=(15, 5))
sns.distplot(dressesData['clothing_id'])
plt.show()


# In[355]:


ax = sns.countplot(dressesData['division_name'], palette=SNS_PALETTE)
ax.set(xlabel='Division of the Cloth',
       ylabel = 'Number of Divisions')
plt.show()


# In[356]:


plt.figure(figsize=(15, 5))
ax = sns.countplot(dressesData['department_name'], palette=SNS_PALETTE)
ax.set(xlabel='Department of the Cloth',
       ylabel = 'Number of Departments')
plt.show()


# In[357]:


plt.figure(figsize=(20, 15))
ax = sns.countplot(data = dressesData, y ='class_name', palette=SNS_PALETTE, order = dressesData['class_name'].value_counts().index)

w, h = plt.gcf().get_size_inches()
total = float(len(dressesData['class_name']))
for p in ax.patches:
    percentage =  '{:.1f}%'.format(100 * p.get_width()/total)
    x = p.get_width()+fig.dpi*w/10
    y = p.get_y() + p.get_height()*h/25
    ax.annotate(percentage, (x, y),ha='center')

ax.set(xlabel='Class of the Cloth',
       ylabel = 'Count')
ax.set(xticklabels=[]) 
ax.set(xlabel=None)
ax.tick_params(bottom=False)
plt.show()


# In[358]:


minCountVal = dressesData['class_name'].value_counts().idxmin()
dressesData.drop(dressesData.index[dressesData['class_name'] == minCountVal], inplace = True)
# FIXME remove cat minCountVal


# In[371]:


xColumnsName = ['division_name', 'department_name']
plt.figure(figsize=(15, 5))
for i, c in enumerate(xColumnsName):
    ax = plt.subplot(1, 2, i+1)
    sns.countplot(data = dressesData, x = c, hue ='rating')
    ax.set(title = c)
    ax.set(ylabel=None)
plt.show()


# In[372]:


plt.figure(figsize=(25, 5))
sns.boxplot(data = dressesData, x='class_name', y='rating')


# In[377]:


plt.figure(figsize=(25, 10))
sns.boxplot(data=dressesData, y='age', x='class_name')


# In[403]:


sns.countplot(data = dressesData, x='recommend_index', hue='rating')

