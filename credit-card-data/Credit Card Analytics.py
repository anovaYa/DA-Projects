#!/usr/bin/env python
# coding: utf-8

# # Credit Card Customer Data
# A Customer Credit Card Information Dataset which can be used for Identifying Loyal Customers, Customer Segmentation, Targeted Marketing and other such use cases in the Marketing Industry.
# 

# ## Importing Libraries

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings 
warnings.filterwarnings('ignore')


# # Loading Data

# In[2]:


#/kaggle/input/credit-card-customer-data/Credit Card Customer Data.csv
creditCardData = pd.read_csv('../input/credit_card_customer_data.csv')


# In[3]:


creditCardData.head()


# In[4]:


creditCardData.info()


# In[5]:


creditCardData = creditCardData.loc[:, creditCardData.columns!='Sl_No'].set_index('Customer Key')


# In[6]:


creditCardData.describe()


# In[7]:


creditCardData.hist(figsize=(20, 20))


# # Data Preprocessing
# ## Standardization

# In[92]:


scaler = StandardScaler()
# Standardizing the features
standardData = scaler.fit(creditCardData)
scaledData = pd.DataFrame(scaler.transform(creditCardData),columns= creditCardData.columns )


# In[93]:


scaledData.describe()


# # Principal Component Analysis (PCA)

# In[94]:


pca = PCA(n_components = 3)
principalComponents = pca.fit_transform(scaledData)
principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2', 'principal component 3'])


# In[95]:


print('Mean vector: ', pca.mean_)
print('Projection: ', pca.components_)
print ('Explained variance ratio: ', pca.explained_variance_ratio_)


# In[96]:


plt.scatter(principalComponents[:, 1],  principalComponents[:, 2], c=principalComponents[:, 0], cmap='hot')


# In[97]:


# Principal components correlation coefficients
loadings = pca.components_
 
# Number of features before PCA
n_features = pca.n_features_
 
# Feature names before PCA
featureNames = scaledData.columns
 
# PC names
principalComponentsList = [f'PC{i}' for i in list(range(1, n_features + 1))]
 
# Match PC names to loadings
principalComponentsLoadings = dict(zip(principalComponentsList, loadings))
 
# Matrix of corr coefs between feature names and PCs
loadings_df = pd.DataFrame.from_dict(principalComponentsLoadings)
loadings_df['feature_names'] = featureNames
loadings_df = loadings_df.set_index('feature_names')
loadings_df


# In[116]:


fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(loadings[1], loadings[2],loadings[0])
ax.set_xlabel('PC3')
ax.set_ylabel('PC2')
ax.set_zlabel('PC1')


# In[117]:


fig = plt.figure(figsize=(10,15))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(loadings[1], loadings[2], loadings[0], c='red', marker='v', s=50)
ax.scatter(principalComponents[:, 1], principalComponents[:, 2], principalComponents[:, 0], alpha=0.5)
ax.set_title("A 3D projection of data")
ax.set_xlabel('PC2')
ax.set_ylabel('PC3')
ax.set_zlabel('PC1')


# In[100]:


#get correlation matrix plot for loadings
sns.heatmap(loadings_df, annot=True, cmap='Spectral')


# # Clustering

# In[108]:


model = KMeans(n_clusters=3)
scaledData['cluster_label'] = model.fit_predict(scaledData)


# In[109]:


scaledData


# In[136]:


plt.scatter(principalComponents[:, 0], principalComponents[:, 1], c=scaledData['cluster_label'], cmap='viridis')
plt.xlabel('PC1')
plt.ylabel('PC2')


# In[124]:


fig = plt.figure(figsize=(10,15))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(principalComponents[:, 0], principalComponents[:, 1],  scaledData['cluster_label'], alpha=0.3)
ax.scatter(loadings[0], loadings[1], c='red', marker='v', s=50)
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('cluster')
ax.set_zticks([0, 1, 2])

