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

# In[8]:


scaler = StandardScaler()
standardData = scaler.fit(creditCardData)
scaledData = pd.DataFrame(scaler.transform(creditCardData),columns= creditCardData.columns )


# In[9]:


scaledData.describe()


# In[10]:


scaledData.hist(figsize=(20, 20))


# In[11]:


pca = PCA(n_components = 5)
XPCAreduced = pca.fit_transform(scaledData)


# In[12]:


print('Mean vector: ', pca.mean_)
print('Projection: ', pca.components_)
print ('Explained variance ratio: ', pca.explained_variance_ratio_)


# In[13]:


plt.scatter(XPCAreduced[:, 1],  XPCAreduced[:, 2], c=XPCAreduced[:, 0], cmap='hot')


# In[15]:


# Principal components correlation coefficients
loadings = pca.components_
 
# Number of features before PCA
n_features = pca.n_features_
 
# Feature names before PCA
feature_names = scaledData.columns
 
# PC names
pc_list = [f'PC{i}' for i in list(range(1, n_features + 1))]
 
# Match PC names to loadings
pc_loadings = dict(zip(pc_list, loadings))
 
# Matrix of corr coefs between feature names and PCs
loadings_df = pd.DataFrame.from_dict(pc_loadings)
loadings_df['feature_names'] = feature_names
loadings_df = loadings_df.set_index('feature_names')
loadings_df


# In[16]:


fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(loadings[0], loadings[1],loadings[2])


# In[31]:


fig = plt.figure(figsize=(10,15))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(loadings[1], loadings[2], loadings[0], c='red', marker='v', s=50)
ax.scatter(XPCAreduced[:, 1], XPCAreduced[:, 2],XPCAreduced[:, 0], alpha=0.5)
ax.set_title("A 3D projection of data")


# In[100]:


#get correlation matrix plot for loadings
sns.heatmap(loadings_df, annot=True, cmap='Spectral')


# In[52]:


import matplotlib.pyplot as plt 
import numpy as np
 
# Get the loadings of x and y axes
xs = loadings[0]
ys = loadings[1]
 
# Plot the loadings on a scatterplot
for i, varnames in enumerate(feature_names):
    plt.scatter(xs[i], ys[i], s=200)
    plt.arrow(
        0, 0, # coordinates of arrow base
        xs[i], # length of the arrow along x
        ys[i], # length of the arrow along y
        color='r', 
        head_width=0.01
        )
    plt.text(xs[i], ys[i], varnames)
 
# Define the axes
xticks = np.linspace(-0.8, 0.8, num=5)
yticks = np.linspace(-0.8, 0.8, num=5)
plt.xticks(xticks)
plt.yticks(yticks)
plt.xlabel('PC1')
plt.ylabel('PC2')
 
# Show plot
plt.title('2D Loading plot with vectors')
plt.show()


# # Clustering

# In[110]:


model = KMeans(n_clusters=3)
scaledData['cluster_label'] = model.fit_predict(scaledData)


# In[111]:


scaledData


# In[122]:


plt.scatter(XPCAreduced[:, 0], XPCAreduced[:, 1], c=scaledData['cluster_label'], cmap='hot')


# In[125]:


fig = plt.figure(figsize=(10,15))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(XPCAreduced[:, 0],XPCAreduced[:, 1],scaledData['cluster_label'])


# In[ ]:




