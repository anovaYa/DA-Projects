#!/usr/bin/env python
# coding: utf-8

# # Evaluating language knowledge of ELL students from grades 8-12

# ## Import Libraries

# In[96]:


import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings 
warnings.filterwarnings('ignore')

import re
from wordcloud import WordCloud
from wordcloud import STOPWORDS

import nltk
nltk.download('wordnet')
from textblob import TextBlob
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report


# ## Loading The Dataset

# *Using Pandas Library, we’ll load the CSV file. Named it with ellTrainData for the dataset.*

# In[111]:


ellTrainData = pd.read_csv('input/train.csv')


# In[112]:


ellTrainData.head()


# ## Data Profiling & Cleaning

# *Get the number of columns and rows*

# In[113]:


ellTrainData.shape


# In[114]:


ellTrainData.info()


# *From the info, we know that there are 3911 entries and 8 columns.*

# In[115]:


ellTrainData.isnull().sum()


# *There are no null entries.*

# In[116]:


ellTrainData.describe()


# In[117]:


# Basic text cleaning function
def remove_noise(text):
    
    # Make lowercase
    text = text.apply(lambda x: " ".join(x.lower() for x in x.split()))
    
    # Remove whitespaces
    text = text.apply(lambda x: " ".join(x.strip() for x in x.split()))
    
    # Remove special characters
    text = text.apply(lambda x: "".join([" " if ord(i) < 32 or ord(i) > 126 else i for i in x]))
    
    # Remove punctuation
    text = text.str.replace('[^\w\s]', '')
    
    # Remove numbers
    text = text.str.replace('\d+', '')
    
    # Remove Stopwords
    text = text.apply(lambda x: ' '.join([word for word in x.split() if word not in (STOPWORDS)]))
    
    # Convert to string
    text = text.astype(str)
        
    return text


# In[118]:


# Applying noise removal function to data
ellTrainData['filtered_text'] = remove_noise(ellTrainData['full_text'])
ellTrainData.head()


# In[119]:


# Defining a sentiment analyser function
def sentiment_analyser(text):
    return text.apply(lambda Text: pd.Series(TextBlob(Text).sentiment.polarity))

# Applying function to reviews
ellTrainData['polarity'] = sentiment_analyser(ellTrainData['filtered_text'])
ellTrainData.head()


# ## Lexicon Normalisation

# In[120]:


# Instantiate the Word tokenizer & Word lemmatizer
w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()

# Define a word lemmatizer function
def lemmatize_text(text):
    return [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)]

# Apply the word lemmatizer function to data
ellTrainData['filtered_text'] = ellTrainData['filtered_text'].apply(lemmatize_text)
ellTrainData.head()


# ## Exploratory Analysis and Visualization

# In[121]:


ellTrainData['text_len'] = ellTrainData['full_text'].apply(lambda x: len(x))
ellTrainData['words_num'] = ellTrainData['full_text'].apply(lambda x: len(x.split()))


# In[122]:


ellTrainData.head()


# In[123]:


# Length of full_text and words num
fig, ax = plt.subplots(2, 2, figsize=(15, 8))
sns.boxplot(ellTrainData['text_len'], palette='PRGn', ax = ax[0, 0])
sns.distplot(ellTrainData['text_len'], ax = ax[1, 0])
sns.boxplot(ellTrainData['words_num'], palette='PRGn', ax = ax[0, 1])
sns.distplot(ellTrainData['words_num'], ax = ax[1, 1])


# ## WordCloud

# In[124]:


text = " ".join(j for i in ellTrainData['filtered_text'] for j in i)
stopwords = set(STOPWORDS)
wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)
plt.figure( figsize=(15,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# ## Getting a text matrix

# In[125]:


# Getting a count of words from the documents
# Ngram_range is set to 1,2 - meaning either single or two word combination will be extracted
cvec = CountVectorizer(min_df=.05, max_df=.9, ngram_range=(1,2), tokenizer=lambda x: x, lowercase=False)
cvec.fit(ellTrainData['filtered_text'])


# In[126]:


# Getting the total n-gram count
len(cvec.vocabulary_)


# In[127]:


# Creating the bag-of-words representation
cvec_counts = cvec.transform(ellTrainData['filtered_text'])
print('sparse matrix shape:', cvec_counts.shape)
print('nonzero count:', cvec_counts.nnz)
print('sparsity: %.2f%%' % (100.0 * cvec_counts.nnz / (cvec_counts.shape[0] * cvec_counts.shape[1])))


# In[128]:


# Instantiating the TfidfTransformer
transformer = TfidfTransformer()

# Fitting and transforming n-grams
transformed_weights = transformer.fit_transform(cvec_counts)
transformed_weights


# In[129]:


# Getting a list of all n-grams
transformed_weights = transformed_weights.toarray()
vocab = cvec.get_feature_names()

# Putting weighted n-grams into a DataFrame and computing some summary statistics
model = pd.DataFrame(transformed_weights, columns=vocab)
model['Keyword'] = model.idxmax(axis=1)
model['Max'] = model.max(axis=1)
model['Sum'] = model.drop('Max', axis=1).sum(axis=1)
model.head()


# ### Merging datasets

# In[130]:


# Merging td-idf weight matrix with original DataFrame
model = pd.merge(ellTrainData, model, left_index=True, right_index=True)
model.head()


# In[131]:


# Getting a view of the top 20 occurring words
occ = np.asarray(cvec_counts.sum(axis=0)).ravel().tolist()
counts_df = pd.DataFrame({'Term': cvec.get_feature_names(), 'Occurrences': occ})
counts_df.sort_values(by='Occurrences', ascending=False).head(25)


# In[132]:


# Getting a view of the top 20 weights
weights = np.asarray(transformed_weights.mean(axis=0)).ravel().tolist()
weights_df = pd.DataFrame({'Term': cvec.get_feature_names(), 'Weight': weights})
weights_df.sort_values(by='Weight', ascending=False).head(25)


# In[133]:


# Countplot
scList = ['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']
plt.figure(figsize=(15, 10))
for i, c in enumerate(scList):
    ax = plt.subplot(2, 3, i+1)
    sns.countplot(data = model, x = c)
    ax.set(title = c)
    ax.set(ylabel=None)
plt.show()

# Value counts
model[scList].apply(pd.Series.value_counts)


# In[134]:


# Visualising polarity between scores
for score in scList:
    g = sns.FacetGrid(model, col=score, col_order=[5, 4.5, 4, 3.5, 3, 2.5, 2, 1.5, 1], aspect=.65)
    g = g.map(plt.hist, "polarity", bins=20)


# ## Machine Learning

# In[143]:


# Drop all columns not part of the text matrix
ml_model = model.drop(['text_id', 'full_text', 'filtered_text', 'polarity', 'Keyword', 'Max', 'Sum'], axis=1)


# In[144]:


# Create X & y variables for Machine Learning
X = ml_model.drop(scList, axis=1)
y = ml_model[scList]

# Create a train-test split of these variables
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)


# In[145]:


# Defining a function to fit and predict ML algorithms
def modelRes(mod, model_name, x_train, y_train, x_test, y_test):
    mod.fit(x_train, y_train)
    print(model_name)
    acc = cross_val_score(mod, X_train, y_train, scoring = "accuracy", cv = 5)
    predictions = cross_val_predict(mod, X_train, y_train, cv = 5)
    print("Accuracy:", round(acc.mean(),3))
    cm = confusion_matrix(predictions, y_train)
    print("Confusion Matrix:  \n", cm)
    print("Classification Report \n",classification_report(predictions, y_train))


# In[ ]:




