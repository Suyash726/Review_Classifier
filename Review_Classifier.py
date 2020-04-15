#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[2]:


df = pd.read_csv('Desktop/Natural_Language_Processing/Restaurant_Reviews.tsv',delimiter='\t')


# In[3]:


print(df)


# In[4]:


# IMporting required nltk modules
import re # It has the tools to clean the text effeciently
import nltk
nltk.download('stopwords')# Stopwords are the words that are irrelevant.Ex :- the,is ,are
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


# In[5]:


# We will now apply the cleaning process to all the elements in the review column
corp = []
for i in range(0,1000):
    rev = re.sub('[^a-zA-Z]',' ',df['Review'][i]).lower()
    # Get rid of the useless words
    rev = rev.split()
    ps = PorterStemmer()
    rev = [ps.stem (i) for i in rev if not i in set(stopwords.words('english'))]
    # Joining back the elements
    rev = ' '.join(rev)
    corp.append(rev)


# In[6]:


# The Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corp).toarray()
y = df.iloc[:,1].values


# In[7]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)


# In[8]:


from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train,y_train)

pred = classifier.predict(X_test)

from sklearn.metrics import accuracy_score,confusion_matrix
CB = confusion_matrix(y_test,pred)

print(CB)


# In[9]:


print(accuracy_score(y_test,pred))


# In[ ]:





# In[ ]:




