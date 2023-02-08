#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score,recall_score,f1_score


# In[2]:


df = pd.read_csv('Churn_Modelling.csv')


# In[3]:


#display 5 rows of the dataset
df.head()


# In[4]:


#check 5 last rows
#chech total data
df.tail()


# In[5]:


#find shape(rows and columns)shape is not a method it is an attribute
df.shape


# In[6]:


#get data types
df.info()


# In[7]:


#check null values
df.isnull() .sum()


# In[8]:


#get statistics of the dataframe
df.describe()


# In[9]:


#drop irrelevant features
df.columns


# In[10]:


df=df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)


# In[11]:


df.head()


# In[12]:


#ecoding of categorical data
df['Geography'].unique()


# In[13]:


df = pd.get_dummies(df,drop_first=True)


# In[14]:


df.head()


# In[15]:


#not handling imblanced
df['Exited'].value_counts()


# In[16]:


sns.countplot(df['Exited'])


# In[17]:


x =  df.drop('Exited',axis=1)
y = df['Exited']
y


# In[18]:


#splitting dataset
#import from skilearn
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=42,stratify=y)


# In[19]:


#feature scaling(if not scaled features with high range start dominating)
sc=StandardScaler()


# In[20]:


x_train=sc.fit_transform(x_train)
x_test=sc.fit_transform(x_test)


# In[21]:


x_train


# In[22]:


#logistic regression
log= LogisticRegression() #(not best for imblanced data)


# In[23]:


log.fit(x_train,y_train)


# In[24]:


y_pred1 = log.predict(x_test)


# In[25]:


accuracy_score(y_test,y_pred1)


# In[28]:


precision_score(y_test,y_pred1)


# In[30]:


recall_score(y_test,y_pred1)


# In[31]:


f1_score(y_test,y_pred1)


# In[ ]:




