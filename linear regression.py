#!/usr/bin/env python
# coding: utf-8

# In[19]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# In[20]:


df = pd.read_csv("insurance.csv")


# In[21]:


df.head()


# In[4]:


df.shape


# In[5]:


df.info()


# In[6]:


df.columns


# In[7]:


df.describe()


# In[8]:


plt.figure(figsize=(6,6))
sns.countplot(x='sex',data=df)
plt.title('Gender graph')
plt.show()


# In[9]:


plt.figure(figsize=(6,6))
sns.countplot(x='smoker',data=df)
plt.title('smoker')
plt.show()


# In[10]:


plt.figure(figsize=(6,6))
sns.countplot(x='region', data=df)
plt.title('Region')
plt.show()


# In[11]:


plt.figure(figsize=(6,6))
sns.barplot(x='region', y='charges', data=df)
plt.title('Cost vs Region')
plt.show()


# In[12]:


plt.figure(figsize=(6,6))
sns.barplot(x='sex', y='charges',hue='smoker', data=df)
plt.title('smokers charges')
plt.show()


# In[ ]:





# In[13]:


df[['age','bmi','children','charges']].hist(bins=30, figsize=(10,10), color='green')
plt.show()


# In[14]:


df.head()


# In[15]:


#convert strings to integers
df['sex'] = df['sex'].apply({'male':0, 'female':1}.get)
df['smoker'] = df['smoker'].apply({'yes':1, 'no':0}.get)
df['region'] = df['region'].apply({'southwest':1, 'southeast':2, 'northwest':3, 'northeast':4}.get)


# In[16]:


df.head()


# In[17]:


#show correlation using mapheat
plt.figure(figsize=(10,7))
sns.heatmap(df.corr(), annot = True)
plt.show()


# In[18]:


X = df.drop(['charges', 'sex'], axis=1)
y = df.charges


# In[22]:


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)
print("X_train shape: ", X_train.shape)
print("X_test shape: ", X_test.shape)
print("y_train shpae: ", y_train.shape)
print("y_test shape: ", y_test.shape)


# In[23]:


#creating a linear regression model
linreg = LinearRegression()


# In[24]:


#create variableto store value
linreg.fit(X_train, y_train)
pred = linreg.predict(X_test)


# In[25]:


#create r2 score
from sklearn.metrics import r2_score


# In[26]:


# a higher r2 value indicates a better fit
print("R2 score: ",(r2_score(y_test, pred)))


# In[27]:


#create graph of actual versus predicted value
plt.scatter(y_test, pred)
plt.xlabel('Y test')
plt.ylabel('Y pred')
plt.show()


# In[28]:


#predict for customer
data = {'age':50, 'bmi':25, 'children':2, 'smoker':1, 'region':2}
index = [0]
cust_df = pd.DataFrame(data, index)
cust_df


# In[29]:


cost_pred = linreg.predict(cust_df)
print("The medical insurance cost of the new customer is: ", cost_pred)


# In[ ]:




