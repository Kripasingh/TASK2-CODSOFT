#!/usr/bin/env python
# coding: utf-8

# # TASK-2: MOVIE RATING PREDICTION 

#  IMPORTING LIBRARIES

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


# IMPORTING DATASETS

# In[2]:


df = pd.read_csv("MOVIE Prediction.csv", encoding="latin1")


# In[3]:


df


# In[4]:


df.head()


# In[5]:


df.tail()


# In[6]:


df.isna().any()


# Data Cleaning

# In[7]:


df.isnull().sum()


# In[8]:


df.copy()


# In[9]:


df.info()


# In[10]:


df.duplicated().sum()


# In[11]:


df.dropna(inplace=True)


# In[12]:


df.shape


# In[13]:


df.columns


# Data Pre-Processing

# In[15]:


df['Duration'] = pd.to_numeric(df['Duration'].str.replace('min',''))


# In[16]:


df['Genre'] = df['Genre'].str.split(',')
df = df.explode('Genre')
df['Genre'].fillna(df['Genre'].mode()[0], inplace=True)


# In[17]:


df['Votes'] = pd.to_numeric(df['Votes'].str.replace(',',''))


# In[18]:


df.info()


# Data Visualizing

# In[19]:


year = px.histogram(df,x = 'Year' , histnorm= 'probability density', nbins = 30)
year.show()


# In[20]:


avg_rating_by_year =  df.groupby(['Year' , 'Genre'])['Rating'].mean().reset_index()
top_genres = df['Genre'].value_counts().head(10).index
average_rating_by_year = avg_rating_by_year[avg_rating_by_year['Genre'].isin(top_genres)]
fig = px.line(avg_rating_by_year, x='Year', y='Rating', color="Genre")
fig.update_layout(title='Average Rating by year for Top Genres', xaxis_title='Year', yaxis_title='Average Rating')
fig.show()


# In[21]:


rating_fig = px.histogram(df, x = 'Rating', histnorm='probability density', nbins = 40)
rating_fig.update_layout(title='Distribution of Rating', title_x=0.5, title_pad=dict(t=20), title_font=dict(size=20), xaxis_title='Rating', yaxis_title='probability density')
rating_fig.show()


# Feature Engineering

# In[22]:


from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_absolute_error , mean_squared_error, r2_score


# In[23]:


df.drop('Name', axis = 1, inplace = True)


# In[24]:


genre_mean_rating = df.groupby('Genre')['Rating'].transform('mean')
df['Genre_mean_rating'] = genre_mean_rating
director_mean_rating = df.groupby('Director')['Rating'].transform('mean')
df['Director_mean_rating'] = director_mean_rating
actor1_mean_rating = df.groupby('Actor 1')['Rating'].transform('mean')
df['Actor1_mean_rating'] = actor1_mean_rating
actor2_mean_rating = df.groupby('Actor 2')['Rating'].transform('mean')
df['Actor2_mean_rating'] = actor2_mean_rating
actor3_mean_rating = df.groupby('Actor 3')['Rating'].transform('mean')
df['Actor3_mean_rating'] = actor3_mean_rating


# In[25]:


X= df[['Year', 'Votes','Duration','Genre_mean_rating','Director_mean_rating', 'Actor1_mean_rating', 'Actor2_mean_rating', 'Actor3_mean_rating']]
y= df['Rating']


# In[26]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)


# Model Building

# In[27]:


Model = LinearRegression()
Model.fit(X_train,y_train)
Model_pred = Model.predict(X_test)


# In[28]:


print('The performance evaluation of Logistic Regression is below: ','\n')
print('Mean squared error:',mean_squared_error(y_test,Model_pred))
print('Mean absolute error:',mean_absolute_error(y_test,Model_pred))
print('R2 score:',r2_score(y_test,Model_pred))


# Model Testing

# In[29]:


X.head(5)


# In[30]:


y.head(5)


# In[31]:


data={'Year':[2018], 'Votes':[30],'Duration': [113],'Genre_mean_rating':[5.6],'Director_mean_rating':[4.3], 'Actor1_mean_rating':[5.2], 'Actor2_mean_rating':[4.3], 'Actor3_mean_rating':[4.5]}
trail = pd.DataFrame(data)


# In[32]:


rating_predicted = Model.predict(trail)
print("Predicted Rating:", rating_predicted[0])

