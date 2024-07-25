#!/usr/bin/env python
# coding: utf-8

# In[1]:


print('hello world')


# In[2]:


#importing liberies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[3]:


#import dataset
df = pd.read_csv('Titanic-Dataset.csv')
y = df.pop("Survived")


# In[4]:


y.head()


# In[5]:


df.tail()


# In[6]:


df.info()


# In[47]:


import seaborn as sns



# In[7]:


df.describe()


# In[8]:


# get just the mumeric varibales by selecting only the varibales that not 'object ' datatypes
numeric_varibels = list(df.dtypes[df.dtypes != "object"].index)
df[numeric_varibels].head()


# In[ ]:





# In[9]:


df['Age'].fillna(df.Age.mean(), inplace = True)
df.describe()


# passenger id looks lide worthless 

# In[10]:


#import the model
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score



# In[11]:


model = RandomForestRegressor(n_estimators = 100 , oob_score=True, random_state = 42)
model.fit(df[numeric_varibels] , y)


# In[12]:


model.oob_score_


# In[15]:


y_oob = model.oob_prediction_
print("c-stats"), roc_auc_score(y,y_oob)


# In[22]:


#smaple function to show descriptive stats on catgorical varible
def describe_catagorical(df):
    " just like .describe(), but returns the results for categorical variables only"
    if df.empty or not df.columns.any():
        print("The DataFrame is empty or has no columns.")
    else:
        from IPython.display import display, HTML
        display(HTML(df[df.columns[df.dtypes == "object"]].describe().to_html()))
    


# In[23]:


describe_catagorical(df)


# In[24]:


df.drop (["Name","Ticket","PassengerId"], axis = 1, inplace = True)


# In[28]:


def clean_cabin(df):
    try:
        return df[0]
    except TypeError:
        return "None"
    
df["Cabin"]= df.Cabin.apply(clean_cabin)


# In[35]:


categorical_variables = ['Sex','Cabin','Embarked']

for variable  in categorical_variables:
    df[variable].fillna("Missing", inplace=True)
    dummies = pd.get_dummies(df[variable], prefix=variable)
    
    df= pd.concat([df, dummies],axis=1)
    df.drop([variable], axis=1, inplace=True)


# In[34]:


def printall(df, max_rows=10):
    from IPython.display import display, HTML
    display(HTML(df.to_html(max_rows=max_rows)))
    
printall(df)


# In[37]:


model = RandomForestRegressor(100, oob_score= True, n_jobs= -1, random_state =42)
model.fit(df, y)
print("C-stats"), roc_auc_score(y,model.oob_prediction_)


# In[38]:


#important featurs

model.feature_importances_


# In[45]:


feature_importances = pd.Series(model.feature_importances_, index= df.columns)
feature_importances.sort_values()
feature_importances.plot(kind="bar",figsize=(7,6));


# In[53]:


df['Age'] = df['Age'].astype('category')

# Create a count plot
sns.countplot(data=df, x='Age')
plt.xlabel('Age')
plt.ylabel('Count')
plt.title('Distribution of Age')
plt.show()


# In[58]:


df['Sex_male']= df['Sex_male'].astype('category')

# Create a count plot
sns.countplot(data=df, x='Sex_male')
plt.xlabel('Sex')
plt.ylabel('Count')
plt.title('Distribution of Sex')
plt.show()


# In[ ]:




