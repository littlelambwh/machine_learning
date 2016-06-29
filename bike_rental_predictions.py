
# coding: utf-8

# In[1]:

import pandas as pd

bike_rentals=pd.read_csv("bike_rental_hour.csv")


# In[5]:

bike_rentals.head(5)


# In[6]:

get_ipython().magic(u'matplotlib inline')

import matplotlib.pyplot as plt

plt.hist(bike_rentals["cnt"])


# In[7]:


bike_rentals.corr()["cnt"]


# In[18]:

def assign_labels(hour):
    if hour > 6 and hour <=12:
        sign=1
    elif hour >12 and hour <=18:
        sign=2
    elif hour >18 and hour <=24:
        sign=3
    elif hour >=0 and hour <=6:
        sign=4
    
    return sign


# In[19]:

bike_rentals["time_label"]=bike_rentals['hr'].apply(assign_labels)


# In[20]:

bike_rentals.head()


# In[21]:

### choose error metric 


train = bike_rentals.sample(frac=.8)

test = bike_rentals.loc[~bike_rentals.index.isin(train.index)]



# In[22]:

predictor=["hr","temp","hum","atemp"]
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
target=train["cnt"]
reg.fit(train[predictor],target)


# In[23]:

predictions=reg.predict(test[predictor])


# In[24]:

import numpy as np 
error=np.mean((predictions-test['cnt'])**2)

print error 


# In[25]:

predictors = list(train.columns)
predictors.remove("cnt")
predictors.remove("casual")
predictors.remove("registered")
predictors.remove("dteday")


# In[26]:

reg=LinearRegression()
target=train["cnt"]
reg.fit(train[predictors],target)


# In[27]:

predictions=reg.predict(test[predictors])


# In[28]:

error=np.mean((predictions-test['cnt'])**2)


# In[29]:

error


# In[46]:

##### decision tree 

from sklearn.tree import DecisionTreeRegressor 
tree1=DecisionTreeRegressor(min_samples_leaf=10)
tree1.fit(train[predictors],target)



# In[47]:

predictions_tree=tree1.predict(test[predictors])


# In[48]:

error=np.mean((predictions_tree-test['cnt'])**2)

print error 


# In[50]:

#####random forest

from sklearn.ensemble import RandomForestRegressor


# In[51]:

tree2=RandomForestRegressor(min_samples_leaf=2)


# In[52]:

tree2.fit(train[predictors],target)


# In[53]:

predictions_tree2=tree2.predict(test[predictors])


# In[56]:

error=np.mean((predictions_tree2-test['cnt'])**2)


# In[57]:

error


# In[ ]:



