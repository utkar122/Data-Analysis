#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import power_transform
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


# In[2]:


train = pd.read_csv(r"C:\Users\dixit\Desktop\Credit Card Lead Prediction\train_s3TEQDk.csv")
test = pd.read_csv(r"C:\Users\dixit\Desktop\Credit Card Lead Prediction\test_mSzZ8RL.csv")


# In[3]:


train.head()


# In[4]:


test.head()


# # EDA

# In[5]:


train.Is_Lead.value_counts()


# In[6]:


train.groupby(['Gender', 'Is_Lead']).size().unstack().plot(kind='bar', figsize=(12,6))


# In[7]:


train.groupby(['Occupation', 'Is_Lead']).size().unstack().plot(kind='bar', figsize=(12,6))


# In[8]:


train.groupby(['Channel_Code', 'Is_Lead']).size().unstack().plot(kind='bar', figsize=(12,6))


# In[9]:


train.groupby(['Vintage', 'Is_Lead']).size().unstack().plot(kind='bar', figsize=(12,6))


# In[10]:


train.groupby(['Credit_Product', 'Is_Lead']).size().unstack().plot(kind='bar', figsize=(12,6))


# In[11]:


train.groupby(['Is_Active', 'Is_Lead']).size().unstack().plot(kind='bar', figsize=(12,6))


# In[12]:


train_data=train
test_data = test


# # Removing irrevelant coloumn

# In[13]:


train = train.drop(columns = ['ID','Region_Code'], axis = 0)
test = test.drop(columns = ['ID','Region_Code'],axis = 0)


# # Creating a dummy variable

# In[14]:


train = pd.get_dummies(train, columns = ['Gender','Occupation','Credit_Product','Channel_Code','Is_Active'],prefix = '',prefix_sep = '')
test = pd.get_dummies(test, columns = ['Gender','Occupation','Credit_Product','Channel_Code','Is_Active'],prefix = '',prefix_sep = '')


# In[15]:


train.head()


# In[16]:


test.head()


# In[10]:


train.boxplot(column = ['Avg_Account_Balance'])
plt.show()


# In[12]:


def remove_outliers(col):
    Q1,Q3 = col.quantile([0.25,0.75])
    IQR = Q3-Q1
    lower_range = Q1-(1.5*IQR)
    upper_range = Q3+(1.5*IQR)
    return upper_range,lower_range


# In[13]:


upper_bound,lower_bound = remove_outliers(train['Avg_Account_Balance'])
train['Avg_Account_Balance'] = np.where(train['Avg_Account_Balance']>=upper_bound,upper_bound,train['Avg_Account_Balance'])
train['Avg_Account_Balance'] = np.where(train['Avg_Account_Balance']<=lower_bound,lower_bound,train['Avg_Account_Balance'])


# In[14]:


train.boxplot(column = ['Avg_Account_Balance'])
plt.show()


# In[17]:


test.boxplot(column = ['Avg_Account_Balance'])
plt.show()


# In[18]:


upper_bound,lower_bound = remove_outliers(test['Avg_Account_Balance'])
test['Avg_Account_Balance'] = np.where(test['Avg_Account_Balance']>=upper_bound,upper_bound,test['Avg_Account_Balance'])
test['Avg_Account_Balance'] = np.where(test['Avg_Account_Balance']<=lower_bound,lower_bound,test['Avg_Account_Balance'])


# In[19]:


test.boxplot(column = ['Avg_Account_Balance'])
plt.show()


# # Droping a target variable from train data

# In[17]:


X = train.drop(labels=['Is_Lead'], axis=1)
y = train['Is_Lead'].values


# # Data pre processing using standard scaler

# In[18]:


from sklearn import preprocessing
ss = preprocessing.StandardScaler()


# In[19]:


X = ss.fit_transform(X)
test = ss.transform(test)


# # Spliting data into train and cross validate

# In[20]:


from sklearn.model_selection import train_test_split
X_train, X_cv, y_train, y_cv = train_test_split(X, y, test_size=0.10, random_state=101)


# In[21]:


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# In[22]:


from sklearn.metrics import roc_auc_score


# # Model Training

# # LGMRegressor

# In[25]:


from lightgbm import LGBMRegressor


# In[26]:


lgbm = LGBMRegressor(boosting_type='gbdt', num_leaves=100, max_depth=31, learning_rate=0.01, n_estimators=1000, min_child_samples=20, subsample=0.80)


# In[27]:


lgbm.fit(X_train, y_train, verbose=0)


# In[28]:


y_pred_lgbm = lgbm.predict(X_cv)
print("roc_auc_score: ", roc_auc_score(y_cv, y_pred_lgbm))


# In[29]:


y_pred_lgbm1 = lgbm.predict(test)


# In[30]:


y_pred_lgbm1


# In[31]:


submission = pd.DataFrame({
        "ID": test_data['ID'],
        "Is_Lead": y_pred_lgbm1
    })
submission.to_csv('submit2.csv', index=False)
print(submission)

