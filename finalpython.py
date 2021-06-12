# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 15:49:36 2019

@author: Administrator
"""

import pandas as pd
import numpy as np
#Reading the file
loan_credit = pd.read_csv('C:\\Users\\Administrator\\Desktop\\Tools\\Project\\Python\\XYZCorp_LendingData.txt', sep = '\t',na_values = 'NaN',low_memory = False)

#understanding the data
loan_credit.describe()
print(loan_credit.head(4))
print("The rows and columns of the dataset is : {} " .format(loan_credit.shape)) #Fetching the number of rows and columns
loan_credit.apply(lambda x: sum(x.isnull()),axis=0) #checking to see if which column has how many null values

#filtering data
del_columns= ['member_id','pymnt_plan','title','emp_title','sub_grade','addr_state','earliest_cr_line','zip_code','desc','last_pymnt_d','next_pymnt_d','last_credit_pull_d','collections_12_mths_ex_med','mths_since_last_major_derog','policy_code','application_type','annual_inc_joint','dti_joint','verification_status_joint','open_acc_6m','open_il_6m','open_il_12m','open_il_24m','mths_since_rcnt_il','total_bal_il','il_util','open_rv_12m','open_rv_24m','inq_fi','max_bal_bc','all_util','total_cu_tl','inq_last_12m','mths_since_last_record','mths_since_last_delinq']

data1 = loan_credit.drop(labels = del_columns,axis = 1)
data1.apply(lambda x: sum(x.isnull()),axis=0) 

#missing value treatment
data1['total_rev_hi_lim'].describe()
data1['total_rev_hi_lim'].unique()

data1['tot_cur_bal'].describe()
data1['tot_cur_bal'].unique()

data1['tot_coll_amt'].describe()
data1['tot_coll_amt'].unique()

data1['revol_util'].describe()
data1['revol_util'].unique()

data1['emp_length'].describe()
data1['emp_length'].unique()

#replacing numerical values with mean
mean_data = data1[['total_rev_hi_lim','tot_cur_bal','tot_coll_amt','revol_util']].mean()

data1[['total_rev_hi_lim','tot_cur_bal','tot_coll_amt','revol_util']]= data1[['total_rev_hi_lim','tot_cur_bal','tot_coll_amt','revol_util']].fillna(mean_data)

data1.apply(lambda x: sum(x.isnull()),axis=0) 

#converting categorical data into numerical data
data1['emp_length'].unique()
data1['term'].unique()
data1['grade'].unique()
data1['home_ownership'].unique() 
data1['verification_status'].unique()
data1['purpose'].unique()

data1['emp_length'] = data1["emp_length"].replace({'years':'','year':'',' ':'','<':'','\+':''}, regex = True)
data1['emp_length'] = pd.to_numeric(data1['emp_length'],errors='coerce')
mean_elen = data1['emp_length'].mean()
data1['emp_length']=data1['emp_length'].fillna(mean_elen)
data1.apply(lambda x: sum(x.isnull()),axis=0)

data1['term'] = data1['term'].replace({'months':'',' ':''},regex = True)
data1['term'] = pd.to_numeric(data1['term'],errors='coerce')
data1['term'] = data1['term'].astype(int)
data1.dtypes[data1.dtypes != 'int64']

data1['grade'] = data1['grade'].map({'A':7,'B':6,'C':5,'D':4,'E':3,'F':2,'G':1})

data1['home_ownership'] = data1['home_ownership'].map({'MORTGAGE':6,'RENT':5,'OWN':4,'OTHER':3,'NONE':2,'ANY':1})

data1['verification_status'] = data1['verification_status'].map({'Verified':3,'Source Verified':2,'Not Verified':1})

data1['purpose'] = data1['purpose'].map({'credit_card':1, 'car':2, 'small_business':3, 'other':4, 'wedding':5, 'debt_consolidation':6, 'home_improvement':7, 'major_purchase':8, 'medical':9, 'moving':10, 'vacation':11, 'house':12, 'renewable_energy':13,'educational':14})

data1['initial_list_status'] = data1['initial_list_status'].map({'f':1, 'w':2})

#Splitting the data
data2 = data1

data2['str_split'] = data2.issue_d.str.split('-')
data2['issue'] = data2.str_split.str.get(0) 
data2['d'] = data2.str_split.str.get(1)

data2['issue'].unique()
data2['d'].unique()

data2['issue'] = data2['issue'].replace({'Jan':'01','Feb':'02','Mar':'03','Apr':'04','May':'05','Jun':'06','Jul':'07','Aug':'08','Sep':'09','Oct':'10','Nov':'11','Dec':'12'} ,regex = True)

print(data2['issue']) 
print(data2['d']) 

data2['period'] = data2['d'].map(str) + data2['issue'] #Concat the columns issue and d for creating a column for splitting the data into Train and test set
data2['period'].unique()
data2 = data2.sort_values('period') #Sorting the data on the basis of period column

final_data = data2
more_column = ['str_split','issue','d','issue_d']
final_data = final_data.drop(labels = more_column, axis = 1)

final_data = final_data.set_index('period') #Setting the period column as index for splitting purpose


Train_data = final_data.loc['200706':'201505',:]  #Creating the Training data set using index column to differentiate
Train_data.index.unique()

Test_data = final_data.loc['201506':'201512',:]  #Creating the Training data set using index column to differentiate
Test_data.index.unique()

Train_data.apply(lambda x: sum(x.isnull()),axis=0)
Test_data.apply(lambda x: sum(x.isnull()),axis=0)

#Splitting the output variable into another dataframe
train_target = pd.DataFrame(Train_data['default_ind'])
train_target = train_target.astype(int)
test_target = pd.DataFrame(Test_data['default_ind'])
test_target = test_target.astype(int)

final_train = Train_data.iloc[:,0:37]
final_test = Test_data.iloc[:,0:37]
final_train.dtypes[final_train.dtypes != 'int64']

#Scaling the variables
from sklearn.preprocessing import StandardScaler

scaler_train = StandardScaler()
scaler_test = StandardScaler()

scaler_train.fit(final_train)
scaler_test.fit(final_test)

#Logisitc Regression --------------  99.97 % ----------------------------------

from sklearn.linear_model import LogisticRegression

#Create a model

classifier=(LogisticRegression())

#Training the model
classifier.fit(final_train,train_target)

#predicting the values using logisitic Regression model
Y_pred=classifier.predict(final_test)

print(list(zip(test_target['default_ind'].values,Y_pred)))

#Comparing the results and checking the accuracy

from sklearn.metrics import confusion_matrix,accuracy_score,classification_report

print(confusion_matrix(test_target['default_ind'].values,Y_pred))
print(accuracy_score(test_target['default_ind'].values,Y_pred))
print(classification_report(test_target['default_ind'].values,Y_pred)) 


#Getting the ROC curve
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score


y_pred_prob = classifier.predict_proba(final_test)[:,1]
fpr,tpr,threshold  = roc_curve(test_target,y_pred_prob)
plt.xlabel('Fpr')
plt.ylabel('Tpr')
plt.plot([0,1],[0,1],'k--')
plt.plot(fpr,tpr,label = 'logistic regression')

roc_auc_score(test_target,y_pred_prob)


#Running Decision Tree Model------- 100 % -------------------------
#predicting using the DecisionTreeClassifier

from sklearn.tree import DecisionTreeClassifier

model_DecisionTree=DecisionTreeClassifier()

model_DecisionTree.fit(final_train,train_target)

#fit the model on the data and predict the values
Y_dtree = model_DecisionTree.predict(final_test)

#print(Y_pred)

print(list(zip(test_target['default_ind'].values,Y_dtree)))

#confusion matrix

print(confusion_matrix(test_target['default_ind'].values,Y_dtree))

print(accuracy_score(test_target['default_ind'].values,Y_dtree))

print(classification_report(test_target['default_ind'].values,Y_dtree)) 


#running Random Forest Model--------------- 100 % ---------------------------------------

#predicting using the Random_Forest_Classifier
from sklearn.ensemble import RandomForestClassifier

model_RandomForest=RandomForestClassifier(20) #no of tress to be made

#fit the model on the data and predict the values

model_RandomForest.fit(final_train,train_target)

Y_rfm=model_RandomForest.predict(final_test)

print(list(zip(test_target['default_ind'].values,Y_pred)))


print(confusion_matrix(test_target['default_ind'].values,Y_rfm))
print(accuracy_score(test_target['default_ind'].values,Y_rfm))
print(classification_report(test_target['default_ind'].values,Y_rfm)) 

#Knn algorithm--------------- 99.8 %--------------------------------------

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(final_train,train_target)

y_knn = knn.predict(final_test)


print(confusion_matrix(test_target['default_ind'].values,y_knn))
print(accuracy_score(test_target['default_ind'].values,y_knn))
print(classification_report(test_target['default_ind'].values,y_knn))

#Plotting the ROC curve

knn_predict = knn.predict_proba(final_test)[:,1]
fpr,tpr,threshold = roc_curve(test_target,knn_predict)
plt.xlabel('Fpr')
plt.ylabel('Tpr')
plt.plot([0,1],[0,1],'k--')
plt.plot(fpr,tpr,label = 'knn')

roc_auc_score(test_target,knn_predict)


#Ensemble Modelling-----------------------------------------------------------------------

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier

#Creating the sub models

estimators=[]
model1=LogisticRegression()
estimators.append(('log',model1))

model2=DecisionTreeClassifier()
estimators.append(('cart',model2))

model3=SVC()
estimators.append(('svm',model3))


#Create the ensemble model
ensemble=VotingClassifier(estimators) #Voting Classifier refers that the method used for ensemble model is Voting and not Mean or Average weighted mean 

ensemble.fit(final_train,train_target)

Y_ensemble=ensemble.predict(final_test)

#Confusion Matrix
print(confusion_matrix(test_target['default_ind'].values,y_knn))
print(accuracy_score(test_target['default_ind'].values,y_knn))
print(classification_report(test_target['default_ind'].values,y_knn))