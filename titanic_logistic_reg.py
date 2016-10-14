import os
import pandas as pd
import numpy as np
import sklearn.metrics as metrics
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation as cval
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix


#load data
os.chdir(r'C:\programming\kaggle\titanic problem')
df=pd.read_csv('train2.csv')

#creating new columns title
df['Title']='Mr.'
name=df['Name']
for i in range(0,891):
    if 'Mr.' or 'Master.' or 'Mrs.' or 'Miss.' in df['Name'][i]:
        if 'Mr.' in df['Name'][i]:
            df.ix[i,'Title']='Mr.'
        elif 'Master.' in df['Name'][i]:
            df.ix[i,'Title']='Master.'
        elif 'Mrs.' in df['Name'][i]:
            df.ix[i,'Title']='Mrs.'
        elif 'Miss.' in df['Name'][i]:
            df.ix[i,'Title']='Miss.'


df=df.drop(axis=1,labels=['PassengerId','Name','Ticket','Cabin'])   #dropping unwanted columns

#filling missing values
df['Embarked']=df['Embarked'].fillna(method='ffill')
df['Age']=df['Age'].fillna(df.Age.median())

df=pd.get_dummies(df,columns=['Sex','Embarked','Title'])    #creating dummy variables for string features

#creating input and output variables
X=df.ix[0:,1:].values
y=df.ix[0:,[0]].values

#y=np.asmatrix(y)
y=y.ravel()     #converting the output variable to a 1D array

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=1)      #splitting data into train and test

#train and fit the LogisticRegression model
model = LogisticRegression(max_iter=400,C=1)
model=model.fit(X_train,y_train)

prediction=model.predict(X_test)    #making prediction

#print accuracy on train data
print("accuracy score on training data=",model.score(X_train, y_train))

#evaluation metrics on test data
print("accuracy score for lg = ",metrics.accuracy_score(y_test,prediction))
print("precision score for adaboost = ",metrics.precision_score(y_test,prediction))
print("recall score for adaboost = ",metrics.recall_score(y_test,prediction))
print("f1 score for adaboost = ",metrics.f1_score(y_test,prediction))

crossval_score_adaboost=cval.cross_val_score(model,X_train,y_train,cv=10,scoring='f1')      #crossvalidation

print("cross validation =\n",cval.cross_val_score(model,X_train,y_train,cv=10,scoring='f1'))
print("adaboost mean cross validation = ",crossval_score_adaboost.mean())   #mean cross validation score


########## TO LOAD THE TEST DATA AND PREDICT THE OUTPUT ########

df2=pd.read_csv("test.csv")     #test data


df2['Title']='Mr.'
name=df2['Name']
for i in range(0,418):
    if 'Mr.' or 'Master.' or 'Mrs.' or 'Miss.' in df2['Name'][i]:
        if 'Mr.' in df2['Name'][i]:
            df2.ix[i,'Title']='Mr.'
        elif 'Master.' in df2['Name'][i]:
            df2.ix[i,'Title']='Master.'
        elif 'Mrs.' in df2['Name'][i]:
            df2.ix[i,'Title']='Mrs.'
        elif 'Miss.' in df2['Name'][i]:
            df2.ix[i,'Title']='Miss.'

passangerID=df2['PassengerId'].values      #store passangerid to use in creating csv submission.
df2=df2.drop(axis=1,labels=['PassengerId','Name','Ticket','Cabin'])     #dropping useless features from the test data

#filling in missing values
df2['Age']=df2['Age'].fillna(df2['Age'].median())
df2['Fare']=df2['Fare'].fillna(df2['Fare'].median())

df2=pd.get_dummies(df2,columns=['Sex','Embarked','Title'])      #converting to dummy variables.

features2=df2.ix[0:,0:].values      #storing input and output variables

prediction=model.predict(features2)  #making prediction for test data

solution=pd.DataFrame(prediction,passangerID,columns=['Survived'])
solution.to_csv("solution2.csv",index_label='PassengerId')
