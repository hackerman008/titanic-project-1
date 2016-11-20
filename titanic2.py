import os
import pandas as pd
import numpy as np

#splitting module

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn import cross_validation as cval
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
import sklearn.metrics as metrics


#load data
def load_data():
    os.chdir(r'C:\programming\kaggle\titanic problem')
    df=pd.read_csv('train.csv')
    return df

def create_column_title(df):
    df['Title']='Mr.'
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
    return df

def create_column_deck(df):
    df['Deck']='U'
    df['Cabins_owned']=0
    for i in range(0,891):
        if pd.isnull(df.ix[i,'Cabin']) :
            df.ix[i,'Deck']='Unknown' 
        else :   
            df.ix[i,'Deck']=df.ix[i,'Cabin'][0]
            df.ix[i,'Cabins_owned']=len(df.ix[i,'Cabin'].split())
    
    return df
    
def create_column_totalfamilysize(df):
    df['total_family_size']=df['SibSp']+df['Parch']          
    return df
   
   
def wrangling_data(df):
    #wrangling data
    df=df.drop(axis=1,labels=['PassengerId','Name','Ticket','Cabin'])
    df['Embarked']=df['Embarked'].fillna(method='ffill')
    df['Age']=df['Age'].fillna(df.Age.median())
    
    #creating dummy columns
    df=pd.get_dummies(df,columns=['Sex','Embarked','Title','Deck'])
      
    return df

'''
#adaboost
ada=AdaBoostClassifier(DecisionTreeClassifier(),
                       
                       learning_rate=1,n_estimators = 50,random_state=1)
ada=ada.fit(X_train,y_train)
prediction_adaboost=ada.predict(X_test) 


#print accuracy
print("adaboost score=",ada.score(X_train, y_train))

#print(prediction)
print("accuracy score for adaboost = ",metrics.accuracy_score(y_test,prediction_adaboost))
print("precision score for adaboost = ",metrics.precision_score(y_test,prediction_adaboost))
print("recall score for adaboost = ",metrics.recall_score(y_test,prediction_adaboost))
print("f1 score for adaboost = ",metrics.f1_score(y_test,prediction_adaboost))

crossval_score_adaboost=cval.cross_val_score(ada,X_train,y_train,cv=10,scoring='f1')

print("cross validation for ada =\n",cval.cross_val_score(ada,X_train,y_train,cv=10,scoring='f1'))
print("adaboost mean cross validation = ",crossval_score_adaboost.mean())
#print("feature importance adaboost =\n",ada.feature_importances_)
cm=confusion_matrix(y_test,prediction_adaboost)
print(cm)
'''

'''
fpr,tpr,thresholds=metrics.roc_curve(y_test,prediction_adaboost,pos_label=1)
print("fpr=",fpr)
print("tpr=",tpr)
'''


def cross_validation_test(X_train,X_test,y_train,y_test):
 
    print("\n")
    
    #declaring the RandomForestClassifier with parameters
    mytree=RandomForestClassifier(max_depth=6,min_samples_split=25,
                                  n_estimators=110,random_state = 2)
   
    #fitting the model    
    mytree=mytree.fit(X_train,y_train)
    prediction_random_forest=mytree.predict(X_test)

    #training  accuracy
    print("training set score=",mytree.score(X_train, y_train))

    #metrrics of prediction
    print("Classification accuracy= ",
          metrics.accuracy_score(y_test,prediction_random_forest))
    print("precision = ",
          metrics.precision_score(y_test,prediction_random_forest))
    print("recall = ",
          metrics.recall_score(y_test,prediction_random_forest))
    print("f1 score= ",
          metrics.f1_score(y_test,prediction_random_forest))
    
    #cross-validation
    crossval_score_random_forest=cval.cross_val_score(mytree,X_train,y_train,cv=10)
    
    print("cross validation using accuracy metric=\n",
          cval.cross_val_score(mytree,X_train,y_train,cv=10))
    print("mean cross validation score = ",
          crossval_score_random_forest.mean(),'\n')

    #print("feature importance randomforest =\n",mytree.feature_importances_)
    #confusing matrix     
    cm=confusion_matrix(y_test,prediction_random_forest)
    print('Confusion matrix=\n',cm)



if __name__ == '__main__':
    #loading data    
    df=load_data()
    
    #feature engineering
    df=create_column_title(df)
    df=create_column_deck(df)
    df=create_column_totalfamilysize(df)
    
    #cleaning data    
    df=wrangling_data(df)
        
    #creating input and output variables
    X=df.ix[0:,1:].values
    y=df.ix[0:,[0]].values
    
    #y=np.asmatrix(y)
    #converting the output variable to a 1D array
    y=y.ravel()
    
    #splitting the data into training and testing set
    X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=1)
    
    cross_validation_test(X_train,X_test,y_train,y_test)