import pandas as pd
import numpy as np
import os

from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
#loading data
#be aware of escape characters in directory path
os.chdir(r"C:\programming\kaggle\titanic problem")
df=pd.read_csv("train.csv")
df2=pd.read_csv("test.csv")

#wrangling data
df=df.drop(axis=1,labels=['PassengerId','Name','Ticket','Cabin'])

# filling missing values
df['Embarked']=df['Embarked'].fillna(method='ffill',axis=0)
df['Age']=df['Age'].fillna(df.Age.median())
df=pd.get_dummies(df,columns=['Sex','Embarked'])


#creating input and output variables
features=df.ix[0:,1:].values
target=df['Survived'].values





'''
df.Sex=df['Sex'][df.Sex=="male"]=0
df.Sex=df['Sex'][df.Sex=="female"]=1

df['Embarked']=df['Embarked'][df['Embarked']=='S']=0
df['Embarked']=df['Embarked'][df['Embarked']=='C']=1
df['Embarked']=df['Embarked'][df['Embarked']=='Q']=2
'''

#print(df.columns)
#print(df)
print(features)



#model
max_depth=10
min_samples_split=5
my_tree=tree.DecisionTreeClassifier(max_depth = max_depth, min_samples_split = min_samples_split, random_state = 1)
my_tree=my_tree.fit(features,target)
print(my_tree.score(features, target))


my_tree2=RandomForestClassifier(max_depth = max_depth, min_samples_split = 2,n_estimators = 100, random_state = 1)
my_tree2=my_tree2.fit(features,target)
print(my_tree2.score(features, target))

passangerID=df2['PassengerId'].values

df2=df2.drop(axis=1,labels=['PassengerId','Name','Ticket','Cabin'])

df2=pd.get_dummies(df2,columns=['Sex','Embarked'])

df2['Age']=df2['Age'].fillna(df2['Age'].median())
df2['Fare']=df2['Fare'].fillna(df2['Fare'].median())

features2=df2.ix[0:,0:].values
print(features2)

#target2=df2['Survived'].values
prediction=my_tree2.predict(features2)

solution=pd.DataFrame(prediction,passangerID,columns=['Survived'])    
solution.to_csv("solution1.csv",index_label='PassengerId')
print(prediction)










