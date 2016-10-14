import pandas as pd
import os

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

#loading data
os.chdir(r"C:\programming\kaggle\titanic problem")
df=pd.read_csv("train2.csv")     #train data

#adding new column Title
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

#adding new column Deck
df['Deck']='U'
for i in range(0,891):
    if pd.isnull(df.ix[i,'Cabin']) :
       df.ix[i,'Deck']='Unknown'
    else :
        df.ix[i,'Deck']=df.ix[i,'Cabin'][0]


df['total_family_size']=df['SibSp']+df['Parch']     #adding new column total_family_size
df=df.drop(axis=1,labels=['PassengerId','Name','Ticket','Cabin'])       #dropping unwanted columns

# filling missing values
df['Embarked']=df['Embarked'].fillna(method='ffill')
df['Age']=df['Age'].fillna(df.Age.median())

#creating dummy variables
df=pd.get_dummies(df,columns=['Sex','Embarked','Title','Deck'])

#creating input and output variables to use to traing the model
features=df.ix[0:,1:].values
target=df['Survived'].values

'''
print(features.shape)
print(target)
'''

'''
#model decision tree classifier
my_tree=tree.DecisionTreeClassifier(max_depth = 6, min_samples_split =25, random_state = 1)
my_tree=my_tree.fit(features,target)
print(my_tree.score(features, target))
'''
#model Ada boost
ada=AdaBoostClassifier(DecisionTreeClassifier(),learning_rate=1,n_estimators = 50,random_state=1)
ada=ada.fit(features,target)
print("adaboost score = ",ada.score(features, target))

#model random forest classifier
my_tree2=RandomForestClassifier(max_depth=6,min_samples_split=25,n_estimators = 110,random_state = 1)
my_tree2=my_tree2.fit(features,target)
print("random forest score=",my_tree2.score(features, target))


df2=pd.read_csv("test.csv")     #loading the test.csv file

#adding new column Title to the test data
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

#adding a new columns Deck to the test data
df['Deck']='U'
for i in range(0,418):
    if pd.isnull(df2.ix[i,'Cabin']) :
       df2.ix[i,'Deck']='Unknown'
    else :
        df2.ix[i,'Deck']=df2.ix[i,'Cabin'][0]


df2['total_family_size']=df2['SibSp']+df2['Parch']      #adding a new columns total_family_size to the test data

#creating a passangerID feature to store passangerid in solution1.csv
passangerID=df2['PassengerId'].values      #store passangerid to use in creating csv submission.


df2=df2.drop(axis=1,labels=['PassengerId','Name','Ticket','Cabin'])     #dropping unwanted columns

#filling missing values
df2['Age']=df2['Age'].fillna(df2['Age'].median())
df2['Fare']=df2['Fare'].fillna(df2['Fare'].median())

df2=pd.get_dummies(df2,columns=['Sex','Embarked','Title','Deck'])   #creating dummy variables

#adding extra columns,these values are missing in the cabin feature in test data
df2['Deck_T']=0.0
df2['Deck_U']=0.0

features2=df2.ix[0:,0:].values      #creating input variable to use for prediction
#print(features2.shape)


prediction=my_tree2.predict(features2)  #making prediction for test data

solution=pd.DataFrame(prediction,passangerID,columns=['Survived'])  #creating dataframe with two  columns
solution.to_csv("solution1.csv",index_label='PassengerId')          #storing the dataframe in solution1.csv
#print(prediction)
