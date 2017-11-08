import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_score



train = pd.read_csv('titanic_train.csv')
test = pd.read_csv('titanic_test.csv')

train.describe()
train['Age'].fillna(train['Age'].median(),inplace=True)

# Visualizing Survival by Sex
plt.hist([train[train['Survived']==1]['Age'], train[train['Survived']==0]['Age']],stacked=True,
         color=['g','r'],bins=30,label=['Survived','Dead'])

# Survival by Fare
plt.hist([train[train['Survived']==1]['Fare'], train[train['Survived']==0]['Fare']],stacked=True,
         color=['g','r'],bins=30,label=['Survived','Dead'])

# Combine data for processing
target = train.Survived
train.drop('Survived',1,inplace=True)
combine = train.append(test)
combine.reset_index(inplace=True)
combine.drop('index',inplace=True,axis=1)

# get title
combine['Title']= combine['Name'].map(lambda name: name.split(',')[1].split('.')[0].strip())

# Age chart
grouped=combine.groupby(['Sex','Pclass','Title'])
grouped_median=grouped.median()

# Fill age NA with grouped_median table
combine['Age']=combine['Age'].map(lambda age: grouped_median.loc['female',1,'Miss']['Age'] if np.isnan(age) else age)
# Fill fare NA with mean
combine.Fare.fillna(combine.Fare.mean(),inplace=True)
# Fill embark with S
combine.Embarked.fillna('S',inplace=True)
#Drop unhelpful features
combine.drop('Ticket',axis=1,inplace=True)
combine.drop('Cabin',axis=1,inplace=True)
combine.drop('Name',axis=1,inplace=True)

# Add family size
combine['FamilySize']=combine['Parch'] + combine['SibSp'] + 1

# Dummy variables
sex_dummy = pd.get_dummies(combine['Sex'],drop_first=True)
embark_dummy = pd.get_dummies(combine['Embarked'],drop_first=True)
title_dummy = pd.get_dummies(combine['Title'],drop_first=True)
pclass_dummy = pd.get_dummies(combine['Pclass'],drop_first=True)

combine.drop(['Sex','Embarked','Pclass','Title','PassengerId'],axis=1,inplace=True)

# concat
combine = pd.concat([combine,sex_dummy,embark_dummy,title_dummy,pclass_dummy],axis=1)
#print(combine.info())


# split training and testing data set
trainOriginal=pd.read_csv('titanic_train.csv')
targets = trainOriginal.Survived
train = combine.head(891)
test = combine.iloc[891:]

print(train.info())
# Feature selection
clf = RandomForestClassifier(n_estimators=50, max_features='sqrt')
clf = clf.fit(train, targets)

features = pd.DataFrame()
features['feature'] = train.columns
features['importance'] = clf.feature_importances_
features.sort_values(by=['importance'], ascending=True, inplace=True)
features.set_index('feature',inplace=True)

features.plot(kind='barh')
#plt.show()

# Eliminate unwanted features
model = SelectFromModel(clf, prefit=True)
train_reduced = model.transform(train)
test_reduced = model.transform(test)


# Final model
parameters = {'bootstrap': False, 'n_estimators': 50, 'max_features': 'sqrt', 'max_depth': 6}
model = RandomForestClassifier(**parameters)
model.fit(train, targets)

def compute_score(clf, X, y, scoring='accuracy'):
    xval = cross_val_score(clf, X, y, scoring='accuracy')
    return np.mean(xval)

print(compute_score(model, train, targets, scoring='accuracy'))

#output
output = model.predict(test).astype(int)
df_output = pd.DataFrame()
aux = pd.read_csv('titanic_test.csv')
df_output['PassengerId'] = aux['PassengerId']
df_output['Survived'] = output
df_output[['PassengerId','Survived']].to_csv('output.csv',index=False)