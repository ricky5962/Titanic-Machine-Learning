import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPClassifier


train = pd.read_csv('titanic_train.csv')
test = pd.read_csv('titanic_test.csv')
test_temp = test
train.info()
#Missing data/ plot
#sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
#sns.set_style('whitegrid')
#sns.countplot(x='Survived', hue='Sex', data=train, palette='RdBu_r')
#sns.countplot(x='Survived', hue='Pclass', data=train)
#sns.distplot(train['Age'].dropna(),kde=False,bins=30)
#sns.countplot(x='SibSp',data=train)


# Dealing with NA
sns.boxplot(x='Pclass', y='Age', data=train)
def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    if pd.isnull(Age):
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        else:
            return 24
    else:
        return Age


# Training set process
train['Age'] = train[['Age','Pclass']].apply(impute_age, axis=1)
# drop Cabin
train.drop('Cabin', axis=1,inplace=True)
# drop na
train.dropna(inplace=True)
# dummy variables
sex = pd.get_dummies(train['Sex'], drop_first=True)
embark = pd.get_dummies(train['Embarked'],drop_first=True)

train = pd.concat([train,sex,embark],axis=1)
train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
train.drop('PassengerId',axis=1,inplace=True)

# Testing set process
test['Age'] = test[['Age','Pclass']].apply(impute_age, axis=1)
# drop Cabin
test.drop('Cabin', axis=1,inplace=True)
# drop na
test.dropna(inplace=True)

# dummy variables
sex = pd.get_dummies(test['Sex'], drop_first=True)
embark = pd.get_dummies(test['Embarked'],drop_first=True)

test = pd.concat([test,sex,embark],axis=1)
test.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
test.drop('PassengerId',axis=1,inplace=True)


# Logistic Regression
X_train = train.drop('Survived', axis=1)
y_train = train['Survived']

mlp = MLPClassifier(hidden_layer_sizes=(10,10,10),max_iter=500)
mlp.fit(X_train,y_train)
predictions = mlp.predict(test)
print(predictions)
submission = pd.DataFrame(test_temp['PassengerId'],columns=['PassengerId','Survived'])
submission['Survived'] = predictions
submission.to_csv('Submission.csv',index=False)