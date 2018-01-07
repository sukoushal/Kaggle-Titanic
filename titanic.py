import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from sklearn.naive_bayes import BernoulliNB


df_train = pd.read_csv('/home/sukoushal/Dropbox/python/Titanic/train.csv')
df_test = pd.read_csv('/home/sukoushal/Dropbox/python/Titanic/test.csv')
combined = pd.concat([df_train, df_test])

sex_mapping = {'male' : 1, 'female' : 2 }
title_mapping = {'Mr':0, 'Mrs':1, 'Miss':2, 'Master':3, 'Dr':4, 'Don':5, 'Rev':6, 'Mme':7, 'Ms':8,'Major':9, 'Lady':10, 'Sir':11, 'Mlle':12, 'Col':13, 'Capt':14, 'the Countess':15,'Jonkheer':16}

age = combined[combined.Age == combined.Age]
age_na = combined[combined.Age != combined.Age]

# Filling Age
female = age[(age.Sex == 'male') & (age.Pclass > 2)].Age.mean()
male = age[(age.Sex == 'male') & (age.Pclass > 2)].Age.mean()



combined['title'] = combined['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())


df_c1 = combined[(combined.Age !=combined.Age) & (combined.Sex == 'male')]
df_c1['Age'] = df_c1['Age'].fillna(male)
df_c2 = combined[(combined.Age != combined.Age) & (combined.Sex == 'female')]
df_c2['Age'] = df_c2['Age'].fillna(female)
df_c3 = combined[combined.Age == combined.Age]

c = pd.concat([df_c3, df_c1, df_c2])
c['Sex'] = c['Sex'].map(sex_mapping)
c['title_map'] = c['title'].map(title_mapping)


train = c[c.Survived == c.Survived]
test = c[(c.Survived!=c.Survived) & (c.title_map == c.title_map)]

#DataSet
X = train[['Sex','Pclass','Age','SibSp','Parch', 'title_map']].values
y = train['Survived'].values
z = test[['Sex','Pclass','Age', 'SibSp', 'Parch', 'title_map']].values


#BernaulliNB
clf = BernoulliNB()
clf.fit(X,y)
p = clf.predict(z)



#Accuracy
ans = pd.read_csv('/home/sukoushal/Downloads/gender_submission.csv')
ans = ans.drop(414)
df_ans = pd.merge(test, ans, how = 'inner', on = 'PassengerId')
pred = pd.DataFrame(p, columns = ['Predicted'])
df_ans['Predicted'] = pred 
df_ans.to_csv('/home/sukoushal/Dropbox/python/Titanic/predict1.csv')


count_t = 0
count_f = 0
for i in range(len(df_ans)):
	if df_ans.Survived_y.iloc[i] == df_ans.Predicted.iloc[i]:
		count_t = count_t +1
	else:
		count_f = count_f +1

count_t = 391
count_f = 26