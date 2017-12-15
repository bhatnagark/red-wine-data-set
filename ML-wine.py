#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 17:44:20 2017

@author: kshitijbhatnagar
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_validate    
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
from sklearn.feature_selection import RFE 
from sklearn.cross_validation import cross_val_score
import seaborn as sns
from sklearn.metrics import recall_score
from sklearn.cross_validation import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics.classification import cohen_kappa_score
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('/Users/kshitijbhatnagar/Desktop/winequality-red.csv', sep=";")

colNames = list(df)
print(colNames)
#   print(df[:,12])
nVar = len(colNames)
print(nVar)

#graphs
df.hist(column='alcohol')
df.hist(column='chlorides')
df.hist(column='residual sugar')
df.hist(column='fixed acidity')
df.hist(column='volatile acidity')
df.hist(column='citric acid')
df.hist(column='density')
df.hist(column='sulphates')
df.hist(column='pH')
df.hist(column='total sulfur dioxide')

#summary statistic
result=df.describe()
result.to_csv('stats.csv')

#to check imbalance in data

df.hist(column='quality')
extra = df[df.duplicated()]
extra.shape
df.quality.value_counts()

#co-relation matrix
def correlation_matrix(df):
    from matplotlib import pyplot as plt
    from matplotlib import cm as cm

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    cmap = cm.get_cmap('jet', 30)
    cax = ax1.imshow(df.corr(), interpolation="nearest", cmap=cmap)
    ax1.grid(True)
    plt.title('Wine Data set correlation')
    labels=['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'quality',]
    ax1.set_xticklabels(labels,fontsize=6)
    ax1.set_yticklabels(labels,fontsize=6)
    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    fig.colorbar(cax, ticks=[.75,.8,.85,.90,.95,1])
    plt.show()

correlation_matrix(df)
#splitiing data in train and test
X = df.iloc[:,:(nVar-1)]
print(X)
Y = df.iloc[:,(nVar-1):]
print(Y)
X_train, X_test,Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state=24) 
print(Y_train)


## Model 1-RandomForest
clf1 = RandomForestClassifier()
clf1.fit(X_train,Y_train)

#feature importance
clf1.feature_importances_()

predict = clf1.predict(X_test)


#cross val score
score1 = np.mean(cross_val_score(clf, X, Y, scoring='accuracy', cv=10))
print(score1)
## Metrics-accuracy
print(accuracy_score(predict,Y_test))

#kappa score
score3 = cohen_kappa_score(Y_test,predict)
print(score3)
#recall score
score2=recall_score(Y_test, predict, average='macro') 
print(score2)


#resampling to get a better model
minor_class = [4,8,3]
major_class = [5,6,7]
df.info()
df_minor = df[df.quality.isin(minor_class)]
df_major = df[df.quality.isin(major_class)]

#Upsampling
df_minor_upsampled = resample(df_minor,replace=True,n_samples = len(df_major), random_state = 123)
upsampled_df = pd.concat([df_major,df_minor_upsampled])

print(upsampled_df.quality.value_counts())

X = upsampled_df.iloc[:,:(nVar-1)]
Y = upsampled_df.iloc[:,(nVar-1):]

X_train, X_test,Y_train, Y_test = train_test_split(X,Y, test_size = 0.33, random_state=24) 


##RandomForest
clf2 = RandomForestClassifier()
clf2.fit(X_train,Y_train)

#clf2.feature_importances_()
predict = clf2.predict(X_test)

#Metric accuracy
print(accuracy_score(predict,Y_test))

#kappa score
print(cohen_kappa_score(Y_test,predict))

#recall score
print(recall_score(Y_test, predict, average='macro') )

#cross validation score
print(np.mean(cross_val_score(clf, X, Y, scoring='accuracy', cv=10)))
#confusion matrix
cf = confusion_matrix(predict,Y_test)
print(cf)
        
#model 2 - Log reg
X = df.iloc[:,:(nVar-1)]
print(X)
Y = df.iloc[:,(nVar-1):]
print(Y)
X_train, X_test,Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state=24) 
print(Y_train)
clf3 = LogisticRegression()
clf3.fit(X_train,Y_train)
predict2 = clf3.predict(X_test)
#np.mean(cross_val_score(clf, X, Y, scoring='accuracy', cv=10))
print(score1)
## Metrics-accuracy
print(accuracy_score(predict2,Y_test))

#kappa score
print(cohen_kappa_score(Y_test,predict2))

#recall score
print(recall_score(Y_test, predict2, average='macro')) 

#Upsampling
df_minor_upsampled = resample(df_minor,replace=True,n_samples = len(df_major), random_state = 123)
upsampled_df = pd.concat([df_major,df_minor_upsampled])

print(upsampled_df.quality.value_counts())

X = upsampled_df.iloc[:,:(nVar-1)]
Y = upsampled_df.iloc[:,(nVar-1):]

X_train, X_test,Y_train, Y_test = train_test_split(X,Y, test_size = 0.33, random_state=24) 

clf4 = LogisticRegression()
clf4.fit(X_train,Y_train)
predict3 = clf4.predict(X_test)
#np.mean(cross_val_score(clf, X, Y, scoring='accuracy', cv=10))
#print(score1)
## Metrics-accuracy
print(accuracy_score(predict3,Y_test))

#kappa score
print(cohen_kappa_score(Y_test,predict3))

#recall score
print(recall_score(Y_test, predict3, average='macro')) 

#plotting
from sklearn import metrics
# testing score
score = metrics.f1_score(y_test, pred, pos_label=list(set(y_test)))
# training score
score_train = metrics.f1_score(y_train, pred_train, pos_label=list(set(y_train)     

   