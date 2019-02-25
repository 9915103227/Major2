#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 17:18:23 2019

@author: pranav
"""

newFeature=[0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1,
 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1,
 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
 1, 1, 0, 1, 1, 0, 1, 0, 0, 0]

from sklearn.impute import SimpleImputer
from sklearn import preprocessing
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import math
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import random
import time
colWithCategories=[0,1,2,3,4,18,19,20,23,24,25,28,31,34,35,41,43,47,51,52,54,55,56,59,60,61,64,66,67,70]
dataset=pd.read_csv('train1.csv')
X=dataset.iloc[:,:-1].values;
y=dataset.iloc[:,82].values;
X_check=pd.DataFrame(X)
dataPoints=X.shape[0]
features=X.shape[1];

#Preprocessing
ordinalEncoder=preprocessing.OrdinalEncoder()
ordinalList=[ordinalEncoder for i in range(dataPoints)]
for feature in colWithCategories:
    o=preprocessing.OrdinalEncoder()
    missingValueImputer=SimpleImputer(missing_values=np.nan,strategy="constant", fill_value=0)
    X[:,feature]=missingValueImputer.fit_transform(X[:,feature].reshape(-1,1)).reshape(-1,)
    X[:,feature]=o.fit_transform(X[:,feature].reshape(-1,1)).reshape(-1,)
    ordinalList[feature]=o;

missingValueImputer=SimpleImputer(missing_values=np.nan,strategy="constant", fill_value=0)
X=missingValueImputer.fit_transform(X)

count=0
for feature in range(features):
    if newFeature[feature]==1:
        count=count+1

newX=np.zeros((dataPoints,count))
count=0
for feature in range(features):
    if newFeature[feature]==1:
        newX[:,count]=X[:,feature]
        count=count+1
    
from sklearn.cluster import KMeans
clusters=2
kMeans=KMeans(n_clusters=clusters,max_iter=10000)
kMeans.fit(newX)

y_pred=np.zeros((dataPoints))
for dataPoint in range(dataPoints) :
    if kMeans.predict([newX[dataPoint]])[0]==0:
        y_pred[dataPoint]=0
    else:
        y_pred[dataPoint]=1

from sklearn.metrics import confusion_matrix
confusion_matrix(y,y_pred)

from sklearn.metrics import accuracy_score
print(accuracy_score(y,y_pred))
        
        
        
        
        
        
        