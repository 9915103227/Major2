#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 20:51:54 2019

@author: pranav
"""

from sklearn.impute import SimpleImputer
from sklearn import preprocessing
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

dataset=pd.read_csv('train1.csv')
X=dataset.iloc[:,:-1].values;
y=dataset.iloc[:,82].values;
missingValueImputer=SimpleImputer(strategy="most_frequent")
missingValueImputer.fit(X)
X=missingValueImputer.transform(X)
X_check=pd.DataFrame(X)

ordinalEncoder=preprocessing.OrdinalEncoder()
ordinalEncoder.fit(X)
X=ordinalEncoder.transform(X)
X_check=pd.DataFrame(X)

from sklearn.cluster import KMeans
clusters=100
kMeans=KMeans(n_clusters=clusters,max_iter=1000)
kMeans.fit(X)

label=kMeans.labels_

logisticRegression=LogisticRegression()
logisticRegression.fit(X,y)

y_pred1=logisticRegression.predict(X)

from sklearn.metrics import confusion_matrix
confusion_matrix(y,y_pred1)


dataPoints=X.shape[0]
features=X.shape[1];

#mapDataAndCluster= [[0 for dataPoint in range(dataPoints)] for cluster in range(clusters)]
mapDataAndCluster=[[] for cluster in range(clusters)]
mapDataAndClusterY=[[] for cluster in range(clusters)]
numberOfDataPointsInCluster=[0 for cluster in range(clusters)]

for dataPoint in range(dataPoints):
    correspondingCentroid=label[dataPoint]
    #mapDataAndCluster[correspondingCentroid][dataPoint]=1
    mapDataAndCluster[correspondingCentroid].append(X[dataPoint])    
    mapDataAndClusterY[correspondingCentroid].append(y[dataPoint])
    numberOfDataPointsInCluster[correspondingCentroid]=numberOfDataPointsInCluster[correspondingCentroid]+1

#numberOfDataPointsInCluster=np.array(numberOfDataPointsInCluster)
#mapDataAndCluster=np.array(mapDataAndCluster)
#mapDataAndClusterY=np.array(mapDataAndClusterY)
correspondingCentroidTest=[[] for cluster in range(clusters)]
y_pred=[1 for dataPoint in range(dataPoints)]
for dataPoint in range(dataPoints):
    correspondingCentroidTest[kMeans.predict([X[dataPoint]])[0]].append(dataPoint)

for cluster in range(clusters):
    logisticRegression=LogisticRegression()
    logisticRegression.fit(mapDataAndCluster[cluster],mapDataAndClusterY[cluster])
    
    for point in correspondingCentroidTest[cluster]:
        y_pred[point]=logisticRegression.predict([X[point]])[0]
    
    print(cluster)
    #newX=[[0 for feature in range(features)] for dataPoint1 in range(numberOfDataPoints)]
    #newY=[0 for dataPoint1 in range(numberOfDataPoints)]
    #newX=np.array(newX)
    #newY=np.array(newY)
    #currRow=0;
    #for dataPoint1 in mapDataAndCluster[correspondingCentroid[0]]:
    #newX=mapDataAndCluster[correspondingCentroid[0]]
    #newY=mapDataAndClusterY[correspondingCentroid[0]]
    #currRow=currRow+1
    


y_pred=np.array(y_pred)

from sklearn.metrics import confusion_matrix
confusion_matrix(y,y_pred)

from sklearn.metrics import accuracy_score
print(accuracy_score(y,y_pred))

print(accuracy_score(y,y_pred1))