from sklearn.impute import SimpleImputer
from sklearn import preprocessing
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from array import *
import math

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
clusters=10
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

mapDataAndCluster= [[0 for dataPoint in range(dataPoints)] for cluster in range(clusters)]
numberOfDataPointsInCluster=[0 for cluster in range(clusters)]

for dataPoint in range(dataPoints):
    correspondingCentroid=label[dataPoint]
    mapDataAndCluster[correspondingCentroid][dataPoint]=1
    numberOfDataPointsInCluster[correspondingCentroid]=numberOfDataPointsInCluster[correspondingCentroid]+1

numberOfDataPointsInCluster=np.array(numberOfDataPointsInCluster)


y_pred=[1 for dataPoint in range(dataPoints)]
for dataPoint in range(dataPoints):
    correspondingCentroid=kMeans.predict([X[dataPoint]])
    
    numberOfDataPoints=numberOfDataPointsInCluster[correspondingCentroid]
    newX=[[0 for feature in range(features)] for dataPoint1 in range(numberOfDataPoints)]
    newY=[0 for dataPoint1 in range(numberOfDataPoints)]
    
    currRow=0;
    for dataPoint1 in range(dataPoints):
        if(mapDataAndCluster[correspondingCentroid[0]][dataPoint1]==1):
            #print("hi")
            for feature in range(features):
                newX[currRow][feature]=X[dataPoint1][feature]
            newY[currRow]=y[dataPoint1]
            currRow=currRow+1
    print(float(dataPoint)/float(dataPoints))
    logisticRegression=LogisticRegression()
    logisticRegression.fit(newX,newY)
    y_pred[dataPoint]=logisticRegression.predict([X[dataPoint]])[0]


y_pred=np.array(y_pred)

from sklearn.metrics import confusion_matrix
confusion_matrix(y,y_pred)