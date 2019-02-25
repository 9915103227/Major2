from sklearn.impute import SimpleImputer
from sklearn import preprocessing
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
colWithCategories=[0,1,2,3,4,18,19,20,23,24,25,28,31,34,35,41,43,47,51,52,54,55,56,59,60,61,64,66,67,70]
dataset=pd.read_csv('train1.csv')
X=dataset.iloc[:,:-1].values;
y=dataset.iloc[:,82].values;
X_check=pd.DataFrame(X)
dataPoints=X.shape[0]
features=X.shape[1];
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

from sklearn.cluster import KMeans
clusters=1000
kMeans=KMeans(n_clusters=clusters,max_iter=500)
kMeans.fit(X)

label=kMeans.labels_

logisticRegression=LogisticRegression()
logisticRegression.fit(X,y)

y_pred1=logisticRegression.predict(X)

from sklearn.metrics import confusion_matrix
confusion_matrix(y,y_pred1)




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
    sum=0
    for val in mapDataAndClusterY[cluster]:
        sum=sum+val
    if sum==len(mapDataAndClusterY[cluster]) or sum==0:
        for point in correspondingCentroidTest[cluster]:
            y_pred[point]=mapDataAndClusterY[cluster][0]
        continue
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