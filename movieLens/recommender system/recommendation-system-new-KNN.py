#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 16:05:49 2019

@author: pranav
"""

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import Normalizer

datasetData=pd.read_csv('data.csv')
datasetItem=pd.read_csv('item.csv')
datasetUser=pd.read_csv('user.csv')

lenDatasetData=datasetData.shape[0]
lenDatasetUser=datasetUser.shape[1]
lenDatasetItem=datasetItem.shape[1]
numberOfUser=datasetUser.shape[0]
numberOfItem=datasetItem.shape[0]
X=np.zeros((lenDatasetData,(datasetUser.shape[1]+1)))
y=np.zeros((lenDatasetData))
matrix=np.zeros((numberOfUser,numberOfItem))
for i in range(lenDatasetData):
    print i
    tmp1=datasetData.iloc[i,:].values
    user=tmp1[0]
    item=tmp1[1];
    rate=tmp1[2];
    matrix[user-1][item-1]=rate
    tmp1=datasetUser.iloc[user-1,:].values
    for j in range(lenDatasetUser):
        #X[i][j]=datasetUser.iloc[user-1,:].values[j]
        tmp=tmp1[j]
        if isinstance(tmp,basestring):
            X[i][j]=0
        else:
            X[i][j]=tmp
    tmp1=datasetItem.iloc[item-1,:].values
    for j in range(lenDatasetItem):
        #X[i][lenDatasetUser+j]=datasetItem.iloc[item-1,:].values[j]
        tmp=tmp1[j]
        if(tmp==1):
            X[i][3]=2*(j+1)
    
    y[i]=rate


X=SimpleImputer(strategy="constant",fill_value=0).fit_transform(X)
normalizer=Normalizer().fit(X)
X=normalizer.transform(X)

#create dataset of the user-item on whose ratings are not there

newDataX=np.zeros((943,100,4))

for user in range(numberOfUser):
    print(user)
    i=0
    for item in range(numberOfItem):
        if(matrix[user][item]==0 and i<100):
            #newDataPoint=[]
            tmp1=datasetUser.iloc[user,:].values
            for j in range(lenDatasetUser):
                #X[i][j]=datasetUser.iloc[user-1,:].values[j]
                tmp=tmp1[j]
                if isinstance(tmp,basestring):
                    newDataX[user][i][j]=0
                    #newDataPoint.append(0)
                else:
                    newDataX[user][i][j]=tmp
                    #newDataPoint.append(tmp)
            tmp1=datasetItem.iloc[item,:].values
            for j in range(lenDatasetItem):
                #X[i][lenDatasetUser+j]=datasetItem.iloc[item-1,:].values[j]
                tmp=tmp1[j]
                if(tmp==1):
                    newDataX[user][i][3]=2*(j+1)
                    #newDataPoint.append(2*(j+1))
            #newDataX.append(newDataPoint)
            i+=1

for user in range(numberOfUser):
    newDataX[user]=normalizer.transform(newDataX[user])

        

from sklearn.neighbors import KNeighborsClassifier
lr=KNeighborsClassifier()
lr.fit(X,y)
                  
                    
#RESULT
y_pred=np.zeros((94300))

for iteration in range(1):
    print("iteration "+str(iteration))
    for whale1 in range(1):
        error=0.00
        for dataPoint in range(94300):
            #initializations
            print(dataPoint)
            I1=newDataX[dataPoint/100][dataPoint%100]
             
            y_pred[dataPoint]=lr.predict(I1.reshape(1,-1))
            
            
'''from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(X,y)
y_pred=lr.predict(X)'''

y_pred=y_pred.reshape((943,100))
for user in range(943):
    y_pred[user]=np.sort(y_pred[user])

T=0
ans=[]
for dataPoint in range(100000):
    print(dataPoint)
    if(y[dataPoint]==5):
        print(dataPoint)
        T+=1
        for j in range(1):
            
            I1=X[dataPoint]
            
            maxPos=lr.predict(I1.reshape(1,-1))
            #findRank/
            tmp1=datasetData.iloc[i,:].values
            user=tmp1[0]
            arr=y_pred[user-1]
            k=99
            while(k>=0):
                if(maxPos>=arr[k]):
                    break
                k-=1
            if(k>=0):
                ans.append(99-k+1)
            else:
                ans.append(100)

#ans=np.array(ans)

#calculate recall precision
N=1
print("recall")
print("precision")
recall=np.zeros((90))
precision=np.zeros((90))
while(N<=90):
    print("at"+str(N))
    hit=0.00
    for a1 in ans:
        if(a1<=N):
            hit+=1
    recall[N-1]=hit/T
    print(hit/T)
    precision[N-1]=recall[N-1]/N
    print((hit/T)/N)
    N+=1