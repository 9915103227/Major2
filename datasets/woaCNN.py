#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 18:55:13 2019


"""

import numpy as np
import pandas as pd
from sklearn import preprocessing
import math
import random
import time

dataset=pd.read_csv('breast/data.csv')
X=dataset.iloc[:,1:-1]
y=dataset.iloc[:,31]
X=preprocessing.normalize(X)

numberOfwhale=3
iterations=4

dataPoints = X.shape[0]
features=X.shape[1];
learningRate=0.0001
startTimeAlgo=time.time()
#initialize features
WFC1=np.random.uniform(low=0,high=1,size=(numberOfwhale,features,100))
WC1=np.random.uniform(low=0,high=1,size=(numberOfwhale,5,3,3))
WC2=np.random.uniform(low=0,high=1,size=(numberOfwhale,5,3,3))
WFC2=np.random.uniform(low=0,high=0.01,size=(numberOfwhale,36,84))
WFC3=np.random.uniform(low=0,high=0.0001,size=(numberOfwhale,84,10))
WOP=np.random.uniform(low=0,high=1,size=(numberOfwhale,10))

#Application of WOA

centresOfwhale=[[] for whale in range(numberOfwhale)]
fitness=np.zeros((numberOfwhale))
for whale in range(numberOfwhale):
    for nodeI1 in range(features):
        for nodeF1 in range(100):
            centresOfwhale[whale].append(WFC1[whale][nodeI1][nodeF1])
    
    for kernel in range(5):
        for rowKernel in range(3):
            for colKernel in range(3):
                centresOfwhale[whale].append(WC1[whale][kernel][rowKernel][colKernel])
    
    for depthKernel in range(5):
        for rowKernel in range(3):
            for colKernel in range(3):
                centresOfwhale[whale].append(WC2[whale][depthKernel][rowKernel][colKernel])
    
    for nodeC2 in range(36):
        for nodeF2 in range(84):
            centresOfwhale[whale].append(WFC2[whale][nodeC2][nodeF2])
    
    for nodeF2 in range(84):
        for nodeF3 in range(10):
            centresOfwhale[whale].append(WFC3[whale][nodeF2][nodeF3])
    
    for node in range(10):
        centresOfwhale[whale].append(WOP[whale][node])

centresOfwhale=np.array(centresOfwhale)
features1=centresOfwhale.shape[1]

for iteration in range(iterations):
    print("iteration "+str(iteration))
    fitness=np.zeros((numberOfwhale))
    bestWhale=0
    bestAccuracy=-100
    startTime=time.time();
    for whale in range(numberOfwhale):
        accuracy=0.00
        print("whale: "+str(whale))
        #apply cnn
        for dataPoint in range(dataPoints):
            #initializations
            print dataPoint
            #startTime=time.time()
            I1=np.zeros((features))
            sigmaF1=np.zeros((100))
            sigmaC1=np.zeros((5,8,8))
            sigmaC2=np.zeros((6,6))
            sigmaF2=np.zeros((84))
            sigmaF3=np.zeros((10))
            
            I1=X[dataPoint]
            
            #Forward passs at I1
            for nodeI1 in range(features):
                for nodeF1 in range(100):
                    sigmaF1[nodeF1]+=WFC1[whale][nodeI1][nodeF1]*I1[nodeI1]
            
            F1=sigmaF1
            for nodeF1 in range(100):
                if sigmaF1[nodeF1]<0:
                    F1[nodeF1]*=0.05
            sigmaF1=sigmaF1.reshape((10,10))
            F1=F1.reshape((10,10))
            #Forward Pass at I1 done
            
            #Forward pass at F1
            for kernel in range(5):
                for rowC1 in range(8):
                    for colC1 in range(8):
                        for rowKernel in range(3):
                            for colKernel in range(3):
                                sigmaC1[kernel][rowC1][colC1]+=WC1[whale][kernel][rowKernel][colKernel]*F1[rowC1+rowKernel][colC1+colKernel]
            
            C1=sigmaC1
            for kernel in range(5):
                for rowC1 in range(8):
                    for colC1 in range(8):
                        if C1[kernel][rowC1][colC1]<0:
                            C1[kernel][rowC1][colC1]*=0.05
            #Forward pass at F1 done
            
            #Forward pass at C1
            for rowC2 in range(6):
                for colC2 in range(6):
                    for depthKernel in range(5):
                        for rowKernel in range(3):
                            for colKernel in range(3):
                                sigmaC2[rowC2][colC2]+=WC2[whale][depthKernel][rowKernel][colKernel]*C1[depthKernel][rowKernel+rowC2][colKernel+colC2]
            
            C2=sigmaC2
            for rowC2 in range(6):
                for colC2 in range(6):
                    if sigmaC2[rowC2][colC2]<0:
                        C2[rowC2][colC2]*=0.05
            
            sigmaC2=sigmaC2.reshape((36,1))
            C2=C2.reshape((36,1))
            #Forward pass at C1 done
            
            #Forward pass at C2
            for nodeC2 in range(36):
                for nodeF2 in range(84):
                    sigmaF2[nodeF2]+=WFC2[whale][nodeC2][nodeF2]*C2[nodeC2]
            
            F2=sigmaF2
            for nodeF2 in range(84):
                if F2[nodeF2]<0:
                    F2[nodeF2]*=0.05
            #Forward pass at C2 done
            
            #Forward pass at F2
            for nodeF2 in range(84):
                for nodeF3 in range(10):
                    sigmaF3[nodeF3]+=WFC3[whale][nodeF2][nodeF3]*F2[nodeF3]
                    
            F3=sigmaF3
            for nodeF3 in range(10):
                if F3[nodeF3]<0:
                    F3[nodeF3]*=0.05
            #Forward pass at F2 done
            
            #Forward pass at F3
            sum=0.00
            for node in range(10):
                sum+=F3[node]*WOP[whale][node]
            
            op=sum=1/(1+math.exp(-1*sum))
            accuracy+=(math.pow(op-y[dataPoint],2))
         
        accuracy=math.sqrt(accuracy)
        fitness[whale]=accuracy
        if accuracy>bestAccuracy:
            bestAccuracy=accuracy
            bestWhale=whale
        
    print("cnn applied")
    print(time.time()-startTime)
    startTime=time.time()
    
    #apply woaSa equations
    a=2-iteration*((2.00)/iterations) #eqn 2.3
    a2=-1+iteration*((-1.00)/iterations)
    for whale in range(numberOfwhale):
        print(whale)
        r1=random.random()
        r2=random.random()
        A=2*a*r1-a;  # Eq. (2.3) in the paper
        C=2*r2;      # Eq. (2.4) in the paper
        p=random.random()
        b=1;               #  parameters in Eq. (2.5)
        l=(a2-1)*random.random()+1;   #  parameters in Eq. (2.5)
        
        for cluster in range(features1):
            if p<0.5 :
                if abs(A)>=1 :
                    rand_leader_index = int(math.floor((numberOfwhale-1)*random.random()+1));
                    X_rand = centresOfwhale[rand_leader_index]
                    D_X_rand=abs(C*X_rand[cluster]-centresOfwhale[whale,cluster]); # Eq. (2.7)
                    centresOfwhale[whale,cluster]=X_rand[cluster]-A*D_X_rand*0.001;      # Eq. (2.8)
                elif abs(A)<1 :
                    D_Leader=abs(C*centresOfwhale[bestWhale,cluster]-centresOfwhale[whale,cluster]); # Eq. (2.1)
                    centresOfwhale[whale,cluster]=centresOfwhale[bestWhale,cluster]-A*D_Leader*0.001;      # Eq. (2.2)
            elif p>=0.5 :
                distance2Leader=abs(centresOfwhale[bestWhale,cluster]-centresOfwhale[whale,cluster]);      # Eq. (2.5)
                centresOfwhale[whale,cluster]=distance2Leader*math.exp(b*l)*math.cos(l*2*3.14)*0.001+centresOfwhale[whale,cluster];
        
    print("WOA applied")
    print(time.time()-startTime)



    
    for whale in range(numberOfwhale):
        cnt=0
        for nodeI1 in range(features):
            for nodeF1 in range(100):
                WFC1[whale][nodeI1][nodeF1]=centresOfwhale[whale][cnt]
                cnt+=1
        
        for kernel in range(5):
            for rowKernel in range(3):
                for colKernel in range(3):
                    WC1[whale][kernel][rowKernel][colKernel]=centresOfwhale[whale][cnt]
                    cnt+=1
        
        for depthKernel in range(5):
            for rowKernel in range(3):
                for colKernel in range(3):
                    WC2[whale][depthKernel][rowKernel][colKernel]=centresOfwhale[whale][cnt]
                    cnt+=1
        
        for nodeC2 in range(36):
            for nodeF2 in range(84):
                WFC2[whale][nodeC2][nodeF2]=centresOfwhale[whale][cnt]
                cnt+=1
        
        for nodeF2 in range(84):
            for nodeF3 in range(10):
                WFC3[whale][nodeF2][nodeF3]=centresOfwhale[whale][cnt]
                cnt+=1
        
        for node in range(10):
            WOP[whale][node]=centresOfwhale[whale][cnt]
            cnt+=1








#CNN
whale=bestWhale
yPred=np.zeros((dataPoints))
for iteration in range(iterations):
    print("iteration "+str(iteration))
    for whale1 in range(1):
        error=0.00
        for dataPoint in range(dataPoints):
            #initializations
            print dataPoint
            #startTime=time.time()
            I1=np.zeros((features))
            sigmaF1=np.zeros((100))
            sigmaC1=np.zeros((5,8,8))
            sigmaC2=np.zeros((6,6))
            sigmaF2=np.zeros((84))
            sigmaF3=np.zeros((10))
            
            I1=X[dataPoint]
            
            #Forward passs at I1
            for nodeI1 in range(features):
                for nodeF1 in range(100):
                    sigmaF1[nodeF1]+=WFC1[whale][nodeI1][nodeF1]*I1[nodeI1]
            
            F1=sigmaF1
            for nodeF1 in range(100):
                if sigmaF1[nodeF1]<0:
                    F1[nodeF1]*=0.05
            sigmaF1=sigmaF1.reshape((10,10))
            F1=F1.reshape((10,10))
            #Forward Pass at I1 done
            
            #Forward pass at F1
            for kernel in range(5):
                for rowC1 in range(8):
                    for colC1 in range(8):
                        for rowKernel in range(3):
                            for colKernel in range(3):
                                sigmaC1[kernel][rowC1][colC1]+=WC1[whale][kernel][rowKernel][colKernel]*F1[rowC1+rowKernel][colC1+colKernel]
            
            C1=sigmaC1
            for kernel in range(5):
                for rowC1 in range(8):
                    for colC1 in range(8):
                        if C1[kernel][rowC1][colC1]<0:
                            C1[kernel][rowC1][colC1]*=0.05
            #Forward pass at F1 done
            
            #Forward pass at C1
            for rowC2 in range(6):
                for colC2 in range(6):
                    for depthKernel in range(5):
                        for rowKernel in range(3):
                            for colKernel in range(3):
                                sigmaC2[rowC2][colC2]+=WC2[whale][depthKernel][rowKernel][colKernel]*C1[depthKernel][rowKernel+rowC2][colKernel+colC2]
            
            C2=sigmaC2
            for rowC2 in range(6):
                for colC2 in range(6):
                    if sigmaC2[rowC2][colC2]<0:
                        C2[rowC2][colC2]*=0.05
            
            sigmaC2=sigmaC2.reshape((36,1))
            C2=C2.reshape((36,1))
            #Forward pass at C1 done
            
            #Forward pass at C2
            for nodeC2 in range(36):
                for nodeF2 in range(84):
                    sigmaF2[nodeF2]+=WFC2[whale][nodeC2][nodeF2]*C2[nodeC2]
            
            F2=sigmaF2
            for nodeF2 in range(84):
                if F2[nodeF2]<0:
                    F2[nodeF2]*=0.05
            #Forward pass at C2 done
            
            #Forward pass at F2
            for nodeF2 in range(84):
                for nodeF3 in range(10):
                    sigmaF3[nodeF3]+=WFC3[whale][nodeF2][nodeF3]*F2[nodeF3]
                    
            F3=sigmaF3
            for nodeF3 in range(10):
                if F3[nodeF3]<0:
                    F3[nodeF3]*=0.05
            #Forward pass at F2 done
            
            #Forward pass at F3
            sum=0.00
            for node in range(10):
                sum+=F3[node]*WOP[whale][node]
            
            op=sum=1/(1+math.exp(-1*sum))
            yPred[dataPoint]=op
            
            #backpropogation at output
            delOp=op-y[dataPoint]
            delSigmaOp=delOp*op*(1-op)
            #backpropogation at output done
            
            #backpropogation at F3
            delWOP=delSigmaOp*F3
            delF3=delSigmaOp*WOP[whale]
            delSigmaF3=delF3
            for node in range(10):
                if F3[node]<0:
                    delSigmaF3[node]*=0.005
            #backpropogation at F3 done
            
            #backpropogation at F2
            delWFC3=np.zeros((84,10))
            delF2=np.zeros((84))
            for nodeF2 in range(84):
                for nodeF3 in range(10):
                    delWFC3[nodeF2][nodeF3]+=delSigmaF3[nodeF3]*F2[nodeF2]
                    delF2[nodeF2]+=delSigmaF3[nodeF3]*WFC3[whale][nodeF2][nodeF3]
            
            delSigmaF2=delF2
            for nodeF2 in range(84):
                if F2[nodeF2]<0:
                    delSigmaF2[nodeF2]*=0.05
            #bakpropogation at F2 done
            
            #backpropogation at C2
            delWFC2=np.zeros((36,84))
            delSigmaC2=np.zeros((36))
            for nodeC2 in range(36):
                for nodeF2 in range(84):
                    delWFC2[nodeC2][nodeF2]+=delSigmaF2[nodeF2]*C2[nodeC2]
                    delSigmaC2[nodeC2]+=delSigmaF2[nodeF2]*WFC2[whale][nodeC2][nodeF2]
            
            delC2=delSigmaC2
            for nodeC2 in range(36):
                if C2[nodeC2]<0:
                    delC2[nodeC2]*=0.05
            
            delSigmaC2=delSigmaC2.reshape((6,6))
            delC2=delC2.reshape((6,6))
            #backpropogation at C2
            
            #backpropogation at C1
            delWC2=np.zeros((5,3,3))
            delC1=np.zeros((5,8,8))
            for rowC2 in range(6):
                for colC2 in range(6):
                    for depthKernel in range(5):
                        for rowKernel in range(3):
                            for colKernel in range(3):
                                delWC2[depthKernel][rowKernel][colKernel]+=delSigmaC2[rowC2][colC2]*C1[depthKernel][rowKernel+rowC2][colKernel+colC2]
                                delC1[depthKernel][rowKernel+rowC2][colKernel+colC2]+=delSigmaC2[rowC2][colC2]*WC2[whale][depthKernel][rowKernel][colKernel]
            
            delSigmaC1=delC1
            for depthC1 in range(5):
                for rowC1 in range(8):
                    for colC1 in range(8):
                        if C1[depthC1][rowC1][colC1]<0:
                            delSigmaC1[depthC1][rowC1][colC1]*=0.05
            
            #backpropogation at C1 done
            
            #backpropgation at F1:
            delWC1=np.zeros((5,3,3))
            delF1=np.zeros((10,10))
            for kernel in range(5):
                for rowC1 in range(8):
                    for colC1 in range(8):
                        for rowKernel in range(3):
                            for colKernel in range(3):
                                delWC1[kernel][rowKernel][colKernel]+=delSigmaC1[kernel][rowC1][colC1]*F1[rowC1+rowKernel][colC1+colKernel]
                                delF1[rowC1+rowKernel][colC1+colKernel]+=delSigmaC1[kernel][rowC1][colC1]*WC1[whale][kernel][rowKernel][colKernel]
            
            delSigmaF1=delF1
            for rowF1 in range(10):
                for colF1 in range(10):
                    if F1[rowF1][colF1]<0:
                        delSigmaF1[rowF1][colF1]*=0.05
            
            delSigmaF1=delSigmaF1.reshape((100,1))
            delF1=delF1.reshape((100,1))
            #backpropogation at F1 done
            
            #backpropogation at I1:
            delWFC1=np.zeros((features,100))
            for nodeI1 in range(features):
                for nodeF1 in range(100):
                    delWFC1[nodeI1][nodeF1]+=delF1[nodeF1]*I1[nodeI1]
                            
            #backpropogation at I1 done
            
            #gradient descent:
            for nodeI1 in range(features):
                for nodeF1 in range(100):
                    WFC1[whale][nodeI1][nodeF1]-=learningRate*delWFC1[nodeI1][nodeF1]
            
            for kernel in range(5):
                for rowKernel in range(3):
                    for colKernel in range(3):
                        WC1[whale][kernel][rowKernel][colKernel]-=learningRate*delWC1[kernel][rowKernel][colKernel]
                        
            for depthKernel in range(5):
                for rowKernel in range(3):
                    for colKernel in range(3):
                        WC2[whale][depthKernel][rowKernel][colKernel]-=learningRate*delWC2[depthKernel][rowKernel][colKernel]
            
            for nodeC2 in range(36):
                for nodeF2 in range(84):
                    WFC2[whale][nodeC2][nodeF2]-=learningRate*delWFC2[nodeC2][nodeF2]
            
            for nodeF2 in range(84):
                for nodeF3 in range(10):
                   WFC3[whale][nodeF2][nodeF3]-=learningRate*delWFC3[nodeF2][nodeF3]
            
            for node in range(10):
                WOP[whale][node]-=learningRate*delWOP[node]
            
            
            
#For result
yPred=np.zeros((dataPoints))
for iteration in range(1):
    print("iteration "+str(iteration))
    for whale1 in range(1):
        error=0.00
        for dataPoint in range(dataPoints):
            #initializations
            print dataPoint
            #startTime=time.time()
            I1=np.zeros((features))
            sigmaF1=np.zeros((100))
            sigmaC1=np.zeros((5,8,8))
            sigmaC2=np.zeros((6,6))
            sigmaF2=np.zeros((84))
            sigmaF3=np.zeros((10))
            
            I1=X[dataPoint]
            
            #Forward passs at I1
            for nodeI1 in range(features):
                for nodeF1 in range(100):
                    sigmaF1[nodeF1]+=WFC1[whale][nodeI1][nodeF1]*I1[nodeI1]
            
            F1=sigmaF1
            for nodeF1 in range(100):
                if sigmaF1[nodeF1]<0:
                    F1[nodeF1]*=0.05
            sigmaF1=sigmaF1.reshape((10,10))
            F1=F1.reshape((10,10))
            #Forward Pass at I1 done
            
            #Forward pass at F1
            for kernel in range(5):
                for rowC1 in range(8):
                    for colC1 in range(8):
                        for rowKernel in range(3):
                            for colKernel in range(3):
                                sigmaC1[kernel][rowC1][colC1]+=WC1[whale][kernel][rowKernel][colKernel]*F1[rowC1+rowKernel][colC1+colKernel]
            
            C1=sigmaC1
            for kernel in range(5):
                for rowC1 in range(8):
                    for colC1 in range(8):
                        if C1[kernel][rowC1][colC1]<0:
                            C1[kernel][rowC1][colC1]*=0.05
            #Forward pass at F1 done
            
            #Forward pass at C1
            for rowC2 in range(6):
                for colC2 in range(6):
                    for depthKernel in range(5):
                        for rowKernel in range(3):
                            for colKernel in range(3):
                                sigmaC2[rowC2][colC2]+=WC2[whale][depthKernel][rowKernel][colKernel]*C1[depthKernel][rowKernel+rowC2][colKernel+colC2]
            
            C2=sigmaC2
            for rowC2 in range(6):
                for colC2 in range(6):
                    if sigmaC2[rowC2][colC2]<0:
                        C2[rowC2][colC2]*=0.05
            
            sigmaC2=sigmaC2.reshape((36,1))
            C2=C2.reshape((36,1))
            #Forward pass at C1 done
            
            #Forward pass at C2
            for nodeC2 in range(36):
                for nodeF2 in range(84):
                    sigmaF2[nodeF2]+=WFC2[whale][nodeC2][nodeF2]*C2[nodeC2]
            
            F2=sigmaF2
            for nodeF2 in range(84):
                if F2[nodeF2]<0:
                    F2[nodeF2]*=0.05
            #Forward pass at C2 done
            
            #Forward pass at F2
            for nodeF2 in range(84):
                for nodeF3 in range(10):
                    sigmaF3[nodeF3]+=WFC3[whale][nodeF2][nodeF3]*F2[nodeF3]
                    
            F3=sigmaF3
            for nodeF3 in range(10):
                if F3[nodeF3]<0:
                    F3[nodeF3]*=0.05
            #Forward pass at F2 done
            
            #Forward pass at F3
            sum=0.00
            for node in range(10):
                sum+=F3[node]*WOP[whale][node]
            
            op=sum=1/(1+math.exp(-1*sum))
            yPred[dataPoint]=op



            
            
            
from sklearn.linear_model import LogisticRegression            
from sklearn.metrics import roc_auc_score
lr=LogisticRegression()
lr.fit(X,y)
z=lr.predict_proba(X)
z.shape
roc_auc_score(y,yPred)
            
print(time.time()-startTimeAlgo)            
            
            
            
            
            
