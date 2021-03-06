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

        
import time
numberOfwhale=3
iterations=5

dataPoints = X.shape[0]
features=X.shape[1];
learningRate=0.001
startTimeAlgo=time.time()
bestWhale=0
#initialize features
WFC1=np.random.uniform(low=-1,high=1,size=(numberOfwhale,features,100))*np.sqrt(1.00/features)
WC1=np.random.uniform(low=-1,high=1,size=(numberOfwhale,5,3,3))*np.sqrt(1.00/9)
WC2=np.random.uniform(low=-1,high=1,size=(numberOfwhale,5,3,3))*np.sqrt(1.00/45)
WFC2=np.random.uniform(low=-1,high=1,size=(numberOfwhale,36,84))*np.sqrt(1.00/36)
WFC3=np.random.uniform(low=-1,high=1,size=(numberOfwhale,84,10))*np.sqrt(1.00/84)
WOP=np.random.uniform(low=-1,high=1,size=(numberOfwhale,10,5))*np.sqrt(1.00/10)

#ADAMS parameters
beta1=0.9
beta2=0.999
import math
epsilon=math.pow(10,-8)
VdelWFC1=np.zeros((numberOfwhale,features,100))
VdelWFC2=np.zeros((numberOfwhale,36,84))
VdelWFC3=np.zeros((numberOfwhale,84,10))
VdelWC1=np.zeros((numberOfwhale,5,3,3))
VdelWC2=np.zeros((numberOfwhale,5,3,3))
VdelWOP=np.zeros((numberOfwhale,10,5))

SdelWFC1=np.zeros((numberOfwhale,features,100))
SdelWFC2=np.zeros((numberOfwhale,36,84))
SdelWFC3=np.zeros((numberOfwhale,84,10))
SdelWC1=np.zeros((numberOfwhale,5,3,3))
SdelWC2=np.zeros((numberOfwhale,5,3,3))
SdelWOP=np.zeros((numberOfwhale,10,5))
currIter=0


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
        for node1 in range(5):
            centresOfwhale[whale].append(WOP[whale][node][node1])

centresOfwhale=np.array(centresOfwhale)
features1=centresOfwhale.shape[1]

bestWhale=0



import math
import random
for iteration in range(iterations):
    print("iteration "+str(iteration))
    fitness=np.zeros((numberOfwhale))
    bestWhale=0
    bestAccuracy=math.pow(10,20)
    startTime=time.time();
    for whale in range(numberOfwhale):
        accuracy=0.00
        print("whale: "+str(whale))
        #apply cnn
        for dataPoint in range(1000):
            #initializations
            print("iteration "+str(iteration))
            print("whale: "+str(whale))
            print(dataPoint)
            #startTime=time.time()
            I1=np.zeros((features))
            sigmaF1=np.zeros((100))
            sigmaC1=np.zeros((5,8,8))
            sigmaC2=np.zeros((6,6))
            sigmaF2=np.zeros((84))
            sigmaF3=np.zeros((10))
            sigmaSoftmax=np.zeros((5))
        
            
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
            for nodeF3 in range(10):
                #sum+=F3[node]*WOP[whale][node]
                for nodeSoftmax in range(5):
                    sigmaSoftmax[nodeSoftmax]+=WOP[whale][nodeF3][nodeSoftmax]*F3[nodeF3]
            
            exps=np.array([np.exp(sigmaSoftmax[i]) for i in range(5)])
            sum_exps=np.sum(exps)
            
            softmax=np.array([exps[i]/sum_exps for i in range(5)])
            
            for i in range(5):
                if(y[dataPoint]==(i+1)):
                    sum+=-1*math.log(softmax[i])
                else:
                    sum+=-1*math.log(1-softmax[i])
                
            
            op=sum
            accuracy+=(math.pow(op,2))
         
        accuracy=math.sqrt(accuracy)
        fitness[whale]=accuracy
        if accuracy<bestAccuracy:
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
            for node1 in range(5):
                WOP[whale][node][node1]=centresOfwhale[whale][cnt]
                cnt+=1




#CNN
error=0.00
whale=bestWhale
yPred=np.zeros((10000))
for iteration in range(1):
    #print("iteration "+str(iteration))
    for whale1 in range(1):
        error=0.00
        for jPS in range(5):
            learningRate=0.001
            for dataPoint in range(10000):
                if(y[dataPoint]!=(jPS+1)):
                    continue
                currIter+=1
                #initializations
                print(iteration)
                print(dataPoint)
                #startTime=time.time()
                I1=np.zeros((features))
                sigmaF1=np.zeros((100))
                sigmaC1=np.zeros((5,8,8))
                sigmaC2=np.zeros((6,6))
                sigmaF2=np.zeros((84))
                sigmaF3=np.zeros((10))
                sigmaSoftmax=np.zeros((5))
                
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
                for nodeF3 in range(10):
                    #sum+=F3[node]*WOP[whale][node]
                    for nodeSoftmax in range(5):
                        sigmaSoftmax[nodeSoftmax]+=WOP[whale][nodeF3][nodeSoftmax]*F3[nodeF3]
                
                exps=np.array([np.exp(sigmaSoftmax[i]) for i in range(5)])
                sum_exps=np.sum(exps)
                
                softmax=np.array([exps[i]/sum_exps for i in range(5)])
                '''for i in range(5):
                    if(softmax[i]==1):
                        softmax[i]=0.999
                    if(softmax[i]==0):
                        softmax[i]=0.000001
                
                for i in range(5):
                    if(y[dataPoint]==(i+1)):
                        sum+=-1*math.log(softmax[i])
                    else:
                        sum+=-1*math.log(1-softmax[i])'''
                
                yPred[dataPoint]=sum
                print(sum)
                        
                #yPred[dataPoint]=op
                print(y[dataPoint])
                print(softmax)
                
                #backpropogation at output
                
                delsoftmax=np.zeros(5)
                for i in range(5):
                    if(y[dataPoint]==(i+1)):
                        delsoftmax[i]=-1/(softmax[i]+epsilon);
                        #delsoftmax[i]=softmax[i]*(softmax[i]-1)
                    else:
                        delsoftmax[i]=1/(1-softmax[i]+epsilon)
                        #delsoftmax[i]=softmax[i]*(softmax[i])
                
                print(delsoftmax)
                delSigmaOp=np.zeros(5)
                
                for i in range(5):
                    delSigmaOp[i]=((exps[i]*(sum_exps-exps[i]))/(sum_exps*sum_exps))*delsoftmax[i]  
                
                
                #backpropogation at output done
                
                #backpropogation at F3
                delWOP=np.zeros((10,5))
                delF3=np.zeros(10)
                
                for i in range(10):
                    for j in range(5):
                        delWOP[i][j]=delSigmaOp[j]*F3[i];
                        delF3[i]+=delSigmaOp[j]*WOP[whale][i][j]
                
                
                
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
                        #WFC1[whale][nodeI1][nodeF1]-=learningRate*delWFC1[nodeI1][nodeF1]
                        VdelWFC1[whale][nodeI1][nodeF1]=beta1*VdelWFC1[whale][nodeI1][nodeF1]+(1-beta1)*delWFC1[nodeI1][nodeF1]
                        SdelWFC1[whale][nodeI1][nodeF1]=beta2*SdelWFC1[whale][nodeI1][nodeF1]+(1-beta2)*math.pow(delWFC1[nodeI1][nodeF1],2)
                        Vcorr=VdelWFC1[whale][nodeI1][nodeF1]/(1-math.pow(beta1,currIter))
                        Scorr=SdelWFC1[whale][nodeI1][nodeF1]/(1-math.pow(beta2,currIter))
                        WFC1[whale][nodeI1][nodeF1]-=learningRate*Vcorr/(math.sqrt(Scorr)+epsilon)
                
                for kernel in range(5):
                    for rowKernel in range(3):
                        for colKernel in range(3):
                            #WC1[whale][kernel][rowKernel][colKernel]-=learningRate*delWC1[kernel][rowKernel][colKernel]
                            VdelWC1[whale][kernel][rowKernel][colKernel]=beta1*VdelWC1[whale][kernel][rowKernel][colKernel]+(1-beta1)*delWC1[kernel][rowKernel][colKernel]
                            SdelWC1[whale][kernel][rowKernel][colKernel]=beta2*SdelWC1[whale][kernel][rowKernel][colKernel]+(1-beta2)*math.pow(delWC1[kernel][rowKernel][colKernel],2)
                            Vcorr=VdelWC1[whale][kernel][rowKernel][colKernel]/(1-math.pow(beta1,currIter))
                            Scorr=SdelWC1[whale][kernel][rowKernel][colKernel]/(1-math.pow(beta2,currIter))
                            WC1[whale][kernel][rowKernel][colKernel]-=learningRate*Vcorr/(math.sqrt(Scorr)+epsilon)
                            
                for depthKernel in range(5):
                    for rowKernel in range(3):
                        for colKernel in range(3):
                            #WC2[whale][depthKernel][rowKernel][colKernel]-=0.00001*delWC2[depthKernel][rowKernel][colKernel]
                            VdelWC2[whale][depthKernel][rowKernel][colKernel]=beta1*VdelWC2[whale][depthKernel][rowKernel][colKernel]+(1-beta1)*delWC2[depthKernel][rowKernel][colKernel]
                            SdelWC2[whale][depthKernel][rowKernel][colKernel]=beta2*SdelWC2[whale][depthKernel][rowKernel][colKernel]+(1-beta2)*math.pow(delWC2[depthKernel][rowKernel][colKernel],2)
                            Vcorr=VdelWC2[whale][depthKernel][rowKernel][colKernel]/(1-math.pow(beta1,currIter))
                            Scorr=SdelWC2[whale][depthKernel][rowKernel][colKernel]/(1-math.pow(beta2,currIter))
                            WC2[whale][depthKernel][rowKernel][colKernel]-=learningRate*Vcorr/(math.sqrt(Scorr)+epsilon)
                
                for nodeC2 in range(36):
                    for nodeF2 in range(84):
                        #WFC2[whale][nodeC2][nodeF2]-=0.00001*delWFC2[nodeC2][nodeF2]
                        VdelWFC2[whale][nodeC2][nodeF2]=beta1*VdelWFC2[whale][nodeC2][nodeF2]+(1-beta1)*delWFC2[nodeC2][nodeF2]
                        SdelWFC2[whale][nodeC2][nodeF2]=beta2*SdelWFC2[whale][nodeC2][nodeF2]+(1-beta2)*math.pow(delWFC2[nodeC2][nodeF2],2)
                        Vcorr=VdelWFC2[whale][nodeC2][nodeF2]/(1-math.pow(beta1,currIter))
                        Scorr=SdelWFC2[whale][nodeC2][nodeF2]/(1-math.pow(beta2,currIter))
                        WFC2[whale][nodeC2][nodeF2]-=learningRate*Vcorr/(math.sqrt(Scorr)+epsilon)
                
                for nodeF2 in range(84):
                    for nodeF3 in range(10):
                       #WFC3[whale][nodeF2][nodeF3]-=0.00001*delWFC3[nodeF2][nodeF3]
                       VdelWFC3[whale][nodeF2][nodeF3]=beta1*VdelWFC3[whale][nodeF2][nodeF3]+(1-beta1)*delWFC3[nodeF2][nodeF3]
                       SdelWFC3[whale][nodeF2][nodeF3]=beta2*SdelWFC3[whale][nodeF2][nodeF3]+(1-beta2)*math.pow(delWFC3[nodeF2][nodeF3],2)
                       Vcorr=VdelWFC3[whale][nodeF2][nodeF3]/(1-math.pow(beta1,currIter))
                       Scorr=SdelWFC3[whale][nodeF2][nodeF3]/(1-math.pow(beta2,currIter))
                       WFC3[whale][nodeF2][nodeF3]-=learningRate*Vcorr/(math.sqrt(Scorr)+epsilon)
                
                for node in range(10):
                    for node1 in range(5):
                        #WOP[whale][node][node1]-=0.0000000001*delWOP[node][node1]
                        VdelWOP[whale][node][node1]=beta1*VdelWOP[whale][node][node1]+(1-beta1)*delWOP[node][node1]
                        SdelWOP[whale][node][node1]=beta2*SdelWOP[whale][node][node1]+(1-beta2)*math.pow(delWOP[node][node1],2)
                        Vcorr=VdelWOP[whale][node][node1]/(1-math.pow(beta1,currIter))
                        Scorr=SdelWOP[whale][node][node1]/(1-math.pow(beta2,currIter))
                        WOP[whale][node][node1]-=learningRate*Vcorr/(math.sqrt(Scorr)+epsilon)
                        
                learningRate/=1.2
                  
                    
#RESULT
y_pred=np.zeros((94300))
checkZ=np.zeros((94300,5))
checkSigmaF1=np.zeros((94300,100))
checkSigmaF3=np.zeros((94300,10))
checkSoftmax=np.zeros((94300,5))
whale=bestWhale
yPred=np.zeros((dataPoints))
for iteration in range(1):
    print("iteration "+str(iteration))
    for whale1 in range(1):
        error=0.00
        for dataPoint in range(94300):
            #initializations
            print(iteration)
            print(dataPoint)
            #startTime=time.time()
            I1=np.zeros((features))
            sigmaF1=np.zeros((100))
            sigmaC1=np.zeros((5,8,8))
            sigmaC2=np.zeros((6,6))
            sigmaF2=np.zeros((84))
            sigmaF3=np.zeros((10))
            sigmaSoftmax=np.zeros((5))
            I1=newDataX[dataPoint/100][dataPoint%100]
            
            #Forward passs at I1
            for nodeI1 in range(features):
                for nodeF1 in range(100):
                    sigmaF1[nodeF1]+=WFC1[whale][nodeI1][nodeF1]*I1[nodeI1]
            
            for i in range(100):
                checkSigmaF1[dataPoint][i]=sigmaF1[i]
            
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
            
            for i in range(10):
                checkSigmaF3[dataPoint][i]=sigmaF3[i]
                    
            F3=sigmaF3
            for nodeF3 in range(10):
                if F3[nodeF3]<0:
                    F3[nodeF3]*=0.05
            #Forward pass at F2 done
            
            #Forward pass at F3
            sum=0.00
            for nodeF3 in range(10):
                #sum+=F3[node]*WOP[whale][node]
                for nodeSoftmax in range(5):
                    sigmaSoftmax[nodeSoftmax]+=WOP[whale][nodeF3][nodeSoftmax]*F3[nodeF3]
            
            for i in range(5):
                checkZ[dataPoint][i]=sigmaSoftmax[i]
            
            exps=np.array([np.exp(sigmaSoftmax[i]) for i in range(5)])
            sum_exps=np.sum(exps)
            
            softmax=np.array([exps[i]/sum_exps for i in range(5)])
            
            checkSoftmax[dataPoint]=softmax
            '''for i in range(5):
                if(softmax[i]==1):
                    softmax[i]=0.999
                if(softmax[i]==0):
                    softmax[i]=0.000001
            
            for i in range(5):
                if(y[dataPoint]==(i+1)):
                    sum+=-1*math.log(softmax[i])
                else:
                    sum+=-1*math.log(1-softmax[i])'''
            maxN=softmax[0]
            maxPos=1
            for i in range(5):
                if softmax[i]>maxN:
                    maxN=softmax[i];
                    maxPos=i+1
            y_pred[dataPoint]=maxPos
            error+=abs(y[dataPoint]-maxPos)
            
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
        T+=1
        for j in range(1):
            I1=np.zeros((features))
            sigmaF1=np.zeros((100))
            sigmaC1=np.zeros((5,8,8))
            sigmaC2=np.zeros((6,6))
            sigmaF2=np.zeros((84))
            sigmaF3=np.zeros((10))
            sigmaSoftmax=np.zeros((5))
            I1=X[dataPoint]
            
            #Forward passs at I1
            for nodeI1 in range(features):
                for nodeF1 in range(100):
                    sigmaF1[nodeF1]+=WFC1[whale][nodeI1][nodeF1]*I1[nodeI1]
            
            #for i in range(100):
                #checkSigmaF1[dataPoint][i]=sigmaF1[i]
            
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
            
            '''for i in range(10):
                checkSigmaF3[dataPoint][i]=sigmaF3[i]'''
                    
            F3=sigmaF3
            for nodeF3 in range(10):
                if F3[nodeF3]<0:
                    F3[nodeF3]*=0.05
            #Forward pass at F2 done
            
            #Forward pass at F3
            sum=0.00
            for nodeF3 in range(10):
                #sum+=F3[node]*WOP[whale][node]
                for nodeSoftmax in range(5):
                    sigmaSoftmax[nodeSoftmax]+=WOP[whale][nodeF3][nodeSoftmax]*F3[nodeF3]
            
            '''for i in range(5):
                checkZ[dataPoint][i]=sigmaSoftmax[i]'''
            
            exps=np.array([np.exp(sigmaSoftmax[i]) for i in range(5)])
            sum_exps=np.sum(exps)
            
            softmax=np.array([exps[i]/sum_exps for i in range(5)])
            
            #checkSoftmax[dataPoint]=softmax
            '''for i in range(5):
                if(softmax[i]==1):
                    softmax[i]=0.999
                if(softmax[i]==0):
                    softmax[i]=0.000001
            
            for i in range(5):
                if(y[dataPoint]==(i+1)):
                    sum+=-1*math.log(softmax[i])
                else:
                    sum+=-1*math.log(1-softmax[i])'''
            maxN=softmax[0]
            maxPos=1
            for i in range(5):
                if softmax[i]>maxN:
                    maxN=softmax[i];
                    maxPos=i+1
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