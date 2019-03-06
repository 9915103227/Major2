#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 21:11:35 2019

@author: pranav
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 18:14:18 2019

@author: pranav
"""

newFeature=[1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0]
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
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
colWithCategories=[0, 1, 2, 3, 17, 18, 19, 22, 23, 24, 27, 30, 33, 34, 40, 42, 46, 50, 51, 53, 54, 55, 58, 59, 60, 63, 65, 66, 69]
dataset=pd.read_csv('train1.csv')
X=dataset.iloc[:,1:-1].values;
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
newNewX=newX
newX=newX[:100,:]
newX=preprocessing.normalize(newX)        
numberOfwhale=1
iterations=10
dataPoints = newX.shape[0]
features=newX.shape[1];
learningRate=0.001

#initialize features
WFC1=np.random.uniform(low=0,high=1,size=(numberOfwhale,features,1024))
WC1=np.random.uniform(low=0,high=1,size=(numberOfwhale,1,5,5))
WC2=np.random.uniform(low=0,high=1,size=(numberOfwhale,1,1,5,5))
WFC2=np.random.uniform(low=0,high=1,size=(numberOfwhale,25,84))
WFC3=np.random.uniform(low=0,high=0.0001,size=(numberOfwhale,84,10))
WOP=np.random.uniform(low=0,high=1,size=(numberOfwhale,10))
bestWhale=0
bestMSE=float("inf")
for iteration in range(iterations):
    print("iteration "+str(iteration))
    for whale in range(numberOfwhale):
        error=0.00
        for dataPoint in range(dataPoints):
            #initializations
            startTime=time.time()
            I1=np.zeros((features))
            sigmaF1=np.zeros((1024))
            sigmaC1=np.zeros((1,28,28))
            P1=np.zeros((1,14,14))
            sigmaC2=np.zeros((1,10,10))
            P2=np.zeros((1,5,5))
            sigmaF2=np.zeros((84))
            sigmaF3=np.zeros((10))
            
            I1=newX[dataPoint]
            
            #forward propogation:
            
            #Full Convolution1
            F1=sigmaF1
            '''for nodeF1 in range(1024):
                sum=0.00
                for nodeI1 in range(features):
                    sum=sum+WFC1[whale][nodeI1][nodeF1]
                sigmaF1[nodeF1]=sum
                if(sum>0):
                    F1[nodeF1]=sum
                else:
                    F1[nodeF1]=0.05*sum
            '''
            sigmaF1=np.dot(WFC1[whale].transpose(),I1)
            for nodeF1 in range(1024):
                if sigmaF1[nodeF1]>=0:
                    F1[nodeF1]=sigmaF1[nodeF1]
                else:
                    F1[nodeF1]=0.05*sigmaF1[nodeF1]
                '''if sigmaF1[nodeF1]>=40:
                    sigmaF1[nodeF1]=1.00
                elif sigmaF1[nodeF1]<=-40:
                    F1[nodeF1]=0.00
                else:
                    F1[nodeF1]=1/(1+math.exp(-1*sigmaF1[nodeF1]))'''
            #Full Convolution1 done
            
            #Convolution 1
            F1=F1.reshape((32,32))
            C1=sigmaC1
            for depth in range(1):
                for height in range(28):
                    for width in range(28):
                        sum=0.00
                        for row in range(5):
                            for column in range(5):
                                sum=sum+WC1[whale][depth][row][column]*F1[row+height][column+width]                                
                        sigmaC1[depth][height][width]=sum
                        if(sum>=0):
                            C1[depth][height][width]=sum
                        else:
                            C1[depth][height][width]=0.05*sum
                        '''if sum>=40:
                            sum=1.00
                        elif sum<=-40:
                            sum=0.00
                        else:
                            sum=1/(1+math.exp(-1*sum))
                        C1[depth][height][width]=sum'''
            
            #Convolution1 done
                        
            #Pool1 started
            #stride=2
            
            for depthP1 in range(1):
                for rowP1 in range(14):
                    for colP1 in range(14):
                        sum=0.00
                        for rowKernel in range(2):
                            for colKernel in range(2):
                                sum=sum+C1[depthP1][rowKernel+rowP1*2][colKernel+colP1*2]
                        sum=sum/4
                        P1[depthP1][rowP1][colP1]=sum
            
            #Pool1 ended
            
            #Convolution2 started:
            C2=sigmaC2
            for depthC2 in range(1):
                for rowC2 in range(10):
                    for colC2 in range(10):
                        sum=0.00
                        for depthWC2 in range(1):
                            for rowWC2 in range(5):
                                for colWC2 in range(5):
                                    sum=sum+WC2[whale][depthC2][depthWC2][rowWC2][colWC2]*C1[depthWC2][rowC2+rowWC2][colC2+colWC2]
                        sigmaC2[depthC2][rowC2][colC2]=sum
                        if sum>=0:
                            C2[depthC2][rowC2][colC2]=sum
                        else:
                            C2[depthC2][rowC2][colC2]=0.05*sum
                        '''if sum>=40:
                            sum=1.00
                        elif sum<=-40:
                            sum=0.00
                        else:
                            sum=1/(1+math.exp(-1*sum))
                        C2[depthC2][rowC2][colC2]=sum'''
            
            #Convolution 2 ended
            
            #Pool2 started
            for depthP2 in range(1):
                for rowP2 in range(5):
                    for colP2 in range(5):
                        sum=0.00
                        for rowKernel in range(2):
                            for colKernel in range(2):
                                sum=sum+C2[depthP2][rowP2*2+rowKernel][colP2*2+colKernel]
                        sum=sum/4
                        P2[depthP2][rowP2][colP2]=sum
            #Pool2 ended
            
            #Full convolution2 started:
            P2=P2.reshape((25))
            F2=sigmaF2
            '''
            for nodeF2 in range(84):
                sum=0.00
                for nodeP2 in range(400):
                    sum=sum+WFC2[whale][nodeP2][nodeF2]
                sigmaF2[nodeF2]=sum
                if(sum>0):
                    F2[nodeF2]=sum
                else:
                    F2[nodeF2]=0.05*sum
            '''
            sigmaF2=np.dot(WFC2[whale].transpose(),P2)
            for nodeF2 in range(84):
                if sigmaF2[nodeF2]>=0:
                    F2[nodeF2]=sigmaF2[nodeF2]
                else:
                    F2[nodeF2]=sigmaF2[nodeF2]*0.05
                '''sum=sigmaF2[nodeF2]
                if sum>=40:
                    sum=1.00
                elif sum<=-40:
                    sum=0.00
                else:
                    sum=1/(1+math.exp(-1*sum))
                F2[nodeF2]=sum'''
            #Full Convolution2 done
            
            #Full Convolution3 started
            F3=sigmaF3
            '''
            for nodeF3 in range(10):
                sum=0.00
                for nodeF2 in range(84):
                    sum=sum+WFC3[whale][nodeF2][nodeF3]
                sigmaF3[nodeF3]=sum
                if(sum>0):
                    F3[nodeF3]=sum
                else:
                    F3[nodeF3]=0.05*sum
            '''
            sigmaF3=np.dot(WFC3[whale].transpose(),F2)
            for nodeF3 in range(10):
                if sigmaF3[nodeF3]>=0:
                    F3[nodeF3]=sigmaF3[nodeF3]
                else:
                    F3[nodeF3]=sigmaF3[nodeF3]*0.05
                
                '''sum=sigmaF3[nodeF3]
                if sum>=40:
                    sum=1.00
                elif sum<=-40:
                    sum=0.00
                else:
                    sum=1/(1+math.exp(-1*sum))
                F3[nodeF3]=sum'''
            #Full Convolution1 done
            
            
            #output layer:
            sum=0.00
            for nodeF3 in range(10):
                sum=sum+F3[nodeF3]*WOP[whale][nodeF3]
            if sum>=40:
                sum=1.00
            elif sum<=-40:
                sum=0.00
            else:
                sum=1/(1+math.exp(-1*sum))
            
            print(sum)
            #Forward propogation done
            print("forward pass done "+str(dataPoint))
            #Backward propogation:
            L=pow((float(sum)-float(y[dataPoint])),2)/2
            
            #ouptut Node backpropogation:
            delSum=(float(sum)-float(y[dataPoint]))
            delSigmaSum=delSum*sum*(1-sum)
            #output node done
            
            #backpropogation for F3 started:
            delWOP=np.zeros((10))
            delF3=np.zeros((10))
            delSigmaF3=np.zeros((10))
            '''
            for nodeF3 in range(10):
                delWOP[nodeF3]=delSigmaSum*F3[nodeF3]
                delF3[nodeF3]=delSigmaSum*WOP[whale][nodeF3]
                if F3[nodeF3]>=0:
                    delSigmaF3[nodeF3]=delF3[nodeF3]
                else:
                    delSigmaF3[nodeF3]=delF3[nodeF3]*0.05
            '''
            delWOP=np.dot(delSigmaSum,F3)
            delF3=np.dot(delSigmaSum,WOP[whale])
            for nodeF3 in range(10):
                delSigmaF3[nodeF3]=max(delF3[nodeF3],delF3[nodeF3]*0.05)
                #delSigmaF3[nodeF3]=delF3[nodeF3]*F3[nodeF3]*(1-F3[nodeF3])
            #backpropogation for F3 done
            
            #backpropogation for F2 started:
            delWFC3=np.zeros((84,10))
            delF2=np.zeros((84))
            delSigmaF2=np.zeros((84))
            
            for nodeF2 in range(84):
                for nodeF3 in range(10):
                    delWFC3[nodeF2][nodeF3]=delSigmaF3[nodeF3]*F2[nodeF2]
                    delF2[nodeF2]=delF2[nodeF2]+delSigmaF3[nodeF3]*WFC3[whale][nodeF2][nodeF3]
                if F2[nodeF2]>=0 :
                    delSigmaF2[nodeF2]=delF2[nodeF2]
                else:
                    delSigmaF2[nodeF2]=delF2[nodeF2]*0.05
            
            '''delWFC3=np.outer(F2,F3.transpose())
            delF2=np.dot(WFC3[whale],F3)
            for nodeF2 in range(84):
                delSigmaF2[nodeF2]=max(delF2[nodeF2],delF2[nodeF2]*0.05)
                #delSigmaF2[nodeF2]=delF2[nodeF2]*F2[nodeF2]*(1-F2[nodeF2])'''
            #backpropogation of F2 done
            
            #backpropogation at P2
            delWFC2=np.zeros((25,84))
            delP2=np.zeros((25))
            delSigmaP2=np.zeros((25))
            
            for nodeP2 in range(25):
                for nodeF2 in range(84):
                    delWFC2[nodeP2][nodeF2]=delSigmaF2[nodeF2]*P2[nodeP2]
                    delP2[nodeP2]=delP2[nodeP2]+delSigmaF2[nodeF2]*WFC2[whale][nodeP2][nodeF2]
                if P2[nodeP2]>=0:
                    delSigmaP2[nodeP2]=delP2[nodeP2]
                else:
                    delSigmaP2[nodeP2]=delP2[nodeP2]
            
            '''delWFC2=np.outer(P2,F2)
            delP2=np.dot(WFC2[whale],F2)'''
            delP2=delP2.reshape((1,5,5))
            delSigmaP2=delP2
            #backpropogation at P2 done
            
            #backpropogation at C2 started:
            delC2=np.zeros((1,10,10))
            delSigmaC2=delC2
            for depthP2 in range(1):
                for rowP2 in range(5):
                    for colP2 in range(5):
                        for kernelRow in range(2):
                            for kernelCol in range(2):
                                delC2[depthP2][2*rowP2+kernelRow][2*colP2+kernelCol]+=float(delSigmaP2[depthP2][rowP2][colP2])/4
            
            for depthC2 in range(1):
                for rowC2 in range(10):
                    for colC2 in range(10):
                        if C2[depthC2][rowC2][colC2]>=0:
                            delSigmaC2[depthC2][rowC2][colC2]=delC2[depthC2][rowC2][colC2]
                        else:
                            delSigmaC2[depthC2][rowC2][colC2]=0.05*delC2[depthC2][rowC2][colC2]
                        #delSigmaC2[depthC2][rowC2][colC2]=delC2[depthC2][rowC2][colC2]*C2[depthC2][rowC2][colC2]*(1-C2[depthC2][rowC2][colC2])
            
            #backpropogation at C2 done
            
            #backpropogation at P1 started:
            delP1=np.zeros((1,14,14))
            delWC2=np.zeros((1,1,5,5))
            for depthC2 in range(1):
                for rowC2 in range(10):
                    for colC2 in range(10):
                        for kernelDepth in range(1):
                            for kernelRow in range(5):
                                for kernelCol in range(5):
                                    delWC2[depthC2][kernelDepth][kernelRow][kernelCol]+=delSigmaC2[depthC2][rowC2][colC2]*P1[kernelDepth][rowC2+kernelRow][colC2+kernelCol]
                                    delP1[kernelDepth][kernelRow+rowC2][kernelCol+colC2]+=delSigmaC2[depthC2][rowC2][colC2]*WC2[whale][depthC2][kernelDepth][kernelRow][kernelCol]
            
            
            #backpropogation at P1 done

            #backpropogation at C1 started:
            delC1=np.zeros((1,28,28))
            delSigmaC1=delC1
            for depthP1 in range(1):
                for rowP1 in range(14):
                    for colP1 in range(14):
                        for kernelRow in range(2):
                            for kernelCol in range(2):
                                delC1[depthP1][kernelRow+2*rowP1][kernelCol+2*colP1]+=float(delP1[depthP1][rowP1][colP1])/4
            
            for depthC1 in range(1):
                for rowC1 in range(28):
                    for colC1 in range(28):
                        if C1[depthC1][rowC1][colC1]>=0:
                            delSigmaC1[depthC1][rowC1][colC1]=delC1[depthC1][rowC1][colC1]
                        else:
                            delSigmaC1[depthC1][rowC1][colC1]=delC1[depthC1][rowC1][colC1]*0.05
                        #delSigmaC1[depthC1][rowC1][colC1]=delC1[depthC1][rowC1][colC1]*C1[depthC1][rowC1][colC1]*(1-C1[depthC1][rowC1][colC1])
            
            #backpropogation at C1 done
            
            #backpropogation at F1 started:
            delWC1=np.zeros((1,5,5))
            delF1=np.zeros((32,32))
            delSigmaF1=delF1
            
            for depthC1 in range(1):
                for rowC1 in range(28):
                    for colC1 in range(28):
                        for kernelRow in range(5):
                            for kernelCol in range(5):
                                delWC1[depthC1][kernelRow][kernelCol]+=delSigmaC1[depthC1][rowC1][colC1]*F1[kernelRow+rowC1][kernelCol+colC1]
                                delF1[kernelRow+rowC1][kernelCol+colC1]+=delSigmaC1[depthC1][rowC1][colC1]*WC1[whale][depthC1][kernelRow][kernelCol]
            
            
            for rowF1 in range(32):
                for colF1 in range(32):
                    if F1[rowF1][colF1]>=0:
                        delSigmaF1[rowF1][colF1]=delF1[rowF1][colF1]
                    else:
                        delSigmaF1[rowF1][colF1]=delF1[rowF1][colF1]*0.05
                    
                    #delSigmaF1[rowF1][colF1]=delF1[rowF1][colF1]*F1[rowF1][colF1]*(1-F1[rowF1][colF1])
                        
            delF1=delF1.reshape((1024))
            delSigmaF1=delSigmaF1.reshape((1024))
            #backpropogation at F1 done
            
            #propogation at I1 started:
            delWFC1=np.zeros((features,1024))
            
            for nodeI1 in range(features):
                for nodeF1 in range(1024):
                    delWFC1[nodeI1][nodeF1]=delSigmaF1[nodeF1]*I1[nodeI1]
            
            
            '''delWFC1=np.outer(I1,F1)'''
            #backpropogation at I1 done
            
            #gradient descent:
            #for WFC1
            for nodeI1 in range(features):
                for nodeF1 in range(1024):
                    WFC1[whale][nodeI1][nodeF1]-=learningRate*delWFC1[nodeI1][nodeF1]
            
            #For WC1:
            for depthC1 in range(1):
                for kernelRow in range(5):
                    for kernelCol in range(5):
                        WC1[whale][depthC1][kernelRow][kernelCol]-=learningRate*delWC1[depthC1][kernelRow][kernelCol]
            
            #For WC2
            for depthC2 in range(1):
                for kernelDepth in range(1):
                    for kernelRow in range(5):
                        for kernelCol in range(5):
                            WC2[whale][depthC2][kernelDepth][kernelRow][kernelCol]-=learningRate*delWC2[depthC2][kernelDepth][kernelRow][kernelCol]
            
            #For WFC2
            for nodeP2 in range(25):
                for nodeF2 in range(84):
                    WFC2[whale][nodeP2][nodeF2]-=learningRate*delWFC2[nodeP2][nodeF2]
            
            #for WFC3:
            for nodeF2 in range(84):
                for nodeF3 in range(10):
                    WFC3[whale][nodeF2][nodeF3]-=learningRate*delWFC3[nodeF2][nodeF3]
            
            #for WOP:
            for node in range(10):
                WOP[whale][node]-=learningRate*delWOP[node]
            print(time.time()-startTime)

whale=1
#newX=newNewX
yPred=np.zeros(dataPoints)
for iteration in range(1):
    print("iteration "+str(iteration))
    for whale in range(numberOfwhale):
        error=0.00
        for dataPoint in range(dataPoints):
            #initializations
            startTime=time.time()
            I1=np.zeros((features))
            sigmaF1=np.zeros((1024))
            sigmaC1=np.zeros((1,28,28))
            P1=np.zeros((1,14,14))
            sigmaC2=np.zeros((1,10,10))
            P2=np.zeros((1,5,5))
            sigmaF2=np.zeros((84))
            sigmaF3=np.zeros((10))
            
            I1=newX[dataPoint]
            
            #forward propogation:
            
            #Full Convolution1
            F1=sigmaF1
            '''for nodeF1 in range(1024):
                sum=0.00
                for nodeI1 in range(features):
                    sum=sum+WFC1[whale][nodeI1][nodeF1]
                sigmaF1[nodeF1]=sum
                if(sum>0):
                    F1[nodeF1]=sum
                else:
                    F1[nodeF1]=0.05*sum
            '''
            sigmaF1=np.dot(WFC1[whale].transpose(),I1)
            for nodeF1 in range(1024):
                if sigmaF1[nodeF1]>=0:
                    F1[nodeF1]=sigmaF1[nodeF1]
                else:
                    F1[nodeF1]=0.05*sigmaF1[nodeF1]
                '''if sigmaF1[nodeF1]>=40:
                    sigmaF1[nodeF1]=1.00
                elif sigmaF1[nodeF1]<=-40:
                    F1[nodeF1]=0.00
                else:
                    F1[nodeF1]=1/(1+math.exp(-1*sigmaF1[nodeF1]))'''
            #Full Convolution1 done
            
            #Convolution 1
            F1=F1.reshape((32,32))
            C1=sigmaC1
            for depth in range(1):
                for height in range(28):
                    for width in range(28):
                        sum=0.00
                        for row in range(5):
                            for column in range(5):
                                sum=sum+WC1[whale][depth][row][column]*F1[row+height][column+width]                                
                        sigmaC1[depth][height][width]=sum
                        if(sum>=0):
                            C1[depth][height][width]=sum
                        else:
                            C1[depth][height][width]=0.05*sum
                        '''if sum>=40:
                            sum=1.00
                        elif sum<=-40:
                            sum=0.00
                        else:
                            sum=1/(1+math.exp(-1*sum))
                        C1[depth][height][width]=sum'''
            
            #Convolution1 done
                        
            #Pool1 started
            #stride=2
            
            for depthP1 in range(1):
                for rowP1 in range(14):
                    for colP1 in range(14):
                        sum=0.00
                        for rowKernel in range(2):
                            for colKernel in range(2):
                                sum=sum+C1[depthP1][rowKernel+rowP1*2][colKernel+colP1*2]
                        sum=sum/4
                        P1[depthP1][rowP1][colP1]=sum
            
            #Pool1 ended
            
            #Convolution2 started:
            C2=sigmaC2
            for depthC2 in range(1):
                for rowC2 in range(10):
                    for colC2 in range(10):
                        sum=0.00
                        for depthWC2 in range(1):
                            for rowWC2 in range(5):
                                for colWC2 in range(5):
                                    sum=sum+WC2[whale][depthC2][depthWC2][rowWC2][colWC2]*C1[depthWC2][rowC2+rowWC2][colC2+colWC2]
                        sigmaC2[depthC2][rowC2][colC2]=sum
                        if sum>=0:
                            C2[depthC2][rowC2][colC2]=sum
                        else:
                            C2[depthC2][rowC2][colC2]=0.05*sum
                        '''if sum>=40:
                            sum=1.00
                        elif sum<=-40:
                            sum=0.00
                        else:
                            sum=1/(1+math.exp(-1*sum))
                        C2[depthC2][rowC2][colC2]=sum'''
            
            #Convolution 2 ended
            
            #Pool2 started
            for depthP2 in range(1):
                for rowP2 in range(5):
                    for colP2 in range(5):
                        sum=0.00
                        for rowKernel in range(2):
                            for colKernel in range(2):
                                sum=sum+C2[depthP2][rowP2*2+rowKernel][colP2*2+colKernel]
                        sum=sum/4
                        P2[depthP2][rowP2][colP2]=sum
            #Pool2 ended
            
            #Full convolution2 started:
            P2=P2.reshape((25))
            F2=sigmaF2
            '''
            for nodeF2 in range(84):
                sum=0.00
                for nodeP2 in range(400):
                    sum=sum+WFC2[whale][nodeP2][nodeF2]
                sigmaF2[nodeF2]=sum
                if(sum>0):
                    F2[nodeF2]=sum
                else:
                    F2[nodeF2]=0.05*sum
            '''
            sigmaF2=np.dot(WFC2[whale].transpose(),P2)
            for nodeF2 in range(84):
                if sigmaF2[nodeF2]>=0:
                    F2[nodeF2]=sigmaF2[nodeF2]
                else:
                    F2[nodeF2]=sigmaF2[nodeF2]*0.05
                '''sum=sigmaF2[nodeF2]
                if sum>=40:
                    sum=1.00
                elif sum<=-40:
                    sum=0.00
                else:
                    sum=1/(1+math.exp(-1*sum))
                F2[nodeF2]=sum'''
            #Full Convolution2 done
            
            #Full Convolution3 started
            F3=sigmaF3
            '''
            for nodeF3 in range(10):
                sum=0.00
                for nodeF2 in range(84):
                    sum=sum+WFC3[whale][nodeF2][nodeF3]
                sigmaF3[nodeF3]=sum
                if(sum>0):
                    F3[nodeF3]=sum
                else:
                    F3[nodeF3]=0.05*sum
            '''
            sigmaF3=np.dot(WFC3[whale].transpose(),F2)
            for nodeF3 in range(10):
                if sigmaF3[nodeF3]>=0:
                    F3[nodeF3]=sigmaF3[nodeF3]
                else:
                    F3[nodeF3]=sigmaF3[nodeF3]*0.05
                
                '''sum=sigmaF3[nodeF3]
                if sum>=40:
                    sum=1.00
                elif sum<=-40:
                    sum=0.00
                else:
                    sum=1/(1+math.exp(-1*sum))
                F3[nodeF3]=sum'''
            #Full Convolution1 done
            
            
            #output layer:
            sum=0.00
            for nodeF3 in range(10):
                sum=sum+F3[nodeF3]*WOP[whale][nodeF3]
            if sum>=40:
                sum=1.00
            elif sum<=-700:
                sum=0.00
            else:
                sum=1/(1+math.exp(-1*sum))
            yPred[dataPoint]=sum



yDash=y[:100]

from sklearn.metrics import roc_auc_score
roc_auc_score(yDash,yPred)