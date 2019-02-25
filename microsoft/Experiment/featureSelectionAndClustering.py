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

numberOfwhale=100
iterations=300
#initialization of whales:
centresOfwhale=np.zeros((numberOfwhale,features))
for whale in range(numberOfwhale):
    centresOfwhale[whale]=np.random.choice(2,features)

startTime=0
bestWhale=0
neigh = LogisticRegression()
for iteration in range(iterations):
    print("iteration "+str(iteration))
    fitness=np.zeros((numberOfwhale))
    bestWhale=0
    bestAccuracy=-100
    startTime=time.time();
    for whale in range(numberOfwhale):
        
        #print("whale: "+str(whale))
        numberOfFeature=0
        for feature in range(features):
            if centresOfwhale[whale][feature]==1:
                numberOfFeature=numberOfFeature+1
        
        if(numberOfFeature==0):
            continue
        newX=np.zeros((dataPoints,numberOfFeature))
        currFeature=0
        for feature in range(features):
            if centresOfwhale[whale][feature]==1:
                newX[:,currFeature]=X[:,feature];
        #apply knn
        neigh.fit(newX,y)
        accuracy=neigh.score(newX,y)
        fitness[whale]=accuracy
        if accuracy>bestAccuracy:
            bestAccuracy=accuracy
            bestWhale=whale
        
    print("classification applied")
    print(time.time()-startTime)
    startTime=time.time()
    
    #apply woaSa equations
    a=2-iteration*((2.00)/iterations) #eqn 2.3
    a2=-1+iteration*((-1.00)/iterations)
    for whale in range(numberOfwhale):
        r1=random.random()
        r2=random.random()
        A=2*a*r1-a;  # Eq. (2.3) in the paper
        C=2*r2;      # Eq. (2.4) in the paper
        p=random.random()
        b=1;               #  parameters in Eq. (2.5)
        l=(a2-1)*random.random()+1;   #  parameters in Eq. (2.5)
        
        for feature in range(features):
            if p<0.5 :
                if abs(A)>=1 :
                    #rand_leader_index = int(math.floor((numberOfwhale-1)*random.random()+1));
                    #tournament selection
                    rand_leader_index=0
                    itmp1=random.randint(0,numberOfwhale-1)
                    itmp2=random.randint(0,numberOfwhale-1)
                    r=random.random()
                    if r>0.5:
                        if fitness[itmp1]>fitness[itmp2]:
                            rand_leader_index=itmp1
                        else:
                            rand_leader_index=itmp2
                    else:
                        if fitness[itmp1]>fitness[itmp2]:
                            rand_leader_index=itmp2
                        else:
                            rand_leader_index=itmp1
                    X_rand = centresOfwhale[rand_leader_index]
                    
                    D_X_rand=X_rand
                    #mutationU
                    randomArr=np.zeros((features))
                    for feat in range(features):
                        randomArr[feat]=random.random()
                        if randomArr[feat]>=(iteration/iterations):
                            randomArr[feat]=1
                        else:
                            randomArr[feat]=0
                    
                    for feat in range(features):
                        if(randomArr[feat]==1):
                            random1=random.random()
                            if random1>0.5:
                                D_X_rand[feat]=1
                            else:
                                D_X_rand[feat]=0
                    #mutation done
                    
                    RE=centresOfwhale[whale]
                    #mutationU
                    randomArr=np.zeros((features))
                    for feat in range(features):
                        randomArr[feat]=random.random()
                        if randomArr[feat]>=(iteration/iterations):
                            randomArr[feat]=1
                        else:
                            randomArr[feat]=0
                    
                    for feat in range(features):
                        if(randomArr[feat]==1):
                            random1=random.random()
                            if random1>0.5:
                                RE[feat]=1
                            else:
                                RE[feat]=0
                    #mutation done
                    
                    #D_X_rand=abs(C*X_rand[feature]-centresOfwhale[whale,feature]); # Eq. (2.7)
                    
                    #CrossOverU started
                    for feat in range(features):
                        random1=random.random()
                        if random1<0.5:
                            centresOfwhale[whale][feat]=D_X_rand[feat]
                        else:
                            centresOfwhale[whale][feat]=RE[feat]
                    #CrossOverU done
                    #centresOfwhale[whale,feature]=X_rand[feature]-A*D_X_rand;      # Eq. (2.8)
                elif abs(A)<1 :
                    #D_Leader=abs(C*centresOfwhale[bestWhale,feature]-centresOfwhale[whale,feature]); # Eq. (2.1)
                    #mutationU
                    D_Leader=centresOfwhale[bestWhale]
                    randomArr=np.zeros((features))
                    for feat in range(features):
                        randomArr[feat]=random.random()
                        if randomArr[feat]>=(iteration/iterations):
                            randomArr[feat]=1
                        else:
                            randomArr[feat]=0
                    
                    for feat in range(features):
                        if(randomArr[feat]==1):
                            random1=random.random()
                            if random1>0.5:
                                D_Leader[feat]=1
                            else:
                                D_Leader[feat]=0
                    #mutation done
                    
                    #CrossOverU started
                    for feat in range(features):
                        random1=random.random()
                        if random1<0.5:
                            centresOfwhale[whale][feat]=centresOfwhale[bestWhale][feat]
                        else:
                            centresOfwhale[whale][feat]=D_Leader[feat]
                    #CrossOverU done
                    
                    
                    #centresOfwhale[whale,feature]=centresOfwhale[bestWhale,feature]-A*D_Leader;      # Eq. (2.2)
            elif p>=0.5 :
                distance2Leader=abs(centresOfwhale[bestWhale,feature]-centresOfwhale[whale,feature]);      # Eq. (2.5)
                centresOfwhale[whale,feature]=distance2Leader*math.exp(b*l)*math.cos(l*2*3.14)+centresOfwhale[bestWhale,feature];
                if centresOfwhale[whale][feature]>0.5:
                    centresOfwhale[whale][feature]=1
                else:
                    centresOfwhale[whale][feature]=0
        
    print("WOA applied")
    print(time.time()-startTime)            
        
numberOfFeature=0
whale=bestWhale
for feature in range(features):
    if centresOfwhale[whale][feature]==1:
        numberOfFeature=numberOfFeature+1
newX=np.zeros((dataPoints,numberOfFeature))
currFeature=0
for feature in range(features):
    if centresOfwhale[whale][feature]==1:
        newX[:,currFeature]=X[:,feature];
        #apply knn
neigh.fit(newX,y)
accuracy=neigh.score(newX,y)
    #fitness[whale]=accuracy
     #   if accuracy>bestAccuracy:
      #      bestAccuracy=accuracy
       #     bestWhale=whale
print(accuracy)

neigh.fit(X,y)
accuracy=neigh.score(X,y)
print(accuracy)
print(centresOfwhale[bestWhale])
print(numberOfFeature)