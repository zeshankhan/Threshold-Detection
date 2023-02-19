#Paths to the dataset
dev_path='/kaggle/input/kvasirv2/me2018_densenet169_500_updatingLR_train.csv'
test_path='/kaggle/input/kvasirv2/me2018_densenet169_500_updatingLR_test.csv'


import os
import pandas as pd,numpy as np, os
from sklearn.metrics import f1_score, precision_recall_fscore_support, confusion_matrix
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.utils import shuffle


train=pd.read_csv(dev_path)
test=pd.read_csv(test_path)
print(f1_score(y_true=train['Actual'],y_pred=train['Pred'],average="weighted"))
print(f1_score(y_true=test['Actual'],y_pred=test['Pred'],average="weighted"))
print(confusion_matrix(y_true=test['Actual'],y_pred=test['Pred']))


thresholds=[0.6,0.1,0.2,0.4,0.01,0.6,0.6,0.6,0.4,0.4,0.4,0.8,0.6,0.4,0.6,0.6] #Computed

X_test=test.iloc[:,1:17]
Y_pred=np.zeros((8740),np.uint8)
for i in range(len(Y_pred)):
    for j in range(16):
        if(X_test.iloc[i,j]<=thresholds[j]):
            X_test.iloc[i,j]=0
    Y_pred[i]=np.argmax(X_test.iloc[i,:])+1
    
print(f1_score(y_true=test['Actual'],y_pred=Y_pred,average=None))
print(f1_score(y_true=test['Actual'],y_pred=Y_pred,average="weighted"))
cm=confusion_matrix(y_true=test['Actual'],y_pred=Y_pred)
print(cm)

print(X_test)
count2=0
count0=0
for i in range(len(X_test)):
    if(sum(X_test.iloc[i,:])>1):
        count2+=1
    if(sum(X_test.iloc[i,:])==0):
        count0+=1
print(count0,count2)
X_test.to_csv("temp.csv")
#print(len(thresholds))


label_map16=['retroflex-rectum', 'out-of-patient', 'ulcerative-colitis', 'normal-cecum', 'normal-z-line', 'dyed-lifted-polyps', 'blurry-nothing', 'retroflex-stomach', 'instruments', 'dyed-resection-margins', 'stool-plenty', 'esophagitis', 'normal-pylorus', 'polyps', 'stool-inclusions', 'colon-clear']
label_map16.append("Actual")
thresholds=[0.6,0.2,0.4,0.4,0.3,0.6,0.6,0.6,0.4,0.4,0.4,0.2,0.6,0.4,0.6,0.6]

X_test=test.iloc[:,1:17]
results=test.iloc[:,0:16]
results=results.astype('object')
results["Actual"]=test["Actual"]
results=results.set_index(test['image_name'])
results.columns=label_map16
print("Start")
#np.zeros((8740,16),dtype='object')
Y_pred=np.zeros((8740),np.uint8)
for i in range(len(Y_pred)):
    for j in range(16):
        if(X_test.iloc[i,j]<thresholds[j]):
            X_test.iloc[i,j]=0
        if(X_test.iloc[i,j]>0):
            results.iloc[i,j]=label_map16[j]
        else:
            results.iloc[i,j]=""
print("Completed")
Y_pred[i]=np.argmax(X_test.iloc[i,:])+1
results.to_csv("results.csv")


for c in results.columns[:-2]:
    countt=0
    countp=0
    countn=0
    print(c,results[results[c]==c][c].count())
    for i in range(8740):
        if(results['Actual1'][i]==c):
            countt+=1
        if(results[c][i]==results['Actual1'][i]):
            countp+=1
    print(c,countt,countp)
    
    
def compute_f1(threshold=None,X_test=None,Y_test=None):
    Y_pred=np.zeros((8740),np.uint8)
    for i in range(len(Y_pred)):
        Y_pred[i]=np.argmax(X_test.iloc[i,:])+1
        for j in range(16):
            if(X_test.iloc[i,j]<=threshold[j]):
                X_test.iloc[i,j]=0
            if(X_test.iloc[i,j]>threshold[j]):
                X_test.iloc[i,j]=1
        if(np.sum(X_test.iloc[i,:])==1):
            Y_pred[i]=np.argmax(X_test.iloc[i,:])+1
    return f1_score(y_true=Y_test,y_pred=Y_pred,average="weighted")

def sort(A,On):
    for i in range(len(A)-1):
        for j in range(i+1,len(A)):
            if(On[i]<On[j]):
                x=On[i]
                On[i]=On[j]
                On[j]=x
                x=A[i]
                A[i]=A[j]
                A[j]=x
    return A,On

import random

def mutate(A):
    for i in range(len(A)):
        for j in range(16):
            if(random.choice([True, False])):
                A[i,j]=random.random()
    return A

def cross(A):
    for i in range(len(A)):
        for j in range(16):
            if(random.choice([True, False])):
                A[i,j]+=A[(i+3)%6,j]
                if(A[i,j]>1):
                    A[i,j]-=1
    return A

def cross2(A):
    for i in range(len(A)):
        for j in range(16):
            if(random.choice([True, False])):
                A[i,j]+=A[(i+3)%6,j]
            else:
                A[i,j]-=A[(i+3)%6,j]
            if(A[i,j]>1):
                A[i,j]-=1
            if(A[i,j]<0):
                A[i,j]+=1
    return A
        

from numpy import random

#Genetic Algorithm

def genetic_algorithm(X_test=None,Y_test=None,iterations=1):
    print("GA Started Iterations",iterations)
    thresholds=random.randint(100, size=(10,16))/100
    f1_best=0
    threshold_best=thresholds[0]
    f1=np.zeros((10),np.float32)
    for i in range(iterations):
        for j in range(len(thresholds)):
            f1[j]=compute_f1(threshold=thresholds[j],X_test=X_test,Y_test=Y_test)
        thresholds,f1=sort(thresholds,f1)
        thresholds[8:10]=mutate(thresholds[6:8])
        thresholds[0:6]=cross2(thresholds[0:6])
        print(thresholds,f1)
    return thresholds,f1

thresholds,f1=genetic_algorithm(X_test=test.iloc[:,1:17],Y_test=test['Actual'],iterations=5)
