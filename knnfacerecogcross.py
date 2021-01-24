from sklearn.model_selection import cross_val_score
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA

faces = datasets.fetch_olivetti_faces()

X_train, X_test, y_train, y_test = train_test_split(faces.data,faces.target, test_size=0.2)
pca = PCA(whiten=True)
pca.fit(X_train)
counter=-1
comp=np.arange(10,100,10)
nneighbors=np.arange(1,10)
cv=10
#Cross validation to find the best number of neighbors
 
for k in nneighbors:
 counter=counter+1    
 Knno=KNeighborsClassifier(n_neighbors=k,metric='minkowski',p=2)
 cvscores[counter,:]=np.array([k,np.mean(cross_val_score(Knno, X_train, y_train, cv=cv))])
np.argmax(cvscores[:,1])
cvscores[np.argmax(cvscores[:,1]),:]

#Cross validation to find the best number of neighbors and number of PC scores
cvscorespca=np.empty((len(comp)*len(nneighbors),3))
counter=-1
for c in comp:
    X_train_pca = pca.transform(X_train)[:,:c]
    for k in nneighbors:
     counter=counter+1   
     #Knno=KNeighborsClassifier(n_neighbors=k,metric='minkowski',p=2)
     Knnpca=KNeighborsClassifier(n_neighbors=k,metric='minkowski',p=2)
     #cvscores[l]= np.mean(cross_val_score(Knno, X_train, y_train, cv=cv))
     cvscorespca[counter,:]=np.array([c,k,np.mean(cross_val_score(Knnpca, X_train_pca, y_train, cv=cv))])
np.argmax(cvscorespca[:,2])
cvscorespca[np.argmax(cvscorespca[:,2]),:]

#After cross validation, you can compare your methods
Knno=KNeighborsClassifier(n_neighbors=1,metric='minkowski',p=2)
Knnpca=KNeighborsClassifier(n_neighbors=1,metric='minkowski',p=2)
X_train_pca = pca.transform(X_train)[:,:20]
X_test_pca = pca.transform(X_test)[:,:20]
Knno.fit(X_train,y_train)
Knnpca.fit(X_train_pca,y_train)

print('Knn-PCA Prediction Accuracy:',Knnpca.score(X_test_pca,y_test))
print('Oeigial Knn Prediction Accuracy:',Knno.score(X_test,y_test))


