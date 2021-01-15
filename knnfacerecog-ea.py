import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier#knn
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits

faces = datasets.fetch_olivetti_faces

faces.images.shape# neshan dahandeh 400 ta ax darim ke 64 dar 64
faces.images[0].shape
X=faces.data
X.shape#400 ta ax tabdil kardeh be ye bordar n=400 va p=4096

plt.imshow(faces.images[0],cmap='gray')#mishe ax azash gereft
faces.target# 40 nafar az har kodum 10 ta ax ke shomareha neshon mide
#'''
fig = plt.figure(figsize=(8, 6))
for i in range(15):
    ax = fig.add_subplot(3, 5, i + 1, xticks=[], yticks=[])
    ax.imshow(faces.images[i], cmap='gray')
#'''data train va test joda mikonim faces. data matrix x va faces.target mishe motaghayer pasokh ya hanom y 
   
X_train, X_test, y_train, y_test = train_test_split(faces.data,faces.target, test_size=0.2)
pca = PCA(whiten=True)# whiten=True khodesh scale mikone x ha ro
pca.fit(X_train)
X_train.shape
X_test.shape

X_train_pca = pca.transform(X_train)[:,:10]#pca.transform(X_train) pc-scores ha ro mideh  10 yani 
X_test_pca = pca.transform(X_test)[:,:10]

Knno=KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)# n_neighbors hamun K va metric noe fasele va p=2 yani fasele oghlodusi va p=1 fasele manhatan
Knnpca=KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)
Knno.fit(X_train,y_train)#moghe fit kardan data train midim be classifier ha 
Knnpca.fit(X_train_pca,y_train)
#confusion_matrix(y_test,Knno.predict(X_test))
Knno.predict(X_test) 
Knno.predict(X_test_pca)
Knno.score(X_test,y_test)#yani adadi ke mideh tedad mavaredi ke dorost tashkhis dadeh shodeh be meghdar in adad mishe ke har chi bishtar bashe behtare
print('Knn-PCA Prediction Accuracy:',Knnpca.score(X_test_pca,y_test))
print('Origial Knn Prediction Accuracy:',Knno.score(X_test,y_test))
#'''

