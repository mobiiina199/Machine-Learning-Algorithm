import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import scale
from scipy.cluster.hierarchy import dendrogram, linkage,fcluster

df=pd.read_csv('D:/Laptop backup 2/course work2/course works2/Data Science program/files/knurls2.csv')

X=df.iloc[:,1:]
#clusterer = linkage(X, 'average',metric='correlation')
clusterer = linkage(X, 'average',metric='Euclidean')
'''
A 4 by (n-1) matrix Z is returned. At the i-th iteration, clusters with 
indices Z[i, 0] and Z[i, 1] are combined to form cluster n + i. A cluster with an
index less than n corresponds to one of the n original observations. 
The distance between clusters Z[i, 0] and Z[i, 1] is given by Z[i, 2]. 
The fourth value Z[i, 3] represents the number of original observations in the
newly formed cluster.
'''
dendrogram(clusterer)
cluster_labels = fcluster(clusterer,3,criterion='maxclust')  
#for correlation use this line:        
#cluster_labels = fcluster(clusterer,3,criterion='maxclust')  

X['labels']=cluster_labels


fig,ax=plt.subplots(1)
ax.plot(X.loc[X['labels']==1,:].T,c='b',lw=0.3,alpha=0.4)


ax.plot(X.loc[X['labels']==2,:].T,lw=0.5,c='red',alpha=0.2)

ax.plot(X.loc[X['labels']==3,:].T,c='g',linestyle='dashdot',lw=0.5,alpha=0.3)
#'''
ax.set_yticklabels([])
ax.set_xticklabels([])
ax.tick_params(length=0)

#'''

