import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.metrics import confusion_matrix,silhouette_samples, silhouette_score
import datetime
import matplotlib.cm as cm
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
from datetime import date
df = pd.read_csv('D:/Laptop backup 2/course work2/course works2/Data Science program/files/Marketnc.csv')

df.columns
np.sum(df.isna())
df.dropna(inplace=True)
df.describe()
df.dtypes

cancellations=df['InvoiceNo'].str.contains('C')
df=df.loc[cancellations!=True]

df['InvoiceDate']=df['InvoiceDate'].agg(pd.Timestamp)
today=pd.Timestamp('04/11/2012')
df['Days']=df['InvoiceDate'].agg(lambda x:(today-x).days)


Recency=df.groupby('CustomerID').Days.min()
Frequency=df.groupby('CustomerID').InvoiceNo.unique().agg(len)
df['Cost']=df['Quantity']*df['UnitPrice']
Monetary=df.groupby('CustomerID').sum().loc[:,'Cost']

X=pd.concat((pd.DataFrame(Recency),pd.DataFrame(Frequency),pd.DataFrame(Monetary)),axis=1)

X.columns=['R','F','M']
n_clusters=3
kmeans = KMeans(n_clusters=n_clusters).fit(X)

pd.DataFrame(kmeans.cluster_centers_,columns=X.columns)


labels=kmeans.labels_
average_silhouette=silhouette_score(X,labels)
sample_silhouette_values=silhouette_samples(X,labels)
