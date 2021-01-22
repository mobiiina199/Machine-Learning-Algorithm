import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
x=np.arange(0,7,0.01)
f=np.sin(x)
y=np.sin(x)+np.random.normal(0,0.6,len(x))
T=f

plt.scatter(x,y,c='b',alpha=0.1)
plt.plot(sorted(x),f,c='black',label='True Function')


from sklearn.linear_model import LinearRegression
yhatpt=[]
yhatptr=[]
lr=LinearRegression()
lr.fit(np.array(x).reshape(-1,1),np.array(y).reshape(-1,1))

yhat=lr.predict(np.array(x).reshape(-1,1))
plt.plot(x,yhat,c='green',alpha=0.5,label='Linear Regression')

m=4
lrp=np.polyfit(x,y,m)
p=np.poly1d(lrp)
yhatp=p(sorted(x))
plt.plot(sorted(x),yhatp,c='deeppink',label='Polynomial Regression')

pt=np.poly1d(lrp)
plt.xlabel('x')
plt.ylabel('y')

plt.arrow(x[229],T[229],0.6,0.6,ls='dashed',color='black',alpha=0.5,head_width=0.06)
plt.text(x[229]+0.6+0.06,T[229]+0.6+0.01,'True Function')

plt.arrow(x[460],yhat[460],0.6,0.6,ls='dashed',color='green',alpha=0.5,head_width=0.06)
plt.text(x[460]-0.1,yhat[460]+0.6+0.1,'Linear Regression',color='green')

plt.arrow(x[300],yhatp[300],-0.6,-0.6,ls='dashed',color='r',alpha=0.5,head_width=0.06)
plt.text(x[300]-0.6-0.6,yhat[300]-0.6-0.3,'Polynomial fit',color='r')

