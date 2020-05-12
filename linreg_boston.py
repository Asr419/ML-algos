
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("C:/Users/Win10/Desktop/ML project/datasets/boston.csv") #read the data
#X=dataset.drop('MEDV',axis=1)
#y=dataset['MEDV']
#ones = np.ones([X.shape[0],1])
#X = np.concatenate((ones,X),axis=1)
X = dataset.iloc[:,0:13]
ones = np.ones([X.shape[0],1])
X = np.concatenate((ones,X),axis=1)

y = dataset.iloc[:,13:14].values #.values converts it from pandas.core.frame.DataFrame to numpy.ndarray


#y = my_data.iloc[:,2:3].values #.values converts it from pandas.core.frame.DataFrame to numpy.ndarray
theta = np.zeros([1,14])

#set hyper parameters
alpha = 0.01
iters = 1000

def computeCost(X,y,theta):
    tobesummed = np.power(((X @ theta.T)-y),2)
    return np.sum(tobesummed)/(2 * len(X))


#gradient descent
def gradientDescent(X,y,theta,iters,alpha):
    cost = np.zeros(iters)
    for i in range(iters):
        theta = theta - (alpha/len(X)) * np.sum((X @ theta.T - y)*X, axis=0)
        cost[i] = computeCost(X, y, theta)
    
    return theta,cost

#running the gd and cost function
g,cost = gradientDescent(X,y,theta,iters,alpha)
print(g)

finalCost = computeCost(X,y,g)
print(finalCost)
