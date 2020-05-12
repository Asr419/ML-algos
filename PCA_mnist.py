import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv("C:/Users/Win10/Desktop/ML project/datasets/mnist_train.csv")
X=data.drop('label',axis=1)
Y=data['label']
X=X.T
data_mean = np.mean(X)
data_center = X-data_mean
cov_matrix = np.cov(data_center)
eigenval, eigenvec = np.linalg.eig(cov_matrix)
significance = [np.abs(i)/np.sum(eigenval) for i in eigenval]
plt.figure()
plt.plot(np.cumsum(significance))
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)') #for each component
plt.title('MNIST Dataset Explained Variance')
plt.show()
