import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv("C:/Users/Win10/Desktop/ML project/datasets/heart.csv")
X1=data.drop('target',axis=1)
Y=data['target']
X=(X1-X1.mean())/(X1.max()-X1.min())
X=X.T
data_mean = np.mean(X)
data_center = X-data_mean
cov_matrix = np.cov(data_center)
eigenval, eigenvec = np.linalg.eig(cov_matrix)
significance = [np.abs(i)/np.sum(eigenval) for i in eigenval]
print(significance)
plt.figure()
plt.plot(np.cumsum(significance))
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)') #for each component
plt.title('Heart Dataset Explained Variance')
plt.show()
