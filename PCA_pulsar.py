import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

data=pd.read_csv("C:/Users/Win10/Desktop/ML project/datasets/pulsar_stars.csv")
X1=data.drop('target_class',axis=1)
Y=data['target_class']
X=(X1-X1.mean())/(X1.max()-X1.min())
X=X.T
Y=data['target_class']
data_mean = np.mean(X)
data_center = X-data_mean
cov_matrix = np.cov(data_center)
eigenval, eigenvec = np.linalg.eig(cov_matrix)
significance = [np.abs(i)/np.sum(eigenval) for i in eigenval]
plt.figure()
plt.plot(np.cumsum(significance))
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)') #for each component
plt.title('Pulsar Dataset Explained Variance')
plt.show()
