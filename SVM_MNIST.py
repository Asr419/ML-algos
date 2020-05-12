import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics
import seaborn as sns

data=pd.read_csv("C:/Users/Win10/Desktop/ML project/datasets/mnist_train.csv")
X=data.drop('label',axis=1)
Y=data['label']
x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=0.2)
print(X.shape)
print(data['label'].value_counts())
mdl = SVC(C=500, kernel='rbf', random_state=2019, gamma="scale", verbose=True)
mdl.fit(x_train, y_train)
predicted = mdl.predict(x_val)
print("accuracy", metrics.accuracy_score(y_val, predicted))
sns.heatmap(pd.DataFrame(metrics.confusion_matrix(y_val, predicted)), annot=True, cmap="YlGn", fmt='g')
plt.title("SVM Confusion Matrix rbf")
plt.show()
