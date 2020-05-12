import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
plt.figure(figsize=(24,12))

plt.suptitle("Confusion Matrixes",fontsize=24)


data=pd.read_csv("C:/Users/Win10/Desktop/ML project/datasets/pulsar_stars.csv")
X1=data.drop('target_class',axis=1)
Y=data['target_class']
X=(X1-X1.mean())/(X1.max()-X1.min())
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.2)
from sklearn.svm import SVC
#poly
svm_model = SVC(random_state=42,C=250,gamma=1.6,kernel="poly",probability=True)

svm_model.fit(x_train,y_train)

y_head_svm = svm_model.predict(x_test)

svm_score = svm_model.score(x_test,y_test)
print(svm_score)

cm_svm = confusion_matrix(y_test,y_head_svm)
plt.subplot(2,3,6)
plt.title("SVM Confusion Matrix polynomial(deg=3)")
sns.heatmap(cm_svm,cbar=False,annot=True,cmap="CMRmap_r",fmt="d")
#poly5
svm_model4 = SVC(random_state=42,C=250,degree=5,gamma=1.6,kernel="poly",probability=True)

svm_model4.fit(x_train,y_train)

y_head_svm4 = svm_model4.predict(x_test)

svm_score4 = svm_model4.score(x_test,y_test)
print(svm_score4)

cm_svm4 = confusion_matrix(y_test,y_head_svm4)
plt.subplot(2,3,5)
plt.title("SVM Confusion Matrix polynomial(deg=5)")
sns.heatmap(cm_svm4,cbar=False,annot=True,cmap="CMRmap_r",fmt="d")
#poly6
svm_model5 = SVC(random_state=42,C=250,degree=6,gamma=1.6,kernel="poly",probability=True)

svm_model5.fit(x_train,y_train)

y_head_svm5 = svm_model5.predict(x_test)

svm_score5 = svm_model5.score(x_test,y_test)
print(svm_score5)

cm_svm5 = confusion_matrix(y_test,y_head_svm5)
plt.subplot(2,3,4)
plt.title("SVM Confusion Matrix polynomial(deg=6)")
sns.heatmap(cm_svm5,cbar=False,annot=True,cmap="CMRmap_r",fmt="d")
#rbf
svm_model1 = SVC(random_state=42,C=250,gamma=1.6,kernel="rbf",probability=True)

svm_model1.fit(x_train,y_train)

y_head_svm1 = svm_model1.predict(x_test)

svm_score1 = svm_model1.score(x_test,y_test)
print(svm_score1)
cm_svm1 = confusion_matrix(y_test,y_head_svm1)
plt.subplot(2,3,2)
plt.title("SVM Confusion Matrix radial basis")
sns.heatmap(cm_svm1,cbar=False,annot=True,cmap="CMRmap_r",fmt="d")
#sigmoid
svm_model2 = SVC(random_state=42,C=250,gamma=1.6,kernel="sigmoid",probability=True)

svm_model2.fit(x_train,y_train)

y_head_svm2 = svm_model2.predict(x_test)

svm_score2 = svm_model2.score(x_test,y_test)
print(svm_score2)
cm_svm2 = confusion_matrix(y_test,y_head_svm2)
plt.subplot(2,3,3)
plt.title("SVM Confusion Matrix sigmoid")
sns.heatmap(cm_svm2,cbar=False,annot=True,cmap="CMRmap_r",fmt="d")
#linear
svm_model3 = SVC(random_state=42,kernel="linear",probability=True)

svm_model3.fit(x_train,y_train)

y_head_svm3 = svm_model3.predict(x_test)

svm_score3 = svm_model1.score(x_test,y_test)
print(svm_score3)
cm_svm3 = confusion_matrix(y_test,y_head_svm3)
plt.subplot(2,3,1)
plt.title("SVM Confusion Matrix linear")
sns.heatmap(cm_svm3,cbar=False,annot=True,cmap="CMRmap_r",fmt="d")
plt.show()
