import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataset = pd.read_csv("C:/Users/Win10/Desktop/ML project/datasets/pulsar_stars.csv") #use location of your dataset.
print(dataset) # prints the dataset 
X1=dataset.drop('target_class',axis=1)
Y=dataset['target_class']
X=(X1-X1.mean())/(X1.max()-X1.min())
X=X.T
#Y = Y.values.reshape(-1,1)# This will reshape Y as a column vector. conversally reshape(1,-1) will reshape an array as row vector
Y=np.array([Y])
def sigmoid(Z):
    return(1/(1+np.exp(-Z)))

def mse(y_pred,target):
    m=17898
    J=(1/(2*m))*np.sum(np.square(y_pred-target))
    return J
    #return -np.mean((target*np.log(y_pred)+(1-target)*np.log(1-y_pred)))

def predict(X_test):
    preds = []
    for i in sigmoid(np.dot(X_test, W) + b):
        if i>0.5:
            preds.append(1)
        else:
            preds.append(0)
    return preds

  #shape[0] gives the number of rows and shape[1] gives the number of columns(i.e. features)

np.random.seed(0)
lenw=X.shape[0]
W=np.random.randn(1,lenw)
#W=(0.2,0.3,0.1,0.4,0.4,0.2,0.5,0.6)
#W = np.random.uniform(0,1,size=(X.shape[1],8))
b=0.5
def b_prop(X,Y,z):
    m=Y.shape[1]
    dz=(1/m)*(z-Y)
    dw=np.dot(dz,X.T)
    db=np.sum(dz)
    return dw,db
#linear_regression_model(X,Y,0.8,20)
def grad_desc(w,b,dw,db,learning_rate):
    W=W-learning_rate*dw
    b=b-learning_rate*db
    return w,b

for i in range(100):
    Es=[]
    Z = np.dot(W,X) + b
    Y_output = sigmoid(Z)
    E = mse(Y_output,Y)
    dw,db=b_prop(X,Y,z)
    W,b=grad_desc(W,b,dw,db,learning_rate)
    if i%10==0:
       Es.append(E)
       print('Training cost'+str(E))
    grad= Y_output - Y
    grad_weight= np.dot(X.T,grad)/X.shape[0]
    grad_bias = np.average(grad)
    W=W-.01*grad_weight
    b=b-.01*grad_bias
plt.plot(E)
plt.xlabel('Iterations(per tens)')
plt.ylabel('Training cost')
plt.title('Plot b='+str(b))
Y_test = predict(X_test=[20,32,11,46,78,56,32,65])
print(Y_test)
