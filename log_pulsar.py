import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataset = pd.read_csv("C:/Users/Win10/Desktop/ML project/datasets/pulsar_stars.csv") #use location of your dataset.
print(dataset) # prints the dataset 
X1=dataset.drop('target_class',axis=1)
Y=dataset['target_class']
X=(X1-X1.mean())/(X1.max()-X1.min())
X=X.T
Y=np.array([Y])
def sigmoid(Z):
    return(1/(1+np.exp(-Z)))
def initialize_parameter(lenw):
    w=np.random.randn(1,lenw)
    #w=np.zeros((1,lenw))
    b=0
    return w,b
def cost_function(z,y,w):
    m=17898
    J=(1/(2*m))*np.sum(np.square(z-y))
    return J
def f_prop(X,w,b):
    z=np.dot(w,X)+b
    return z
def b_prop(X,Y,z):
    m=Y.shape[1]
    dz=(1/m)*(z-Y)
    dw=np.dot(dz,X.T)
    db=np.sum(dz)
    return dw,db
#linear_regression_model(X,Y,0.8,20)
def grad_desc(w,b,dw,db,learning_rate):
    w=w-learning_rate*dw
    b=b-learning_rate*db
    return w,b
def logistic_regression(X,Y,learning_rate,epochs):
    lenw=X.shape[0]
    w,b=initialize_parameter(lenw)
    costs_train=[]
    m_train=524
    for i in range(1,epochs+1):
        z=f_prop(X,w,b)
        z1=sigmoid(z)
        cost_train=cost_function(z1,Y,w)
        dw,db=b_prop(X,Y,z1)
        w,b=grad_desc(w,b,dw,db,learning_rate)
        if i%10==0:
            costs_train.append(cost_train)
            MAE_train=(1/m_train)*np.sum(np.abs(z-Y))
            print('Epochs'+str(i)+'/'+str(epochs)+':')
            print('Training cost' +str(cost_train))
            print('Training MAE' +str(MAE_train))
    plt.plot(costs_train)
    plt.xlabel('Iterations(per tens)')
    plt.ylabel('Training cost')
    plt.title('Learning rate'+str(learning_rate))
    plt.show()
