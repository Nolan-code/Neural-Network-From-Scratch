import numpy as np

def Relu(z):  #Relu fonction
  return np.maximum(z,0)

def softmax(z):   #softmax function
    z = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def compute_y_onehot(y):             #instead of having the class (0 or 1 or 2) for every row, create a list where the class is the index of the 1 
  y_onehot = np.zeros((len(y),3))    #in this list (all the other term are set to 0)
  for k in range(len(y)):
    y_onehot[k][y[k]] = 1
  return y_onehot

def y_pred(X,W2,b2,W1,b1):   #return the prediction based on the input X and the weight and the bias given
  Z1 = X @ W1 + b1           
  A1 = Relu(Z1)               
 
  Z2 = A1 @ W2 + b2          
  A2 = softmax(Z2)

  class_ = np.argmax(A2, axis=1) 
  return class_

def gradient_descent(X,y,n_iters,learning_rate):
  n = len(y)   #number of example

  X = np.array(X)
  y = np.array(y)

  n_features = X.shape[1]  #number of features

  W1 = np.random.rand(n_features,64)   #weight of the first layer
  b1 = np.random.rand(1,64)            #bias of the first layer

  W2 = np.random.rand(64,3)        #weight of the second layer
  b2 = np.random.rand(1,3)         #bias of the second layer

  y_onehot = compute_y_onehot(y)    #convert to (len(y),3) matrix with a 1 for at the indice of the class that the first example is part of (3 for 3 classes)
  Losses = []

  for k in range(n_iters):
    Z1 = X @ W1 + b1            #linear part of the first layer
    A1 = Relu(Z1)               #Activation funciton of the first layer:Relu
 
    Z2 = A1 @ W2 + b2          #linear part of the second layer
    A2 = softmax(Z2)           #Activation funciton of the first layer:softmax

    Loss = - np.sum(y_onehot*np.log(A2))/n     #cross-entropy loss
    Losses.append(Loss)

    dZ2 = (1/n)*(A2 - y_onehot)     # shape (n,3)       #d? is the derivative of the loss fonction with respect to ?
    #gradients layer 2
    dW2 = A1.T @ dZ2      # shape (64, C)   #computed with the chain rule of derivation
    db2 = np.sum(dZ2, axis=0, keepdims=True)  # shape (1, C)

    dA1 = dZ2 @ W2.T   # shape (n, 64)
    dZ1 = dA1 * (Z1 > 0)  # shape (N, 64)
    #gradients layer 1
    dW1 = X.T @ dZ1      # shape (n_features, 64)
    db1 = np.sum(dZ1, axis=0, keepdims=True)  # shape (1, 64)

    # gradient descent
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

  return W2,b2,W1,b1,Losses