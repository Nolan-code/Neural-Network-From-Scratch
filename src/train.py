import numpy as np
from .utils import split_data, scale_features, accuracy
from .model import gradient_descent, y_pred

def NN(df,n_iters,lr,target_name):
    X_train_1,y_train,X_val_1,y_val,X_test_1,y_test = split_data(df,target_name)    #split the data into training/validation/test set
    X_val_1, X_test_1 = np.array(X_val_1), np.array(X_test_1)

    X_train, means, stds = scale_features(X_train_1)       #normalize the trainig set
    X_val, X_test = (X_val_1 - means)/stds, (X_test_1 - means)/stds      #normalize the 2 other set with the mean and variance computed for the training set

    W2,b2,W1,b1,Losses = gradient_descent(X_train,y_train,n_iters,lr)    #fit the model with training set

    X_val = np.array(X_val)
    y_val = np.array(y_val)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    y_preds_val = y_pred(X_val,W2,b2,W1,b1)         #predictions for val set
    accuracy_val = accuracy(y_val,y_preds_val)

    y_preds_test = y_pred(X_test,W2,b2,W1,b1)        #predictions for test set
    accuracy_test = accuracy(y_test,y_preds_test)

    return accuracy_val,accuracy_test,Losses
    X_train, means, stds = scale_features(X_train_1)       #normalize the trainig set
    X_val, X_test = (X_val_1 - means)/stds, (X_test_1 - means)/stds      #normalize the 2 other set with the mean and variance computed for the training set

    W2,b2,W1,b1,Losses = gradient_descent(X_train,y_train,n_iters,lr)    #fit the model with training set

    X_val = np.array(X_val)
    y_val = np.array(y_val)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    y_preds_val = y_pred(X_val,W2,b2,W1,b1)         #predictions for val set
    accuracy_val = accuracy(y_val,y_preds_val)

    y_preds_test = y_pred(X_test,W2,b2,W1,b1)        #predictions for test set
    accuracy_test = accuracy(y_test,y_preds_test)

    return accuracy_val,accuracy_test,Losses