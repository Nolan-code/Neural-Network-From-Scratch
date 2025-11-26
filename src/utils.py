import numpy as np
import copy


def split_data(df,target_name): # take the whole df and target's name and return the split data in an adapted form for the gradient_descent function
  df=copy.deepcopy(df)

  train, val, test = np.split(df.sample(frac=1),[int(0.6*len(df)),int(0.8*len(df))]) #split at 0.6 of the df lenght and 0.8 of the df lenght

  df_1=train.drop([target_name],axis=1)
  X_train = df_1.values.tolist()         #list of list of the features for train
  y_train = train[target_name].tolist()   #list of target for train

  df_2=val.drop([target_name],axis=1)
  X_val = df_2.values.tolist()          #list of list of the features for val
  y_val = val[target_name].tolist()    #list of target for val

  df_3=test.drop([target_name],axis=1)
  X_test = df_3.values.tolist()          #list of list of the features for test
  y_test = test[target_name].tolist()    #list of target for test

  return X_train,y_train,X_val,y_val,X_test,y_test

def scale_features(X):    #normalize features 
    X = np.array(X)
    means = X.mean(axis=0)
    stds = X.std(axis=0)
    return (X - means) / stds, means, stds

def accuracy(y,y_preds):
  return np.mean(y == y_preds)