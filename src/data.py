import pandas as pd

df = pd.read_csv("iris.data")
df.head()

df["class"]=df["class"].replace("Iris-setosa",2)
df["class"]=df["class"].replace("Iris-versicolor",1)
df["class"]=df["class"].replace("Iris-virginica",0)
df.head()