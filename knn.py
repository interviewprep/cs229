#Covid mortality ML using KNN classifier algorithm

import pandas as pd
import numpy  as np
import graphviz 
import os
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv("train_data.csv")
df.head(0)
col_names = list(range(7,81))
col_names = [2,3] + col_names
X = df.iloc[:, col_names]
Y = df.iloc[:, 6]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20)

scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
