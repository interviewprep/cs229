import os
import numpy as np
import pandas as pd
import numpy  as np
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import ShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans

df = pd.read_csv("train_data_clensed_by_patient.csv")
df.head(0)

col_names = list(range(4,78))
#col_names = [ 1, 2] + col_names
X = df.iloc[:, col_names]
Y = df.iloc[:, 3]
X.head()
print(X,X.shape)
n_digits = len(np.unique(Y))
cv = ShuffleSplit(n_splits=10, test_size=0.2, train_size=None)

def decision_tree(score_type):
    clf = tree.DecisionTreeClassifier()
    scores = cross_val_score(clf, X, Y, cv=cv, scoring=score_type)
    #print  (scores)
    #print (scores.mean())
    return scores.mean()

#Random forest
def random_forest(score_type):
    rf = RandomForestClassifier(max_depth=2, random_state=0, max_features="sqrt")
    scores = cross_val_score(rf, X, Y, cv=cv, scoring=score_type)
    #print  (scores)
    #print (scores.mean())
    return scores.mean()

def bagging_class(score_type):
    bc = BaggingClassifier(max_samples=1.0)
    scores = cross_val_score(bc, X, Y, cv=cv, scoring=score_type)
    #print  (scores)
    #print (scores.mean())
    return scores.mean()

def logistic_reg(score_type):
    clf = LogisticRegression(random_state=0, penalty = 'l2', C=1.0)
    scores = cross_val_score(clf, X, Y, cv=cv, scoring=score_type)
    #print (scores)
    return scores.mean()

def naive_bayes(score_type):
    gnb = GaussianNB()
    scores = cross_val_score(gnb, X, Y, cv=cv, scoring=score_type)
    #print (scores)
    return scores.mean()
def sgb(score_type,learning_rate=0.25,max_features=76):
    clf = GradientBoostingClassifier(n_estimators=40, learning_rate=learning_rate, max_features=max_features, max_depth=2, random_state=0)
    scores = cross_val_score(clf, X, Y, cv=cv, scoring=score_type)
    return scores.mean()
def knn(score_type):
    knn = KNeighborsClassifier(n_neighbors=7)
    scores = cross_val_score(knn, X, Y, cv=cv, scoring=score_type)
    return scores.mean()

#All fields!!
#calculate average score for 10 calls (each call is again a 10 fold CV)

#X = df.loc[:, ['lactate_dehydrogenase', 'hypersensitive_c_reactive_protein', 'lymphocyte_count']]
score = 0
for i in range(0,10):
    score = score + decision_tree('accuracy')
print ("final score DecisionTreeClassifier all features:",score/10)

score = 0
for i in range(0,10):
    score = score + sgb('accuracy',max_features=74)
print ("final score sgb all features:",score/10)

score = 0
for i in range(0,10):
    score = score + knn('accuracy')
print ("final score knn all features:",score/10)

score = 0
for i in range(0,10):
    score = score + random_forest('accuracy')
print ("final score random_forest all features:",score/10)

score = 0
for i in range(0,10):
    score = score + logistic_reg('accuracy')
print ("final score logistic_reg all features:",score/10)

score = 0
for i in range(0,10):
    score = score + bagging_class('accuracy')
print ("final score bagging_class all features:",score/10)

score = 0
for i in range(0,10):
    score = score + naive_bayes('accuracy')
print ("final score naive_bayes all features:",score/10)


