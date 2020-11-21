from time import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import os

df = pd.read_csv("train_data_clensed_new.csv")
col_names = list(range(7,81))
col_names = [2,3] + col_names
X = df.iloc[:, col_names]
Y = df.iloc[:, 6]
X_train = X
y_train = Y

data = scale(X_train)
labels = y_train

# #############################################################################
# Visualize the results on PCA-reduced data
# #############################################################################

reduced_data = PCA(n_components=2).fit_transform(data)
print(data.shape)
print(reduced_data.shape)

plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, edgecolor='none', alpha=1, cmap=plt.cm.get_cmap('Spectral', 10))
plt.colorbar()
sv_path = "PCA_Dim_Reduction" + ".pdf"
save_path = os.path.join('.', sv_path)
plt.savefig(save_path)
