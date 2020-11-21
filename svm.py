from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib
import numpy as np 
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import plot_confusion_matrix

from sklearn import metrics

def main():
	print('Running SVM')
	df = pd.read_csv("../Data/train_data_clensed_by_patient.csv")
	df.head(0)

	col_names = list(range(4, 77))
	col_names = [1,2] + col_names
	X = df.iloc[:, col_names]
	# X = df.loc[:, ['lactate_dehydrogenase', 'hypersensitive_c_reactive_protein', 'lymphocyte_count']]
	Y = df.iloc[:, 3]
	class_names= ['alive', 'deceased']
	# print(X)

	X_train, X_test, y_train, y_test = train_test_split(X, Y, shuffle=False, test_size=0.2)
	clf1 = make_pipeline(StandardScaler(), SVC(kernel='sigmoid', gamma='auto'))
	clf1.fit(X_train, y_train)
	y_pred = clf1.predict(X_test)
	print(metrics.accuracy_score(y_test, y_pred))

	np.set_printoptions(precision=2)
	title = 'Confustion Matrix Normalized Running SVM'
	disp = plot_confusion_matrix(clf1, X_test, y_test, display_labels=class_names, cmap=plt.cm.Blues, normalize='true')
	disp.ax_.set_title(title)
	print(title)
	print(disp.confusion_matrix)

	plt.show()
	# fig = plt.figure()
	# ax = fig.add_subplot(111, projection='3d')

	# x = df.loc[:, ['lactate_dehydrogenase']].values.tolist()
	# y = df.loc[:, ['hypersensitive_c_reactive_protein']].values.tolist()
	# z = df.loc[:, ['lymphocyte_count']].values.tolist()

	# label = Y.values.tolist()
	# colors = ['green', 'red']

	# ax.scatter(x, y, z, c=label, cmap=matplotlib.colors.ListedColormap(colors))

	# ax.set_xlabel('lactate_dehydrogenase')
	# ax.set_ylabel('hypersensitive_c_reactive_protein')
	# ax.set_zlabel('lymphocyte_count')

	# plt.show()
	# fig = plt.figure()
	# ax = fig.add_subplot(111, projection='3d')

	# x = X_test.iloc[:, 0].values.tolist()
	# y = X_test.iloc[:, 1].values.tolist()
	# z = X_test.iloc[:, 2].values.tolist()

	# label = y_test.values.tolist()
	# colors = ['green', 'red']

	# ax.scatter(x, y, z, c=label, cmap=matplotlib.colors.ListedColormap(colors))

	# ax.set_xlabel('lactate_dehydrogenase')
	# ax.set_ylabel('hypersensitive_c_reactive_protein')
	# ax.set_zlabel('lymphocyte_count')

	# plt.show()


	# fig = plt.figure()
	# ax = fig.add_subplot(111, projection='3d')

	# x = X_test.iloc[:, 0].values.tolist()
	# y = X_test.iloc[:, 1].values.tolist()
	# z = X_test.iloc[:, 2].values.tolist()

	# label = y_pred
	# colors = ['blue', 'red']

	# ax.scatter(x, y, z, c=label, cmap=matplotlib.colors.ListedColormap(colors))

	# ax.set_xlabel('lactate_dehydrogenase')
	# ax.set_ylabel('hypersensitive_c_reactive_protein')
	# ax.set_zlabel('lymphocyte_count')

	# plt.show()
	# print(X)



	clf = make_pipeline(StandardScaler(), SVC(kernel='sigmoid', gamma='auto'))
	cv = ShuffleSplit(n_splits=10, test_size=0.2, train_size=None)
	score = 0
	accuracy = 0
	for i in range(10):
		score += cross_val_score(clf, X, Y, cv=cv, scoring='f1').mean()
		accuracy += cross_val_score(clf, X, Y, cv=cv, scoring='accuracy').mean()
	score = score/10
	accuracy = accuracy/10

	print('Acu: ',accuracy)
	# print(accuracy.mean())
	print('f1: ',score)
	# print(scores.mean())



	# kf = StratifiedKFold(n_splits=10, random_state=None, shuffle=True)
	# scores = cross_val_score(clf, X, Y, cv=kf)

	# print(scores)
	# print(scores.mean())


if __name__ == "__main__":
	main()


