import numpy as np
from sklearn import datasets

from sklearn.neighbors import KNeighborsClassifier

# Create arrays for the features and the response variable
iris = datasets.load_iris()
# np.random.shuffle(iris)
# iris = iris.ra
X = iris.data[1:]
y = iris.target[1:]
z = [X[0]]
print(z)

# Number of Unique Flower classes
print(np.unique(y))

# y = df['party'].values
# X = df.drop('party', axis=1).values

# Create a k-NN classifier with 6 neighbors
knn = KNeighborsClassifier(n_neighbors=6)

# Fit the classifier to the data
knn.fit(X, y)

# Predict the labels for the training data X
y_pred = knn.predict(X)
print y_pred

# Predict and print the label for the new data point X_new
new_prediction = knn.predict(z)
print("Prediction: {}".format(new_prediction))
