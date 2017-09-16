from sklearn.datasets import load_iris

#Prediction using K-Nearest-Neighbors Algorithm
from sklearn.neighbors import KNeighborsClassifier

iris = load_iris()

print (type(iris))
print (iris.data)
print (iris.feature_names)
print (iris.target)
print (iris.target_names)

print (type(iris.data))
print (type(iris.target))

print (iris.data.shape)
print (iris.target.shape)

X = iris.data
y = iris.target

#tuning parameter n_neighbor = 1
knn = KNeighborsClassifier(n_neighbors=1)

print (knn)

knn.fit(X,y)

#predict method returns NumPy array
prediction = knn.predict([[3,5,4,2]])

print (prediction)

#predict method can take multiple observations as argument
X_new = [[3,5,4,2], [5,4,3,2]]

prediction = knn.predict(X_new)

print (prediction)

#Change the tuning parameter
knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X,y)

prediction = knn.predict(X_new)

print (prediction)

#Prediction using Logistic Regression Algorithm
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()

logreg.fit(X,y)

prediction = logreg.predict(X_new)

print (prediction)
