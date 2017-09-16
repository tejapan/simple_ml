from sklearn.datasets import load_iris
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

prediction = knn.predict([[3,5,4,2]])

print (prediction)
