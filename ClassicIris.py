from sklearn.datasets import load_iris

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