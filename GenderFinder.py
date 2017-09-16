#Prediction using Decision Tree Algorithm
from sklearn import tree

#[height, weight, shoe size]
X = [[181,80,44], [177,70,43], [160,60,38], [154,54,37],
	[166,65,40], [190,90,47], [175,64,39], [177,70,40], 
	[159,55,37], [171,75,42], [181,85,43]]

Y = ['male', 'female', 'female', 'female', 'male', 'male', 'male',
	'female', 'male', 'female', 'male']

clf = tree.DecisionTreeClassifier()

print (type(clf))

clf = clf.fit(X,Y)

prediction = clf.predict([[190,70,43]])

print (prediction)

#Prediction using K-Nearest-Neighbor Algorithm
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X,Y)

prediction = knn.predict([[190,70,43]])

print (prediction)

#Prediction using LogisticRegression Algorithm
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()

logreg.fit(X,Y)

predict = logreg.predict([[190,70,43]])

print (prediction)
