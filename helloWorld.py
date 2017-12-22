#Kgadi Kenneth Mphele
#11/1212017
#machine learning hello world equivalent example  using classification between oranges and apples .

from sklearn import tree
 #features : weight , texture 
 #label: orange or apple
#0 represents bumpy then one for smooth
features=[[140,1],[130,1],[150,0],[170,1]]
#0 represents apple and one for orange
labels=[0,0,1,1]

#function that uses features to predict the label(classifier)
classifier=tree.DecisionTreeClassifier()

#find patterns in data
classifier=classifier.fit(features,labels)

#predict a fruit that weigh 150g and its bumpy  using the classifier
#0 means an apple and one means it predicted an orange
print(classifier.predict([[150,0]]))

