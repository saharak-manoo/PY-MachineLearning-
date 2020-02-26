from sklearn import tree

features = [[120, 5], [110, 5], [180, 2], [163, 12], [93, 2]]
print(features)

labels = [
    'Eco car => B-Segment', 'Eco car => B-Segment', 'Van', 'Van', 'Motorcycle'
]
print(labels)

classifier = tree.DecisionTreeClassifier()
classifier = classifier.fit(features, labels)

print(classifier.predict([[60, 2]]))
