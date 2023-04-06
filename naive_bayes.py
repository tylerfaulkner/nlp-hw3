from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
import numpy as np

# Create a Gaussian Naive Bayes classifier
clf = GaussianNB()

# Load the training data
raw_training_data = open('trainingSet.txt', 'r').readlines()

# Split the labels from the text
labels = [x.split(' ', 1)[0] for x in raw_training_data]
training_data = [x.split(' ', 1)[1][:-1] for x in raw_training_data]

#print(labels)
#print(training_data)

# Load the test data
raw_test_data = open('testSet.txt', 'r').readlines()
test_labels = [x.split(' ', 1)[0] for x in raw_test_data]
test_data = [x.split(' ', 1)[1][:-1] for x in raw_test_data]

#print(test_data)

#Bag of words
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
training_data = vectorizer.fit_transform(training_data)

# Add-1 smoothing
training_data = training_data.toarray()
training_data = training_data + 1

# Fit the classifier to the training data
clf.fit(training_data, labels)

# Predict the labels of the test data
# Convert the test data to a bag of words
test_data = vectorizer.transform(test_data).toarray()
pred = clf.predict(test_data)

print(pred)

# Print the accuracy of the classifier
print(clf.score(test_data, test_labels))

# Print the precision
print(metrics.precision_score(test_labels, pred, average='macro'))

# Print the recall
print(metrics.recall_score(test_labels, pred, average='macro'))

#Print the f-score
print(metrics.f1_score(test_labels, pred, average='macro'))

