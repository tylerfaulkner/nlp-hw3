from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn import metrics
import numpy as np
import os

# Load the training data
print('Loading training data...')
data_file = '/media/faulknert/Storage/Amazon Reviews/train_data.txt'
label_file = '/media/faulknert/Storage/Amazon Reviews/train_labels.txt'

input_type = 'content'
raw_training_data = open('/media/faulknert/Storage/Amazon Reviews/train.ft.txt', 'r').readlines()
# Split the labels from the text
labels = [x.split(' ', 1)[0] for x in raw_training_data]
raw_training_data = [x.split(' ', 1)[1][:-1] for x in raw_training_data]


# Load the test data
print('Loading test data...')
raw_test_data = open('/media/faulknert/Storage/Amazon Reviews/test.ft.txt', 'r').readlines()
test_labels = [x.split(' ', 1)[0] for x in raw_test_data]
test_data = [x.split(' ', 1)[1][:-1] for x in raw_test_data]

#print(test_data)

#Bag of words
print('Creating bag of words...')
vectorizer = HashingVectorizer(input=input_type, n_features=2**24, decode_error='ignore', alternate_sign=False)
print('Fitting vectorizer...')
vectorizer.fit(raw_training_data)
print('Transforming training data...')
training_data = vectorizer.transform(raw_training_data)

# Fit the classifier to the training data
print('Fitting classifier...')
# Create a Naive Bayes classifier
clf = MultinomialNB(force_alpha=True)
clf.fit(training_data, labels)

# Predict the labels of the test data
# Convert the test data to a bag of words
test_counts = vectorizer.transform(test_data)
print('Predicting labels...')
preds = clf.predict(test_counts)

i = 0
for i in range(len(preds)):
    print(preds[i], test_data[i])


# Print the accuracy of the classifier
print("Accuracy:", clf.score(test_counts, test_labels))

# Print the precision
print("Precision Score:", metrics.precision_score(test_labels, preds, average='macro', zero_division=0))

# Print the recall
print("Recall Score:", metrics.recall_score(test_labels, preds, average='macro'))

#Print the f-score
print("F1 Score:", metrics.f1_score(test_labels, preds, average='macro'))

#Get class probabilities
print("Class Probabilities:", clf.predict_proba(test_counts))

