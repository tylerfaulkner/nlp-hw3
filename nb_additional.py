from naive_bayes.naive_bayes import  NaiveBayes
from naive_bayes.naive_bayes import  loadData

#Load the training data
training_data, labels = loadData(['trainingSet.txt', 'trainingSetAdditional.txt'])

#Load the test data
test_data, test_labels = loadData(['testSet.txt'])

#Create a Naive Bayes classifier
nb = NaiveBayes()

#Fit the classifier to the training data
nb.fit(training_data, labels)

#Predict the labels of the test data
preds = nb.predict(test_data)

#Print the accuracy of the classifier
print("Accuracy:", nb.acc(test_labels, preds))

#Print the precision
print("Precision Score:", nb.precision(test_labels, preds))

#Print the recall
print("Recall Score:", nb.recall(test_labels, preds))

#Print the f-score
print("F1 Score:", nb.f1(test_labels, preds))
