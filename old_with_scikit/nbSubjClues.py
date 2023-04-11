from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import numpy as np

# Create a Gaussian Naive Bayes classifier
clf = MultinomialNB(force_alpha=True)

# Load the training data
raw_training_data = open('trainingSet.txt', 'r').readlines()

#Load the additional training data
raw_additional_training_data = open('trainingSetAdditional.txt', 'r').readlines()

#Append additonal test data to original data
raw_training_data = raw_training_data + raw_additional_training_data

#Load the subjclues data
raw_subjclues_data = open('subjectivity/subjclues.tff', 'r').readlines()
#Clean subjclues data into correct format
raw_subjclues_data = [x.split(' ') for x in raw_subjclues_data]

#Remove weaksubj clues
#raw_subjclues_data = [x for x in raw_subjclues_data if 'type=weaksubj' not in x[0]]
raw_subjclues_data = [[x[2], x[5]] for x in raw_subjclues_data]

#Remove key names from subjclues
raw_subjclues_data = [[x[0].replace('word1=', ''), x[1].replace('priorpolarity=', '').replace('\n', '')] for x in raw_subjclues_data]

#Convert priorpolarity to NEG,POS, or NEU
raw_subjclues_data = [[x[0], x[1].replace('negative', 'NEG').replace('positive', 'POS').replace('neutral', 'NEU')] for x in raw_subjclues_data]

#Remove words labeled both from subjclues
raw_subjclues_data = [x for x in raw_subjclues_data if x[1] != 'both']

#Concat subjclues lines into string like training data
raw_subjclues_data = [x[1]+' '+x[0] for x in raw_subjclues_data]

#Seperate subjclues data into pos and neg and neu
raw_subjclues_pos_data = [x for x in raw_subjclues_data if x.split(' ', 1)[0] == 'POS']
raw_subjclues_neg_data = [x for x in raw_subjclues_data if x.split(' ', 1)[0] == 'NEG']
raw_subjclues_neu_data = [x for x in raw_subjclues_data if x.split(' ', 1)[0] == 'NEU']

#Get highest count
max_count = min(len(raw_subjclues_pos_data), len(raw_subjclues_neg_data), len(raw_subjclues_neu_data))

#Get random entries from each list to make them all the same length
raw_subjclues_pos_data = np.random.choice(raw_subjclues_pos_data, max_count, replace=False)
raw_subjclues_neg_data = np.random.choice(raw_subjclues_neg_data, max_count, replace=False)
raw_subjclues_neu_data = np.random.choice(raw_subjclues_neu_data, max_count, replace=False)

print("POS Count: ", len(raw_subjclues_pos_data))
print("NEG Count: ", len(raw_subjclues_neg_data))
print("NEU Count: ", len(raw_subjclues_neu_data))

#Append subjclues data to training data
raw_training_data = raw_training_data + raw_subjclues_pos_data.tolist() + raw_subjclues_neg_data.tolist() + raw_subjclues_neu_data.tolist()


# Split the labels from the text
labels = [x.split(' ', 1)[0] for x in raw_training_data]
training_data = [x.split(' ', 1)[1][:-1] for x in raw_training_data]

# Load the test data
raw_test_data = open('testSet.txt', 'r').readlines()
test_labels = [x.split(' ', 1)[0] for x in raw_test_data]
test_data = [x.split(' ', 1)[1][:-1] for x in raw_test_data]

#print(test_data)

#Bag of words
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
training_data = vectorizer.fit_transform(training_data)

# Fit the classifier to the training data
clf.fit(training_data, labels)

# Predict the labels of the test data
# Convert the test data to a bag of words
test_counts = vectorizer.transform(test_data).toarray()
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

