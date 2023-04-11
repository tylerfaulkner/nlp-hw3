from naive_bayes.naive_bayes import  NaiveBayes
from naive_bayes.naive_bayes import  loadData

#Load the training data
training_data, labels = loadData(['trainingSet.txt', 'trainingSetAdditional.txt'])

#Load the subjclues data
raw_subjclues_data = open('subjectivity/subjclues.tff', 'r').readlines()
#Clean subjclues data into correct format
raw_subjclues_data = [x.split(' ') for x in raw_subjclues_data]

#Remove weaksubj clues
raw_subjclues_data = [x for x in raw_subjclues_data if 'type=weaksubj' not in x[0]]
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
"""
#Get highest count
max_count = min(len(raw_subjclues_pos_data), len(raw_subjclues_neg_data), len(raw_subjclues_neu_data))

#Get random entries from each list to make them all the same length
raw_subjclues_pos_data = np.random.choice(raw_subjclues_pos_data, max_count, replace=False)
raw_subjclues_neg_data = np.random.choice(raw_subjclues_neg_data, max_count, replace=False)
raw_subjclues_neu_data = np.random.choice(raw_subjclues_neu_data, max_count, replace=False)
"""
print("POS Count: ", len(raw_subjclues_pos_data))
print("NEG Count: ", len(raw_subjclues_neg_data))
print("NEU Count: ", len(raw_subjclues_neu_data))

print(type(raw_subjclues_data))

#Append subjclues data to training data
training_data += [x.split(' ', 1)[1] for x in raw_subjclues_data]
labels += [x.split(' ', 1)[0] for x in raw_subjclues_data]

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
