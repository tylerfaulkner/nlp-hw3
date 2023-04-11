"""
A Naive Bayes Classifier from scratch
"""
import numpy as np
import math
import re
import time

def loadData(filenames):
    """
    Load the data from the file
    """
    data = []
    labels = []
    for file in filenames:
        raw_data = open(file, 'r').readlines()
        data += [x.split(' ', 1)[1][:-1] for x in raw_data]
        labels += [x.split(' ', 1)[0] for x in raw_data]
    return data, labels

class NaiveBayes:
    def __init__(self, alpha=1, min_count=1, max_count=100000):
        self.alpha = alpha
        self.min_count = min_count
        self.max_count = max_count
    
    def clean(self, docs):
        print("Cleaning docs")
        for i in range(len(docs)):
            docs[i] = docs[i].lower().replace('.', '').replace('!', '').replace('?', '').replace(',', '').replace(';', '').replace(':', '').replace('(', '').replace(')', '').replace('"', '').replace("'", '').replace('-', '').replace('\n', '')
    
    def fit(self, docs, labels):
        self.clean(docs)
        labels = np.array(labels)
        print("Getting classes")
        self.classes = np.unique(labels)
        self.logprior = {}
        N_docs = len(docs)

        print("Building Vocabulary")
        self.vocab = set()
        for i in range(len(docs)):
            doc = docs[i]
            self.vocab.update(doc.split())
        print("Vocab Size: ", len(self.vocab))


        self.bigdoc = {}
        self.loglikelihood = {}
        for c in self.classes:
            print("Calculating logprior and loglikelihood for class: ", c)
            start_time = time.perf_counter()
            #Get all documents in class c
            class_indeces = np.where(labels == c)
            class_total_word_count = 0
            word_counts = {}
            for index in class_indeces[0]:
                words = docs[index].split()
                class_total_word_count += len(words)
                for word in words:
                    if word in word_counts:
                        word_counts[word] += 1
                    else:
                        word_counts[word] = 1
            N_c = len(class_indeces[0])

            print("Calculating class prob for class: ", c)
            self.logprior[c] = np.log(N_c/N_docs)
            print("Creating megadoc for class: ", c)
            # current word count of class/total word count of class
            print("Calculating word prob for class: ", c)
            last_loop_time = time.perf_counter()
            i = 0
            for word in self.vocab:
                #count occurences of word in bigdoc[c]
                if word in word_counts:
                    count = word_counts[word]
                else:
                    count = 0
                #calculate loglikelihood
                #print("Word: ", word, " Count: ", count, " Class Word Count: ", class_word_count, " Vocab Size: ", len(self.vocab), " Alpha: ", self.alpha)
                self.loglikelihood[word, c] = math.log(count + self.alpha) - math.log(class_total_word_count + (self.alpha * len(self.vocab)))
                i += 1
            print("Time to calculate word prob for class: ", c, " (s): ", time.perf_counter() - start_time)
    
    def predict(self, docs):
        self.clean(docs)
        preds = []
        for i, doc in enumerate(docs):
            #calculate logposterior
            log_sum = {}
            for c in self.classes:
                log_sum[c] = self.logprior[c]
                for word in doc.split():
                    if word in self.vocab:
                        log_sum[c] += self.loglikelihood[word, c]
            #get the class with the highest logposterior
            preds.append(max(log_sum, key=log_sum.get))
        for i in range(len(preds)):
            print(preds[i], docs[i])
            if i > 10:
                print("...")
                break
        return np.array(preds)
    
    def acc(self, expected, actual):
        correct = 0
        for i in range(len(expected)):
            if expected[i] == actual[i]:
                correct += 1
        return correct / len(expected)
    
    def precision(self, expected, actual):
        """
        Macro-averaged precision
        """
        expected = np.array(expected)
        actual = np.array(actual)
        summation = 0
        for c in self.classes:
            tp = len(np.where(expected[np.where(expected == actual)] == c)[0])
            actual_c_total = len(np.where(actual == c)[0])
            #print("Class:", c, "| TP:", tp, "| DENOM:", actual_c_total)
            if actual_c_total != 0:
                summation += tp / (actual_c_total)
        return summation / len(self.classes)
    
    def recall(self, expected, actual):
        """
        Macro-averaged recall
        """
        expected = np.array(expected)
        actual = np.array(actual)
        summation = 0
        for c in self.classes:
            tp = len(np.where(expected[np.where(expected == actual)] == c)[0])
            expected_c_total = len(np.where(expected == c)[0])
            if expected_c_total != 0:
                summation += tp / (expected_c_total)
        return summation / len(self.classes)
    
    def f1(self, expected, actual):
        """
        Macro-averaged f1
        """
        expected = np.array(expected)
        actual = np.array(actual)
        prec = self.precision(expected, actual)
        rec = self.recall(expected, actual)
        return 2 * prec * rec / (prec + rec)
    