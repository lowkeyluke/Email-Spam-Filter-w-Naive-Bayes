# <output_file> should be a csv file with two columns for each email:
# <ID> <spam/ham> (email ID, spam or ham)
import argparse
import csv
import numpy as np
# pickle as pkl, random

# loads Train and Test data
def load_data(fileTrain, fileTest):  # , fileTest
    trainData, trainLabels, Xtest = [], [], []

    with open(fileTrain, 'rb') as file:
        reader = csv.reader(file)
        for row in reader:
            rw = row[0].split()
            # rw = map(int, row[0].split())
            trainData.append(rw)

    emailID = []
    for row in trainData:
        rw = row[0]
        emailID.append(rw)
    for row in trainData:
        rw = row[1]
        trainLabels.append(rw)

    # all words
    words, occurrences = [], []
    for row in trainData:
        wordsSample, occurrencesSample = [], []
        for i in range(2, len(row)):
            if i%2 == 0:
                rw = row[i]
                wordsSample.append(rw)
            else:
                rw = int(row[i])
                occurrencesSample.append(rw)
        words.append(wordsSample)
        occurrences.append(occurrencesSample)

    # words split in spam and ham sets
    # GATHERED ONLY FROM TRAINING DATA
    spamWords, spamOccurrences = [], []
    hamWords, hamOccurrences = [], []
    for row in trainData:
        if row[1] == 'spam':
            for i in range(2, len(row)):
                if i%2 == 0:
                    rw = row[i]
                    spamWords.append(rw)
                else:
                    rw = int(row[i])
                    spamOccurrences.append(rw)
        else:
            for i in range(2, len(row)):
                if i % 2 == 0:
                    rw = row[i]
                    hamWords.append(rw)
                else:
                    rw = int(row[i])
                    hamOccurrences.append(rw)

    # test data
    with open(fileTest, 'rb') as file:
        reader = csv.reader(file)
        for row in reader:
            rw = row[0].split()
            # rw = map(int, row[0].split())
            Xtest.append(rw)

    print('Data Loading: done')
    return trainData, Xtest, emailID, trainLabels, words, occurrences, \
           spamWords, spamOccurrences, hamWords, hamOccurrences

# spam: spamWords
# ham: hamWords
# sO: spamOccurrences
# hO: hamOccurrences
# email: TEST email containing only the words
def naiveBayes(spam, ham, sO, hO, email ,probSpam):
    spamicities = {}  # every word in email given a spam score key=word, value=score
    probHam = 1 - probSpam
    # print("probSpam:", probSpam)

    # for all words
    for word in email:
        # if word is in both spam and ham emails
        if word in spam and word in ham:
            spamOcc = np.size(np.where(spam == word))  # list.index
            probWordSpam = spamOcc * probSpam
            # print("pwspam:", probWordSpam)
            hamOcc = np.size(np.where(ham == word))
            probWordHam = hamOcc * probHam
            spamicities[word] = probWordSpam / (probWordSpam+probWordHam)
        # else if word only in spam
        elif word in spam and word not in ham:
            spamicities[word] = .99
        # else if word only in ham
        elif word in ham and word not in spam:
            spamicities[word] = .01

    # print("spamicities:", spamicities)
    a = 0
    b = 0
    for key, val in spamicities.items():
        a += np.log(1-val)
        b += np.log(val)
    c = a-b
    spam_score = 1/(np.exp(c)+1)

    return spam_score

parser = argparse.ArgumentParser()
parser.add_argument('-f1', help='training file in csv format', required=True)
parser.add_argument('-f2', help='test file in csv format', required=True)
parser.add_argument('-o', help='output labels for the test dataset', required=True)

args = vars(parser.parse_args())
Xtrain_name = args['f1']
# Ytrain_name = args['f1'].split('.')[0] + '_labels.csv'
Xtest_name = args['f2']
Ytest_predict_name = args['o']

# trainingData, emailID, trainingLabels, words, occurrences, spamWords, spamOccurrences, hamWords, hamOccurrences = load_data(fileTrain='trainq2.csv')
trainingData, testData, emailID, trainingLabels, words, occurrences, spamWords, spamOccurrences, \
hamWords, hamOccurrences = load_data(fileTrain=Xtrain_name, fileTest=Xtest_name)
# convert all to numpy arrays
NPtrain = np.array(trainingData)
NPtest = np.array(testData)
NPID=  np.array(emailID)
NPlabels = np.array(trainingLabels)
NPwords=  np.array(words)
NPoccurrences = np.array(occurrences)
NPspamWords=  np.array(spamWords)
NPspamOccurrences = np.array(spamOccurrences)
NPhamWords=  np.array(hamWords)
NPhamOccurrences = np.array(hamOccurrences)

# calculate probSpam for training data
spamCount = 0
for word in NPspamWords:
    spamCount += 1
hamCount = 0
for word in NPhamWords:
    hamCount += 1
probSpam = float(spamCount) / (spamCount+hamCount)

email = NPtrain[10]  # random email from train to see if working
ID = email[0]
label = email[1]
index = [0, 1]
# DELETE LABEL
email = np.delete(email, index)
index = []
for i in range(len(email)):
    if i%2 != 0:
        index.append(i)
email = np.delete(email, index)
# train
spamicity = naiveBayes(NPspamWords, NPhamWords, NPspamOccurrences, NPhamOccurrences, email, probSpam)
print("dict:", spamicity)
if spamicity >= .5:
    eLabel = 'spam'
else:
    eLabel = 'ham'
print("predict:", eLabel)
print("actual:", label)

Ypredict = []
# test
for email in NPtest:
    ID = email[0]
    label = email[1]
    index = [0, 1]
    email = np.delete(email, index)
    index = []
    for i in range(len(email)):
        if i%2 != 0:
            index.append(i)
    email = np.delete(email, index)

    spamicity = naiveBayes(NPspamWords, NPhamWords, NPspamOccurrences, NPhamOccurrences, email, probSpam)
    print("dict:", spamicity)
    if spamicity >= .5:
        eLabel = 'spam'
    else:
        eLabel = 'ham'
    Ypredict.append(eLabel)
    print("predict:", eLabel)
    print("actual:", label)

with open(Ytest_predict_name, "wb") as f:
    writer = csv.writer(f)
    writer.writerows(Ypredict)
print("Output files generated")
