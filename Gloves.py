# code for Glove word embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from nltk import LancasterStemmer
  
file = None

#To set the file in the main.
def setFile(fileP):
    global file
    file = fileP

#To get the file in the main.
def getFile():
    global file
    return file

def allWords(file):
    dataCorpus = []
    for comment in file["textasList"]:
        for word in comment:
            dataCorpus.append(word)
    return(dataCorpus)

def allWordsSet(allWordsp):

    allWordsSet = []
    for word in allWordsp:
        if word not in allWordsSet:
            allWordsSet.append(word)
    print("dataCorpus length: " + str(len(allWordsSet)))

    return allWordsSet


def doGloves(featureCount):
    global file
    
    #Create empty dict
    dataCorpus = allWordsSet(allWords(file))
    # glovesDict = dict.fromkeys(dataCorpus, None)
    glovesDict = dict()
    stemmer = LancasterStemmer()

    #Open the txt and then add the vector the of each word, if it exists in the txt.
    with open('glove.6B.50d.txt', encoding="utf8") as f:
        for line in f:
            word, *vector = line.split()
            vector = list(map(float, vector))
            if word in dataCorpus:
                glovesDict[word] = vector[0:featureCount]
            if stemmer.stem(word) in dataCorpus:
                glovesDict[stemmer.stem(word)] = vector[0:featureCount]
    
    
    #create an array and insert each word2Vec Vector
    glovesCol = []

    count = 0
    for comment in file["textasList"]:
        comment2vec = []
        for word in comment:
            wordVec = glovesDict[word]
            try:
                comment2vec.append(wordVec)
            except:
                pass
        if comment2vec == []:
            count+=1
            comment2vec = [[0] * featureCount]
        glovesCol.append(np.mean(comment2vec, axis = 0))

    file["gloves"] = glovesCol