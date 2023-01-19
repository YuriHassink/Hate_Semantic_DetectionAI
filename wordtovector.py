import pandas as ps
import ast
from gensim.models import Word2Vec, FastText    #pip install gensim

file = None
dataCorpus = None
dataCorpusDict = None

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

def allWordsSet(allWords):
    global dataCorpus

    allWordsSet = []
    for word in allWords:
        if word not in allWordsSet:
            allWordsSet.append(word)
    print("dataCorpus length: " + str(len(allWordsSet)))
    dataCorpus = allWordsSet

def w2vDict(featureCount):
    global dataCorpusDict

    allWordsSet(allWords(file))
    w2v = Word2Vec(list(file["textasList"]), min_count=1, vector_size = featureCount)

    w2vDict = dict.fromkeys(dataCorpus, 0)
    for comment in file["textasList"]:
        for word in comment:
                w2vDict[word] = sum(list(w2v.wv[word]))
    return w2vDict

def w2vCol(featureCount):

    global file
    global dataCorpusDict
    
    dataCorpusDict = w2vDict(featureCount)

    w2vCol = []
    for comment in file["textasList"]:
        bow = dict.fromkeys(dataCorpus, 0)
        for word in comment:
            bow[word] = dataCorpusDict[word]
        w2vCol.append(list(bow.values()))
    file["w2v"] = w2vCol