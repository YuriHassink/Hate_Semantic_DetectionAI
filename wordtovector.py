import pandas as ps
import numpy as np
import gensim
#https://code.google.com/archive/p/word2vec/

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

def allWordsSet(allWords):
    allWordsSet = []
    for word in allWords:
        if word not in allWordsSet:
            allWordsSet.append(word)
    print("dataCorpus length: " + str(len(allWordsSet)))
    return allWordsSet


#Add the column "w2v" to the file.
def w2vCol(featureCount):

    global file
    
    #Use our model as base. If googles model has a word, use googles vector of the word.
    w2vOurModel = gensim.models.Word2Vec(file["textasList"], min_count = 1, vector_size = featureCount)
    w2vGoogle = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)
    for word in allWordsSet(allWords(file)):
        try:
            w2vOurModel.wv[word] = w2vGoogle[word]
        except:
            pass

    #create an array and insert each word2Vec Vector
    w2vCol = []
    for comment in file["textasList"]:
        comment2vec = []
        for word in comment:
            comment2vec.append(w2vOurModel.wv[word])
        
        #Now create one vector per comment, which is just the average of the words vectors. 
        w2vCol.append(np.mean(comment2vec, axis = 0))
    file["w2v"] = w2vCol