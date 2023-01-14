import pandas as ps
import numpy as np
import math

file = None
dataCorpus = None

#To set the file in the main.
def setFile(fileArg):
    global file
    file = fileArg

def allWords(file):
    dataCorpus = []
    for comment in file["textasList"]:
        for word in comment:
            dataCorpus.append(word) 
    return(dataCorpus)


#unique wordset = data corpus
def dataCorpus(wordset):
    unique_wordset = []
    for word in wordset:
      if word not in unique_wordset:
        unique_wordset.append(word)
    print("unique words in comments:")
    print(len(unique_wordset))
    print("")
    return unique_wordset

#word for tf and idf
searchWord = "hate"

#TF: Count of word in the comment.
def term_frequency(comment, word):
    
    N = len(comment)
    count = 0
    for w in comment:
        if w == word:
            count += 1

    return count / N

#idf
def idf(word):
    global file
    global dataCorpus

    N = len(dataCorpus)
    com_num = 0
    for comment in file["textasList"]:
        if word in comment:
            com_num += 1

    return math.log(N/com_num)

#tf column
def tfColumn():
    global file

    tf = []
    for comment in file['textasList']:
        tfList = []
        for word in comment:
            tfList.append(term_frequency(comment, word))
        tf.append(tfList)
    file["tf"] = tf  

#idf dict
def idfDict():
    global file
    global dataCorpus

    idf_dict = dict.fromkeys(dataCorpus, 0)

    for word in idf_dict.keys():
        idf_dict[word] = idf(word)

    return idf_dict


"""
Vorgehen:
Für jeden Comment ein Dictionary erstellen, 
"""
def tfIdfColumnTwo():
    global file
    global dataCorpus
    dataCorpus = dataCorpus(allWords(file))
    tfColumn()
    print("tf calculated!")
    idf_dict = idfDict()
    print("idf calculated!")

    tfIdfList = []
    row = 0
    for comment in file["textasList"]:

        #Make tfColumn to List with length of the data Corpus
        currentTfDict = dict.fromkeys(dataCorpus, 0)

        w = 0
        for word in comment:
            currentTfDict[word] = list(file["tf"])[row][w]
            w += 1

        currentTf_idfDict = dict.fromkeys(dataCorpus, 0)
        currentTf_idfDictStrforPPT = dict.fromkeys(dataCorpus, "")
        for word in currentTf_idfDict.keys():
            currentTf_idfDict[word] = idf_dict[word] * currentTfDict[word]
            
        tfIdfList.append(list(currentTf_idfDict.values()))
        row += 1

    file["tfIdf"] = tfIdfList
    
    


#To get the file in the main
def getFile():
    global file
    return file