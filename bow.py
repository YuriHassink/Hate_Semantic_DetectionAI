import pandas as ps

file = None

def setFile(fileArg):
    global file
    file = fileArg

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
    return(allWordsSet)


#Calculate the bow of a comment
def bow(comment, dataCorpus):
    bow = dict.fromkeys(dataCorpus, 0)
    for word in comment:
        bow[word] = comment.count(word)
    return(list(bow.values()))

#Set file["bow"] as the list of bows.
def calcListOfBows():
    global file

    dataCorpus = allWordsSet(allWords(file))
    listOfBows = []
    for comment in file["textasList"]:
        listOfBows.append(bow(comment, dataCorpus))
    file["bow"] = listOfBows

#To get the file in the main
def getFile():
    global file
    return file