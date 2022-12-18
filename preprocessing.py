import matplotlib.pyplot as plt
import pandas as ps
import preprocessing_helper
import re
import ast
from nltk.corpus import stopwords
import string

file = None
Settings = None


# To set the file in the main.
def setFileAndSettings(fileArg, settings):
    global file
    global Settings
    Settings = settings
    file = fileArg


# remove dots
def removeDots():
    global file
    file["text"] = file['text'].astype('str').str.replace(r".", r"", regex=False)


# remove stopwords
def removeStopwords():
    global file

    stopWords = set(stopwords.words('english'))
    i = 0

    for comment in file["text"]:
        commentToStringToList = " ".join(ast.literal_eval(comment)).split(" ")
        for word in commentToStringToList:
            if word in stopWords:
                commentToStringToList.remove(word)
        file.at[i, "text"] = str(commentToStringToList)

        i += 1

    # for word in allWordsInOneString.split():
    #    word = word.lower() # in case they arenet all lower cased
    #    if word not in stopWords:
    #       processed_word_list.append(word)
    # return processed_word_list


# uppercase -> lowercase
def upperToLower():
    for i in range(ord('A'), ord('Z')):
        preprocessing_helper.alterText(file, chr(i), chr(i + 32))


def stemWords():
    global file
    file = preprocessing_helper.stemWords(file)


# adding different formats of text columns:
def addTextToStringAndTextToList():
    global file

    # Add col with just list instead of list in string
    textasListList = []
    for comment in file["text"]:
        textasListList.append(ast.literal_eval(comment))
    file["textasList"] = textasListList

    # Add a column of each text as a real string: list of words -> words as one str in last col
    textasStrList = []
    for x in file["text"]:
        textasStrList.append(" ".join(ast.literal_eval(x)))
    file["textasStr"] = textasStrList

    printTableState()

def removeUndecided():
    global file
    file = file[file["final_label"] != "undecided"]

def undecidedToNontoxic():
    global file
    for i in range(len(file["final_label"])):
        if file["final_label"][i] == "undecided":
            file.at[i, "final_label"] = "non-toxic"

# To get the preprocessed file in the main.
def getFile():
    global file

    # twitter = preprocessing_helper.ifContains(file, "post_id", "twitter")
    # gab = preprocessing_helper.ifContains(file, "post_id", "gab")

    return (file, None, None)

# adds hateword freq cols, returns hateword corpuses: unanimous and mostInclusive
# in: mode : gives back dict of types and calcs HatewordPercentage by this type:
# 1: unanimous voting, : 2: most inclusive voting
# out: dicts
def processVotedHatewords(mode:int) -> dict:
    #tests for 2 funcs in helper file:
    #print(preprocessing_helper.getUnanimousVoteList("[[1,0,1],[1,1,0],[1,0,0]]"))
    #print(preprocessing_helper.getMostInclusiveVoteList("[[1,0,1],[1,0,0],[0,0,0]]"))
    returnDict: dict = {}
    commentHatewordList=None
    # adds new cols
    file["HatewordPercentage"] = None
    file["UnanimousVoteList"] = None
    file["mostInclusiveVoteList"] = None
    # index of row in file
    i = -1
    # select toxic comments
    for label in file["final_label"]:
        # update index of row
        i += 1
        if label == "toxic":
            commentList = file.at[i, "textasList"]
            unanimousVoteList = preprocessing_helper.getUnanimousVoteList(file.at[i, "rationales"])
            mostInclusiveVoteList=preprocessing_helper.getMostInclusiveVoteList(file.at[i, "rationales"])
            file.at[i, "UnanimousVoteList"] = unanimousVoteList
            file.at[i,"mostInclusiveWordList"] = preprocessing_helper.extractVotedHatewords(commentList, mostInclusiveVoteList)

            if mode==1:
                commentHatewordList=preprocessing_helper.extractVotedHatewords(commentList, unanimousVoteList)
            elif mode==2:
                commentHatewordList=file.at[i,"mostInclusiveWordList"]
            else:
                raise Exception("invalid mode!")
            file.at[i, "HatewordPercentage"] = (len(commentHatewordList) / len(commentList)) * 100

            # print(file.at[i, "HatewordPercentage"])
            for hateWord in commentHatewordList:
                # no entry
                if returnDict.get(hateWord) == None:
                    returnDict[hateWord] = 1
                # increment if entry found
                else:
                    returnDict[hateWord] += 1
    printTableState()
    return returnDict

#ideally used if table in file is changed in its colums
def printTableState():
    if Settings["printStatsToConsole"]:
        print("changes to file table state:")
        print(file.all())
        print("")
