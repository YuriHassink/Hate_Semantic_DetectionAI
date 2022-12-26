from nltk import LancasterStemmer
from nltk.corpus import stopwords
import ast
import re
import string


# collection of functions for better collaborative working, no calls in main are addressed to this file directly

# Drops all rows with the given object in the given category.
def ifContains(file, category, object):
    retFile = file
    i = 0
    for x in retFile[category]:
        if not object in x:
            retFile = retFile.drop(i)
        i += 1
    return retFile


# removes suf/a/pre-fixes src:https://en.wikipedia.org/wiki/Word#Morphology
def stemWords(file):
    LStemmer = LancasterStemmer()
    i = 0

    for comment in file["text"]:
        newCommentList = []
        commentToStringToList = " ".join(ast.literal_eval(comment)).split(" ")
        for word in commentToStringToList:
            # replace with stemmed word for all elem in current Commentlist
            newCommentList.append(LStemmer.stem(word))
        file.at[i, "text"] = str(newCommentList)
        i += 1

    return file


# Changes every a to a b..
def alterText(file, a, b):
    i = 0
    words = []

    for x in file["text"]:
        x = x.replace(a, b)
        file.at[i, "text"] = x
        i += 1


# in:   commentAsList
#       voteList
# out: votedHateWordList
# it holds that:
# len(commentAsList)==len(voteList)
# len(votedHateWordList)<= len(commentAsList)
def extractVotedHatewords(commentAsList, voteList):
    returnList = []
    # quick fix: voteList ist sometimes one smaller than other
    if len(voteList) != len(commentAsList):
        voteList.append(1)
    i = 0
    for word in commentAsList:
        if voteList[i] == 1:
            returnList.append(word)
        i += 1
    return returnList


# in: annotatorList
# out: a list with len(comment) containing unanimous voting of the annotators, meaning:
# [1,0,1][1,1,0][1,0,0] -> [1,0,0]
def getUnanimousVoteList(annotatorList):
    # the annotator list is a string in the csv => convert to proper list
    annotatorList = ast.literal_eval(annotatorList)
    unanimousVotumList = annotatorList[0]
    # TODO: unoptimized here: one redundant for loop
    for annotatorVotum in annotatorList:
        i = 0
        for vote in annotatorVotum:
            #quick fix: unanimousVotumList ist sometimes one smaller than other
            if len(unanimousVotumList) != len(annotatorVotum):
                unanimousVotumList.append(1)
            if vote != unanimousVotumList[i]:
                if unanimousVotumList[i] == 1:
                    unanimousVotumList[i] = vote
            i += 1
    return unanimousVotumList


# in: annotatorList
# out: a list with len(comment) containing most inclusive voting of the annotators, meaning:
# [1,0,1][1,0,0][0,0,0] -> [1,0,1]
def getMostInclusiveVoteList(annotatorList):
    # the annotator list is a string in the csv => convert to proper list
    annotatorList = ast.literal_eval(annotatorList)
    mostInclusiveVotumList = annotatorList[0]
    # TODO: unoptimized here: one redundant for loop
    for annotatorVotum in annotatorList:
        i = 0
        for vote in annotatorVotum:
            #quick fix: mostInclusiveVotumList ist sometimes one smaller than other
            if len(mostInclusiveVotumList) != len(annotatorVotum):
                mostInclusiveVotumList.append(1)
            if vote == 1:
                mostInclusiveVotumList[i] = vote
            i += 1
    return mostInclusiveVotumList
