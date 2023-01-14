import pandas as ps
from pandas import DataFrame
import preprocessing
import bow
import tfIdf
import ml

"""
Setting of Keys if everything should run on every comment:
    "printStatsToConsole": True,
    "rowCount": 20147,
    "platform" : 0
"""
Settings = {
    # Settings (globally set action control during runtime):

    # prints stats to console
    "printStatsToConsole": True,

    # Select "bow" or "tfIdf"
    "featureEncoding": "tfIdf",
    # [0,1] == TestDataSizeInPercent
    "TestDataSizeInPercent": 0.2,
    # Size of Random Forest
    "TreeCount": 100,
    # unanimous, all 3 vote for it to be a hate word (1)
    # or most inclusive (2) voting in rationales on a hate word
    "votingMode": 2,

    # [0,20147] == rowCount
    "rowCount": 2000,
    #  Select which file to do the EDA with:
    # 0 = Both Platforms, 1 = twitter, 2 = gab
    "platform": 0
}

def printDictSize(name :str,dict :dict):
    if Settings["printStatsToConsole"]:
        print(name+" dict size:")
        print(len(dict))
        print("")

file: DataFrame = ps.read_csv("Hatespeech_dataset.csv")[0:Settings["rowCount"]]

print(Settings)
print("")

"""Preprocessing and Cleaning"""
preprocessing.setFileAndSettings(file,Settings)
preprocessing.removeDots()
preprocessing.upperToLower()
# executed at this point as no words have been removed yet, following funcs
# add a col or modify length of comments in col file["text"]
# votedHatewordDict: dict = preprocessing.processVotedHatewords(Settings["votingMode"])
# printDictSize("hateword",votedHatewordDict)
preprocessing.removeStopwords()
preprocessing.stemWords()
preprocessing.addTextToStringAndTextToList()
preprocessing.removeUndecided()

# # set textasList as Hateword List in all toxic comments
# i :int=0
# for label in file["final_label"]:
#     if label == "toxic":
#         #overwrite with only voted hate words
#         file.at[i,"textasList"]=file.at[i,"mostInclusiveWordList"]
#     i += 1

file = preprocessing.getFile()[Settings["platform"]]

print("preprocessing fertig!")
print("")

"""Machine Learning"""

if Settings["featureEncoding"] == "bow":
    bow.setFile(file)
    bow.calcListOfBows()

    file = bow.getFile()
    print("bow fertig berechnet")
else:
    tfIdf.setFile(file)
    tfIdf.tfIdfColumnTwo()
    file = tfIdf.getFile()
    print("tfIDF fertig berechnet")

ml.setFile(file, Settings["featureEncoding"])
ml.splitData(Settings["TestDataSizeInPercent"], Settings["featureEncoding"])
ml.doNaiveBayes()
ml.doRandomForest(Settings["TreeCount"])
ml.doSVM('scale')
ml.doBERT()

print("machine learning fertig")

