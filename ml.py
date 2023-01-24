# Needed!!!
# pip install scikit-learn
##pip install seaborn
import optional as optional
import transformers as ppb
import numpy as np
import torch
import os
import time
#import hardware_control

from BertTransformer import BertTransformer
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from sklearn.pipeline import FeatureUnion

from sklearn import model_selection, svm, naive_bayes
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
import warnings
warnings.filterwarnings('ignore')

file = None
featureEncoding = None
Settings = None
Train_X, Test_X, Train_Y, Test_Y = [], [], [], []

def convertLabelToBinary(fileObject)->object:
    i=0
    for label in fileObject:
        print(label)
        if label=="toxic":
            fileObject[i]=1
        elif label=="non-toxic":
            fileObject[i]=0
        else:
            print(label)
            raise Exception("undecided in this function, should have been selected out in preprocessing")
        print(fileObject[i])
        i+=1
    return fileObject

#To set the file in the main.
def setFile(fileArg, inSettings):
    global file
    global featureEncoding
    global Settings
    file = fileArg
    Settings = inSettings
    featureEncoding = Settings["featureEncoding"]

#Split the data into test and training sets.
#Train_X 20% texts, Test_X = 80% texts, Train_Y = labels of the Train_X texts, Test_Y = labels of the Test_X texts.
def splitData(percentage: int, column: str):
    global Train_X, Test_X, Train_Y, Test_Y, Settings
    #inputs features of comments and correct solution to prediction at the split percentage given in this func
    if Settings["featureEncoding"] == "BERT":
        Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(file[featureEncoding],file['final_label'],test_size=percentage, random_state = 0)
    else:
        Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(list(file[featureEncoding]),file['final_label'],test_size=percentage, random_state = 0)


    if Settings["featureEncoding"] == "BERT":
        # Train_Y = convertLabelToBinary(Train_Y)
        # Test_Y = convertLabelToBinary(Test_Y)
        #encoding done with BERT
        return
    #Encode the labels. 0 = non-toxic, 1 = toxic
    Encoder = LabelEncoder()
    Encoder.fit(["non-toxic", "toxic"])
    Train_Y = Encoder.transform(Train_Y)
    Test_Y = Encoder.transform(Test_Y)


def doSVM(gamma):
    # Classifier - Algorithm - SVM
    # fit the training dataset on the classifier
    SVM = svm.SVC(kernel='linear', gamma = gamma)
    SVM.fit(Train_X, Train_Y)
    predictions_SVM = SVM.predict(Test_X)
    
    #Evaluation
    evaluateAndPrintModel(predictions_SVM, "SVM","gamma being:" + str(gamma))

def doNaiveBayes():
    Naive = naive_bayes.MultinomialNB()
    Naive.fit(Train_X,Train_Y)
    predictions_NB = Naive.predict(Test_X)

    #Evaluation
    evaluateAndPrintModel(predictions_NB, "Naive Bayes", "")

def doRandomForest(treeCount :int):
    clf=RandomForestClassifier(n_estimators=treeCount)
    #training of split data
    clf.fit(Train_X, Train_Y)
    #give prediction
    y_pred = clf.predict(Test_X)
    # eval.:
    evaluateAndPrintModel(y_pred, "Random Forest","treeCount being:" + str(treeCount))

def doKNN():
    knn = KNeighborsClassifier()
    knn.fit(Train_X, Train_Y)
    y_pred = knn.predict(Test_X)
    # eval.:
    evaluateAndPrintModel(y_pred, "KNN", "")


#ValueError: Input contains NaN
#http://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/
def doBERT(treeCount :int):
    global Train_X, Test_X, Train_Y, Test_Y
    model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased')
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    model = model_class.from_pretrained(pretrained_weights)

    tokenized1 = Train_X.apply((lambda x: encodeBERT(x,tokenizer)))
    tokenized2 = Test_X.apply((lambda x: encodeBERT(x,tokenizer)))
    padded1=padTokenized(tokenized1)
    padded2=padTokenized(tokenized2)
    attention_mask1 = np.where(padded1 != 0, 1, 0)
    attention_mask2 = np.where(padded2 != 0, 1, 0)

    tic = time.perf_counter()

    last_hidden_states1=calcLastStatesBert(padded1,attention_mask1,model)
    last_hidden_states2=calcLastStatesBert(padded2,attention_mask2,model)
    print("")

    Train_X = last_hidden_states1[0][:, 0, :].numpy()
    Test_X = last_hidden_states2[0][:, 0, :].numpy()

    clf=RandomForestClassifier(n_estimators=treeCount)
    clf.fit(Train_X, Train_Y)

    print("time elapsed for BERT encoding processing and running it in model:")
    toc = time.perf_counter()
    print(f"{toc - tic:0.4f} seconds")
    y_pred=clf.predict(Test_X)
    evaluateAndPrintModel(y_pred, "BERT with RandomForestClassifier","treeCount being:" + str(treeCount))

    """ runs table:
    Rowcount    Time    Accuracy Comment
    2014        430s    66.5
    2014        400s    66.2
    
    2014        398s    68.2    with further preprocessing
    2014        412s    64.0
    
    201         32s     60.9
    201         27s     60.9
    201         33s     63.4
    
    201         33s     58.5    with further preprocessing
    201         31s     70.7    with further preprocessing
    201         33s     60.9    with further preprocessing         
    201         30s     56.0    with further preprocessing     
    """

    #code using BertTransformer class
    """
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert_model = BertModel.from_pretrained("bert-base-uncased")

    bert_transformer = BertTransformer(tokenizer, bert_model)
    classifier = RandomForestClassifier(n_estimators=treeCount)
    model = Pipeline(
        [
            ("vectorizer", bert_transformer),
            ("classifier", classifier),
        ]
    )
    model.fit(Train_X, Train_Y)
    y_pred = model.predict(Test_X)
    evaluateAndPrintModel(y_pred, "BERT with RF","treeCount being:" + str(treeCount))
    """

"""
input:
testPrediction : what the model produces as a prediction
modelname : name of model
specification : inputs to the model, can be left blank if N/A
output:
prints of evaluations of inputted model and a confustion matrix plot
"""
def evaluateAndPrintModel(testPrediction, modelname :str, specification: str):

    print("")
    print("Results for " + modelname + " " + featureEncoding + " " + specification)
    print(""+modelname+" Accuracy Score -> ", accuracy_score(Test_Y, testPrediction)*100)
    print(""+modelname+" Precision Score -> ", precision_score(Test_Y, testPrediction, average=None)*100)
    print(""+modelname+" Recall Score -> ", recall_score(Test_Y, testPrediction, average=None)*100)
    print(""+modelname+" f1 Score -> ", f1_score(Test_Y, testPrediction, average=None)*100)

    if Settings["showPlots"]:
        cm =confusion_matrix(Test_Y, testPrediction)
        print(cm)
        labels = ["non-toxic", "toxic"]
        sns.heatmap(cm, annot=True, fmt = 'd', xticklabels = labels, yticklabels = labels)
        plt.title(modelname, fontsize = 18, color = "black")
        plt.xlabel("Predicted Values", fontsize = 14, color = "blue", fontweight = 50)
        plt.ylabel("Actual Values", fontsize = 14, color = "green", fontweight = 50)

        #savefig requires folder "figures"
        plt.show()
        #plt.savefig("figures/" + modelname + featureEncoding + str(specification) + ".png", dpi = 1000, format='png', transparent=True)
        plt.clf()
    print("")

#BERT encodings stuff:

def encodeBERT(x,tokenizer):
    if x==None:
        x=""
    return tokenizer.encode(x, add_special_tokens=True)

def padTokenized(tokenized):
    max_len = 0
    for i in tokenized.values:
        if len(i) > max_len:
            max_len = len(i)
    return np.array([i + [0] * (max_len - len(i)) for i in tokenized.values])

def calcLastStatesBert(padded, attention_mask, model):

    input_ids1 = torch.tensor(padded)
    attention_mask1 = torch.tensor(attention_mask)
    with torch.no_grad():
        return model(input_ids1, attention_mask=attention_mask1)
























