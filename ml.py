# Needed!!!
# pip install scikit-learn
##pip install seaborn
from typing import Callable, List, Optional, Tuple
import pandas as pd
import optional as optional
import transformers as ppb
import numpy as np
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from sklearn.base import BaseEstimator, TransformerMixin
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

    #Encode the labels. 0 = non-toxic, 1 = toxic
    Encoder = LabelEncoder()
    Encoder.fit(["non-toxic", "toxic"])
    if Settings["featureEncoding"] == "BERT":
        #encoding done with BERT
        return
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

#http://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/
def doBERT(treeCount :int):
    global Train_X, Test_X, Train_Y, Test_Y
    model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased')
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    model = model_class.from_pretrained(pretrained_weights)

    tokenized1 = Train_X.apply((lambda x: encodeBERT(x,tokenizer)))
    tokenized2 = Test_X.apply((lambda x: encodeBERT(x,tokenizer)))

    max_len = 0
    for i in tokenized1.values:
        if len(i) > max_len:
            max_len = len(i)
    padded1 = np.array([i + [0] * (max_len - len(i)) for i in tokenized1.values])

    max_len = 0
    for i in tokenized2.values:
        if len(i) > max_len:
            max_len = len(i)
    padded2 = np.array([i + [0] * (max_len - len(i)) for i in tokenized2.values])

    attention_mask1 = np.where(padded1 != 0, 1, 0)
    attention_mask2 = np.where(padded2 != 0, 1, 0)

    input_ids1 = torch.tensor(padded1)
    attention_mask1 = torch.tensor(attention_mask1)
    with torch.no_grad():
        last_hidden_states1 = model(input_ids1, attention_mask=attention_mask1)

    input_ids2 = torch.tensor(padded2)
    attention_mask2 = torch.tensor(attention_mask2)
    with torch.no_grad():
        last_hidden_states2 = model(input_ids2, attention_mask=attention_mask2)

    Train_X = last_hidden_states1[0][:, 0, :].numpy()
    Test_X = last_hidden_states2[0][:, 0, :].numpy()

    clf=RandomForestClassifier(n_estimators=treeCount)
    clf.fit(Train_X, Train_Y)
    y_pred=clf.predict(Test_X)
    evaluateAndPrintModel(y_pred, "BERT with RandomForestClassifier","treeCount being:" + str(treeCount))

    #code using below class
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
prints of evaluations of inputted model
"""
def evaluateAndPrintModel(testPrediction, modelname :str, specification: str):

    print("")
    print("Results for " + modelname + " " + featureEncoding + " " + specification)
    print(""+modelname+" Accuracy Score -> ", accuracy_score(Test_Y, testPrediction)*100)
    print(""+modelname+" Precision Score -> ", precision_score(Test_Y, testPrediction, average=None)*100)
    print(""+modelname+" Recall Score -> ", recall_score(Test_Y, testPrediction, average=None)*100)
    print(""+modelname+" f1 Score -> ", f1_score(Test_Y, testPrediction, average=None)*100)

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

#custom transformer for our hate detection purposes built on top of BERT base
class BertTransformer(BaseEstimator, TransformerMixin):
    def __init__(
            self,
            bert_tokenizer,
            bert_model,
            max_length: int = 60,
            embedding_func: Optional[Callable[[torch.tensor], torch.tensor]] = None,
    ):
        self.tokenizer = bert_tokenizer
        self.model = bert_model
        self.model.eval()
        self.max_length = max_length
        self.embedding_func = embedding_func

        if self.embedding_func is None:
            self.embedding_func = lambda x: x[0][:, 0, :].squeeze()

    def _tokenize(self, text: str) -> Tuple[torch.tensor, torch.tensor]:
        # Tokenize the text with the provided tokenizer
        if text==None:
            text=""
        tokenized_text = self.tokenizer.encode_plus(text,
                                                    add_special_tokens=True,
                                                    max_length=self.max_length
                                                    )["input_ids"]

        # Create an attention mask telling BERT to use all words
        attention_mask = [1] * len(tokenized_text)

        # bert takes in a batch so we need to unsqueeze the rows
        return (
            torch.tensor(tokenized_text).unsqueeze(0),
            torch.tensor(attention_mask).unsqueeze(0),
        )

    def _tokenize_and_predict(self, text: str) -> torch.tensor:
        tokenized, attention_mask = self._tokenize(text)

        embeddings = self.model(tokenized, attention_mask)
        return self.embedding_func(embeddings)

    def transform(self, text: List[str]):
        if isinstance(text, pd.Series):
            text = text.tolist()

        with torch.no_grad():
            return torch.stack([self._tokenize_and_predict(string) for string in text])

    def fit(self, X, y=None):
        """No fitting necessary so we just return ourselves"""
        return self

def encodeBERT(x,tokenizer):
    if x==None:
        x=""
    return tokenizer.encode(x, add_special_tokens=True)
























