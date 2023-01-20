# Needed!!!
# pip install scikit-learn
##pip install seaborn
from typing import Callable, List, Optional, Tuple
import pandas as pd
import optional as optional
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from transformers import BertTokenizer, BertModel, BertConfig

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
from sklearn import metrics
from sklearn import preprocessing

file = None
featureEncoding = None
Train_X, Test_X, Train_Y, Test_Y = [], [], [], []

#To set the file in the main.
def setFile(fileArg, featureEncodingArg):
    global file
    global featureEncoding
    file = fileArg
    featureEncoding = featureEncodingArg

#Split the data into test and training sets.
#Train_X 20% texts, Test_X = 80% texts, Train_Y = labels of the Train_X texts, Test_Y = labels of the Test_X texts.
def splitData(percentage: int, column: str):
    global Train_X, Test_X, Train_Y, Test_Y
    #inputs features of comments and correct solution to prediction at the split percentage given in this func
    Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(list(file[featureEncoding]),file['final_label'],test_size=percentage, random_state = 0)

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

def doBERT(treeCount :int):
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

    #demo next word prediction in a sentence
    """
    BertModel.from_pretrained("bert-base-uncased")
    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Tokenized input
    text = "[CLS] Who was Jim Henson ? [SEP] Jim Henson was a puppeteer [SEP]"
    tokenized_text = tokenizer.tokenize(text)

    # Mask a token that we will try to predict back with `BertForMaskedLM`
    masked_index = 8
    tokenized_text[masked_index] = '[MASK]'
    assert tokenized_text == ['[CLS]', 'who', 'was', 'jim', 'henson', '?', '[SEP]', 'jim', '[MASK]', 'was', 'a',
                              'puppet', '##eer', '[SEP]']

    # Convert token to vocabulary indices
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    # Define sentence A and B indices associated to 1st and 2nd sentences (see paper)
    segments_ids = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]

    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])
    """

    #feature union: adding TF IDF features
    """
    from sklearn.feature_extraction.text import (
        CountVectorizer, TfidfTransformer
    )

    tf_idf = Pipeline([
        ("vect", CountVectorizer()),
        ("tfidf", TfidfTransformer())
    ])

    model = Pipeline([
        ("union", FeatureUnion(transformer_list=[
            ("bert", bert_transformer),
            ("tf_idf", tf_idf)
        ])),
        ("classifier", classifier),
    ])
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
    #plt.show()
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


























