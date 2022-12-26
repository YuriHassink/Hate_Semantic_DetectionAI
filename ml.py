# Needed!!!!
# pip install scikit-learn
##pip install seaborn
from sklearn import model_selection, svm, naive_bayes
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier

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
    Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(list(file[featureEncoding]),file['final_label'],test_size=percentage, random_state = 0)
    
    #Encode the labels. 0 = non-toxic, 1 = toxic
    Encoder = LabelEncoder()
    Encoder.fit(["non-toxic", "toxic"])
    Train_Y = Encoder.transform(Train_Y)
    Test_Y = Encoder.transform(Test_Y)

def doSVM(g):
    # Classifier - Algorithm - SVM
    # fit the training dataset on the classifier
    SVM = svm.SVC(kernel='linear', gamma = g)
    SVM.fit(Train_X, Train_Y)
    predictions_SVM = SVM.predict(Test_X)
    
    #Evaluation
    evaluateAndPrintModel(predictions_SVM, "SVM", str(g))

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
    evaluateAndPrintModel(y_pred, "Random Forest", str(treeCount))

def doKNN():
    knn = KNeighborsClassifier()
    knn.fit(Train_X, Train_Y)
    y_pred = knn.predict(Test_X)
    # eval.:
    evaluateAndPrintModel(y_pred, "KNN", "")

def evaluateAndPrintModel(testPrediction, modelname :str, specification: str):

    #Calculate Precision for testing
    # TP = 0
    # FP = 0
    # for i in range(len(testPrediction)):
    #     if testPrediction[i] == 1 and Test_Y[i] == 1:
    #         TP += 1 
    #     if testPrediction[i] == 1 and Test_Y[i] == 0:
    #         FP += 1
    # print(TP / (TP+FP))

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



























