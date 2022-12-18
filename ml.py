# Needed!!!!
# pip install scikit-learn
from sklearn import model_selection, svm, naive_bayes
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn import preprocessing

file = None
Train_X, Test_X, Train_Y, Test_Y = [], [], [], []

#To set the file in the main.
def setFile(fileArg):
    global file
    file = fileArg

#Split the data into test and training sets.
#Train_X 20% texts, Test_X = 80% texts, Train_Y = labels of the Train_X texts, Test_Y = labels of the Test_X texts.
def splitData(percentage: int, column: str):
    global Train_X, Test_X, Train_Y, Test_Y
    Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(list(file[column]),file['final_label'],test_size=percentage)
    
    #Encode the labels. 0 = non-toxic, 1 = toxic
    Encoder = LabelEncoder()
    Encoder.fit(["non-toxic", "toxic", "undecided"])
    Train_Y = Encoder.transform(Train_Y)
    Test_Y = Encoder.transform(Test_Y)

def doSVM():
    # Classifier - Algorithm - SVM
    # fit the training dataset on the classifier
    SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
    SVM.fit(Train_X, Train_Y)
    predictions_SVM = SVM.predict(Test_X)
    
    #Evaluation
    evaluateAndPrintModel(predictions_SVM, "SVM")

    """Grafik"""
    # #Generate scatter plot for training data 
    # plt.scatter(Train_X, Train_X , c = Train_Y)
    # plt.title('Linearly separable data')
    # plt.xlabel('X1')
    # plt.show()

def doNaiveBayes():
    Naive = naive_bayes.MultinomialNB()
    Naive.fit(Train_X,Train_Y)
    predictions_NB = Naive.predict(Test_X)

    #Evaluation
    evaluateAndPrintModel(predictions_NB, "NB")


def doRandomForest(treeCount :int):

    clf=RandomForestClassifier(n_estimators=treeCount)
    #training of split data
    clf.fit(Train_X, Train_Y)
    #give prediction
    y_pred = clf.predict(Test_X)
    # eval.:
    evaluateAndPrintModel(y_pred, "RF")

def doKNN():
    knn = KNeighborsClassifier()
    knn.fit(Train_X, Train_Y)
    y_pred = knn.predict(Test_X)
    # eval.:
    evaluateAndPrintModel(y_pred, "KNN")

# #returns matrix (2x2), whose cardinality is 100%
# def doNaiveBayes(predictedData,realData):
#     assert len(predictedData)==len(realData)
#     realToxicCount=0
#     realNonToxicCount=0
#     predictedToxicCount=0
#     predictedNonToxicCount=0
#     matrix=[[realToxicCount,realNonToxicCount],[predictedToxicCount,predictedNonToxicCount]]
#     i=0
#     for predictedDataPoint in predictedData:
#         predictedDataPoint["label"]=realData[i]["label"]
#         i=+1

def evaluateAndPrintModel(testPrediction, modelname :str):
    print("Results for "+modelname+":")
    print(""+modelname+" Accuracy Score -> ", accuracy_score(testPrediction, Test_Y)*100)
    print(""+modelname+" Precision Score -> ", precision_score(testPrediction, Test_Y, average=None)*100)
    print(""+modelname+" Recall Score -> ", recall_score(testPrediction, Test_Y, average=None)*100)
    print(""+modelname+" f1 Score -> ", f1_score(testPrediction, Test_Y, average=None)*100)
    print("")



























