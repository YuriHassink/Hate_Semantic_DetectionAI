# code for Glove word embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
  
file = None
dataCorpus = None
dataCorpusDict = None

#To set the file in the main.
def setFile(fileP):
    global file
    file = fileP

#To get the file in the main.
def getFile():
    global file
    return file

def allWords(file):
    dataCorpus = []
    print(file["textasList"])
    for comment in file["textasList"]:
        for word in comment:
            dataCorpus.append(word)
    return(dataCorpus)

def allWordsSet(allWords):
    global dataCorpus

    allWordsSet = []
    for word in allWords:
        if word not in allWordsSet:
            allWordsSet.append(word)
    print("dataCorpus length: " + str(len(allWordsSet)))
    dataCorpus = allWordsSet

# create the dict.
def glovesDict():
    allWordsSet(allWords(file))
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(dataCorpus)
  
    # number of unique words in dict.
    print("Number of unique words in dictionary=",len(tokenizer.word_index))
    print("Dictionary is = ", tokenizer.word_index)
    return tokenizer.word_index
 
# download glove and unzip it in Notebook.
#!wget http://nlp.stanford.edu/data/glove.6B.zip
#!unzip glove*.zip
  
# vocab: 'the': 1, mapping of words with
# integers in seq. 1,2,3..
# embedding: 1->dense vector
def embedding_for_vocab(filepath, word_index,embedding_dim):
    vocab_size = len(word_index) + 1
      
    # Adding again 1 because of reserved 0 index
    embedding_matrix_vocab = np.zeros((vocab_size,embedding_dim))
  
    with open(filepath, encoding="utf8") as f:
        for line in f:
            word, *vector = line.split()
            if word in word_index:
                idx = word_index[word]
                embedding_matrix_vocab[idx] = np.array(vector, dtype=np.float32)[:embedding_dim]
  
    return embedding_matrix_vocab

def glovesCol(embedding_matrix_vocab):

    global file
    global dataCorpusDict
    
    #create an array and insert each word2Vec Vector
    glovesCol = []
    for i in embedding_matrix_vocab:
        #Now create one vector per comment, which is just the average of the words vectors. 
        glovesCol.append(sum(i)/len(i))
    return glovesCol

def doGloves(embedding_dim):
    global file
    print(file)
    embedding_matrix_vocab = embedding_for_vocab('glove.6B.50d.txt', glovesDict(),embedding_dim)
    print(embedding_matrix_vocab)
    print(glovesCol(embedding_matrix_vocab))
    return embedding_matrix_vocab
