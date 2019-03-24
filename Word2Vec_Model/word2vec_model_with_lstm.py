__author__ = "Md. Ahsan Ayub"
__license__ = "GPL"
__credits__ = ["Ayub, Md. Ahsan", "Smith, Steven", "Yilmiz, Ibrahim",
               "Siraj, Ambareen"]
__maintainer__ = "Md. Ahsan Ayub"
__email__ = "mayub42@students.tntech.edu"
__status__ = "Prototype"

# Importing the libraries
import pandas as pd
import numpy as np
import gensim

#importing the data set
dataset = pd.read_csv('sample_production_data_97K.csv')


# ------ Processing the Data -------

# Processing the domain names (text)
import re
number_of_words = 0
number_of_obs = len(dataset)
corpus = []

for i in range(0,number_of_obs):
    domains = re.sub('[.]', ' ', dataset['Domain'][i]);
    domains = domains.lower()
    domains = domains.split()
    #domains = ' '.join(domains)
    corpus.append(domains)
    if (len(domains) > number_of_words):
        number_of_words = len(domains)


# Creating the Word2Vec model
word_model = gensim.models.Word2Vec(corpus, min_count=1, size=100, 
                                    window=number_of_words, sg = 1, iter=10)
pretrained_weights = word_model.wv.syn0
vocab_size, emdedding_size = pretrained_weights.shape


# Process the dataset aligned with the Word2Vec model's result

def word2idx(word):
  return word_model.wv.vocab[word].index
def idx2word(idx):
  return word_model.wv.index2word[idx]

Y = dataset.iloc[:,1].values
X = np.zeros([len(corpus), number_of_words], dtype=np.int32)
for i, sentence in enumerate(corpus):
    #print(sentence)
    for t, word in enumerate(sentence):
        X[i, t] = word2idx(word) + 1
  
# Spliting the dataset into the Training and Test Set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)


# ------ Making the LSTM model -------

# Importing the Keras libraries and packages
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.layers import Dense, Activation
from keras.models import Sequential

# Initializing the classifer
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Embedding(input_dim=vocab_size, output_dim=emdedding_size, weights=[pretrained_weights]))

# Adding the LSTM
classifier.add(LSTM(units=emdedding_size,activation='relu'))
classifier.add(Dense(units=vocab_size))
classifier.add(Activation('softmax'))

# Compiling the LSTM
classifier.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics = ['accuracy'])

# Fitting the LSTM to the Training set
classifier.fit(X_train, Y_train, batch_size = 32, epochs = 10)


# ------ Evaluation -------

# Predicting the Test set results
Y_pred = classifier.predict(X_test)
Y_pred = (Y_pred > 0.5)

# Making the cufusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)
print("Confusion Matrix:\n", cm)

# Knowing accuracy result
from sklearn.metrics import accuracy_score
print("Accuracy: ", accuracy_score(Y_test, Y_pred))

# Measiring F1 Score
from sklearn.metrics import f1_score
print("F1: ", f1_score(Y_test, Y_pred, average='binary'))

# Measuring precision score
from sklearn.metrics import precision_score
print("Precison: ", precision_score(Y_test, Y_pred, average='binary'))

# Measuring recall score
from sklearn.metrics import recall_score
print("Recall: ", recall_score(Y_test, Y_pred, average='binary'))