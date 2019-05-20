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
dataset = pd.read_csv('C:\\Users\\hemlo\\Downloads\\sample_production_data_97K.csv')
print(dataset.head())


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
    #domains = list(domains)
    corpus.append(domains)
    if (len(domains) > number_of_words):
        number_of_words = len(domains)
# Creating the Word2Vec model
word_model = gensim.models.Word2Vec(corpus, min_count=1, size=100, 
                                    window=number_of_words, sg = 1, iter=10)
pretrained_weights = word_model.wv.vectors
vocab_size, embedding_size = pretrained_weights.shape


# Process the dataset aligned with the Word2Vec model's result

def word2idx(word):
  return word_model.wv.vocab[word].index
def idx2word(idx):
  return word_model.wv.index2word[idx]

Y = dataset['class'].values
X = np.zeros([len(corpus), number_of_words], dtype=np.int32)
for i, sentence in enumerate(corpus):
    for t, word in enumerate(sentence):
        X[i, t] = word2idx(word)
  
# Spliting the dataset into the Training and Test Set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, stratify=Y)
#X_train, X_test, Y_train, Y_test = train_test_split(X_test, Y_test, test_size = 0.2, stratify=Y_test)

# ------ Making the LSTM model -------

# Importing the Keras libraries and packages
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.layers import Dense, Activation, Dropout, Input
from keras.models import Model
from keras.callbacks import EarlyStopping

main_input = Input(shape=(number_of_words,))
embed = Embedding(input_dim=vocab_size, output_dim=embedding_size, weights=[pretrained_weights])(main_input)
lstm = LSTM(units=embedding_size, activation='relu', recurrent_dropout=0.3, dropout=0.3)(embed)
#drop = Dropout(0.5)(lstm)
hidden_1 = Dense(units=64, activation='relu')(lstm)
dense = Dense(units=2)(hidden_1)
activation = Activation('softmax')(dense)
classifier = Model([main_input], activation)


# Adding the LSTM
#classifier.add(LSTM(units=embedding_size,activation='relu', recurrent_dropout=0.3, dropout=0.3))
#classifier.add(Dropout(0.5))
#classifier.add(Dense(units=1, activation='sigmoid'))
# classifier.add(Activation('softmax'))

# Compiling the LSTM
classifier.compile(optimizer='SGD', loss='sparse_categorical_crossentropy', metrics = ['accuracy'])

# Callback to stop if validation loss does not decrease
callbacks = [EarlyStopping(monitor='val_loss', patience=2)]

# Fitting the LSTM to the Training set
history = classifier.fit(X_train, 
               Y_train,
               callbacks=callbacks,
               validation_split=0.1,
               batch_size = 64, 
               epochs = 5,
               shuffle=False)


# ------ Evaluation -------
Y_prob = classifier.predict(X_test)
# Predicting the Test set results
#Y_pred = classifier.predict_classes(X_test)
#results = pd.DataFrame({'actual':Y_test, 'predicted':Y_pred})
#results.to_csv("results.csv", index=False)

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