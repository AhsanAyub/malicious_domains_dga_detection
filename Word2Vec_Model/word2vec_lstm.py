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
dataset = pd.read_csv('D:\\Research\\malicious_domains_dga\\Dataset\\master_dataset.csv')
print(dataset.head())


# ------ Processing the Data -------

# Processing the domain names (text)
import re
number_of_words = 0
number_of_obs = len(dataset)
corpus = []

for i in range(0,number_of_obs):
    domains = re.sub('[.]', ' ', dataset['domain'][i]);
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

Y_family_id = dataset['family_id'].values
# One Hot Encode the TLD column
df = dataset.copy(deep=True)
df = df[['TLD']]
df = pd.get_dummies(df,prefix=['TLD'])

# Concating the one hot encodded dataframe to main dataframe
dataset = pd.concat([dataset, df], axis=1)
dataset = dataset.drop(columns=['TLD', 'domain', 'class', 'family_id'])

del(df)

X_features = dataset.iloc[:,:].values
#X = np.column_stack([X, X_temp])

# Spliting the dataset into the Training and Test Set
from sklearn.model_selection import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2)
for train_idx, test_idx in sss.split(X, Y):
    X_embed_train, X_embed_test = X[train_idx], X[test_idx]
    Y_train, Y_test = Y[train_idx], Y[test_idx]
    X_features_train, X_features_test = X_features[train_idx], X_features[test_idx]



# ------ Making the LSTM model -------

# Importing the Keras libraries and packages
import keras
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.layers import Dense, Activation, Dropout, Input, Concatenate
from keras.models import Model
from keras.callbacks import EarlyStopping

main_input = Input(shape=(number_of_words,))
embed = Embedding(input_dim=vocab_size, output_dim=embedding_size, weights=[pretrained_weights])(main_input)
lstm = LSTM(units=embedding_size, activation='relu', recurrent_dropout=0.8, dropout=0.8)(embed)
aux_out = Dense(1, activation='sigmoid', name='aux_out')(lstm)

feature_input = Input(shape=(len(dataset.columns),), name='feature_input')
conc = keras.layers.concatenate([lstm, feature_input])
drop = Dropout(0.8)(conc)


dense = Dense(units=1, activation='sigmoid', name='dense')(drop)
classifier = Model(inputs=[main_input, feature_input], outputs=[dense, aux_out])


# Compiling the LSTM
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics = ['accuracy'])

# Callback to stop if validation loss does not decrease
callbacks = [EarlyStopping(monitor='val_loss', patience=2)]

# Fitting the LSTM to the Training set
history = classifier.fit(
               [X_embed_train, X_features_train], 
               [Y_train, Y_train],
               validation_split=0.1,
               batch_size = 128, 
               epochs = 10,
               shuffle=True)


# ------ Evaluation -------
Probabilities = classifier.predict([X_embed_test, X_features_test])
Y_prob = Probabilities[0]
# Y_prob is a list of predictions for both output layers.  We only want the dense layer, thus we use index 0
Y_pred = np.where(Y_prob > 0.5, 1, 0)
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