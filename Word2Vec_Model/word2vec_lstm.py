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
import matplotlib.pyplot as plt


BINARY = False

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

Y_class = dataset['class'].values
Y_family_id = dataset['family_id'].values

X = np.zeros([len(corpus), number_of_words], dtype=np.int32)
for i, sentence in enumerate(corpus):
    for t, word in enumerate(sentence):
        X[i, t] = word2idx(word)


# One Hot Encode the TLD column
df = dataset.copy(deep=True)
df = df[['TLD']]
df = pd.get_dummies(df,prefix=['TLD'])

# Concating the one hot encodded dataframe to main dataframe
dataset = pd.concat([dataset, df], axis=1)
dataset = dataset.drop(columns=['TLD', 'domain', 'class', 'family_id'])

del(df)

X_features = dataset.iloc[:,:].values


if BINARY:
    Y = Y_class
else:
    Y = Y_family_id
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
from keras.layers import Dense, Activation, Dropout, Input
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam

from sklearn.metrics import (roc_curve, auc, classification_report, 
                            accuracy_score, f1_score, precision_score, 
                            recall_score, confusion_matrix)

num_classes = len(np.unique(Y_train))
# Word index input
main_input = Input(shape=(number_of_words,))
# Embedding layer using Word2Vec weights
embed = Embedding(input_dim=vocab_size, output_dim=embedding_size, weights=[pretrained_weights])(main_input)
# LSTM layer
lstm = LSTM(units=embedding_size, activation='relu', recurrent_dropout=0.9, dropout=0.9)(embed)
# LSTM layer aux output to allow smooth training with concatenated inputs
aux_out = Dense(1, activation='sigmoid', name='aux_out')(lstm)
# Other feature input
feature_input = Input(shape=(len(dataset.columns),), name='feature_input')
# Concatenate input layers
conc = keras.layers.concatenate([lstm, feature_input])
# Dropout layer to prevent overfitting
drop = Dropout(0.9)(conc)
# One hidden layer to increase model complexity
hidden_1 = Dense(units=64, activation='relu', name='hidden_1')(conc)
# Output layer.  
if num_classes > 2:
    units = len(np.unique(Y_family_id))
    activation = 'softmax'
    model_loss = 'sparse_categorical_crossentropy'
    graph_type = 'Multiclass'
    lr = 0.001
    epochs = 50
else:
    units = 1
    activation = 'sigmoid'
    model_loss = 'binary_crossentropy',
    graph_type = 'Binary'
    lr = 0.0001,
    epochs = 10
    
dense = Dense(units=units, activation='softmax', name='dense')(hidden_1)
# Define the model
classifier = Model(inputs=[main_input, feature_input], outputs=[dense, aux_out])
# Adam optimizer with lowered lr to prevent overfitting
opt = Adam(lr=0.001)
# Compiling the LSTM
classifier.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics = ['accuracy'])
print(classifier.summary())
# Callback to stop if validation loss does not decrease
callbacks = [EarlyStopping(monitor='val_dense_loss', patience=2)]

# Fitting the LSTM to the Training set
history = classifier.fit(
               [X_embed_train, X_features_train], 
               [Y_train, Y_train],
               validation_split=0.1,
               batch_size = 128, 
               epochs = epochs,
               callbacks=callbacks,
               shuffle=True)


# ------ Evaluation -------

Probabilities = classifier.predict([X_embed_test, X_features_test])
# Probabilities is a list of probs for both output layers.  We only want the dense layer, thus we use index 0
Y_prob = Probabilities[0]
if num_classes == 2:
    Y_pred = np.where(Y_prob > 0.5, 1, 0)
else:
    Y_pred = np.argmax(Y_prob, 1)

print(classification_report(Y_test, Y_pred, digits=4))

myFig = plt.figure(figsize=[12,10])

plt.plot(history.history['dense_acc'], linestyle = ':',lw = 2, alpha=0.8, color = 'black')
plt.plot(history.history['val_dense_acc'], linestyle = '--',lw = 2, alpha=0.8, color = 'black')
plt.title('Accuracy over Epoch\nLSTM ' + graph_type, fontsize=20, weight='bold')
plt.ylabel('Accuracy', fontsize=18, weight='bold')
plt.xlabel('Epoch', fontsize=18, weight='bold')
plt.legend(['Train', 'Validation'], loc='lower right', fontsize=14)
plt.xticks(ticks=range(0, len(history.history['dense_acc'])))
  
plt.yticks(fontsize=16)
plt.show()

fileName = 'LSTM_Accuracy_over_Epoch_' + graph_type + '_Classification.eps'

myFig.savefig(fileName, format='eps', dpi=1200)
plt.clf()

plt.plot(history.history['dense_loss'], linestyle = ':',lw = 2, alpha=0.8, color = 'black')
plt.plot(history.history['val_dense_loss'], linestyle = '--',lw = 2, alpha=0.8, color = 'black')
plt.title('Loss over Epoch\nLSTM ' + graph_type, fontsize=20, weight='bold')
plt.ylabel('Loss', fontsize=18, weight='bold')
plt.xlabel('Epoch', fontsize=18, weight='bold')
plt.legend(['Train', 'Validation'], loc='upper right', fontsize=14)
plt.xticks(ticks=range(0, len(history.history['dense_loss'])))
  
plt.yticks(fontsize=16)
plt.show()

fileName = 'LSTM_Loss_over_Epoch_' + graph_type + '_Classification.eps'

myFig.savefig(fileName, format='eps', dpi=1200)
plt.clf()

if num_classes == 2:
    fpr, tpr, _ = roc_curve(Y_test, Y_prob)
    plt.plot(fpr, tpr, color='black',
             label=r'ROC (AUC = %0.3f)' % (auc(fpr, tpr)),
             lw=2, alpha=0.8)
        
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate', fontsize=18, weight='bold')
    plt.ylabel('True Positive Rate', fontsize=18, weight='bold')
    plt.title('Receiver Operating Characteristic (ROC) Curve\nLSTM', fontsize=20, fontweight='bold')
    plt.legend(loc="lower right",fontsize=14)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.show()
        
    fileName = 'LSTM_ROC.eps'
    # Saving the figure
    myFig.savefig(fileName, format='eps', dpi=1200)
    plt.clf()
# Making the cufusion Matrix

#print(classification_report(Y_test, Y_pred, digits=4))
    
# Compute the model's performance

# Making the confusion Matrix
cm = confusion_matrix(Y_test, Y_pred)
print("Confusion Matrix:\n", cm)
print("Accuracy: ", accuracy_score(Y_test, Y_pred))

if num_classes == 2:
    print("F1: ", f1_score(Y_test, Y_pred, average='binary'))
    print("Precison: ", precision_score(Y_test, Y_pred, average='binary'))
    print("Recall: ", recall_score(Y_test, Y_pred, average='binary'))
else:
    print("F1: ",f1_score(Y_test, Y_pred, average=None))
    print("Precison: ", precision_score(Y_test, Y_pred, average=None))
    print("Recall: ", recall_score(Y_test, Y_pred, average=None))