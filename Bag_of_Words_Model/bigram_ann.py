__author__ = "Md. Ahsan Ayub"
__license__ = "GPL"
__credits__ = ["Ayub, Md. Ahsan", "Smith, Steven", "Yilmiz, Ibrahim",
               "Siraj, Ambareen"]
__maintainer__ = "Md. Ahsan Ayub"
__email__ = "mayub42@students.tntech.edu"
__status__ = "Prototype"

# Importing the libraries
import pandas as pd

# Disabling Tensorflow warning, doesn't enable AVX/FMA
#import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#importing the data set
dataset = pd.read_csv('sample_production_data_97K.csv')
print(dataset.head())
print("Bigram | ANN | Size of the Dataset ", len(dataset))

# ------ Processing the Data -------

# Processing the domain names (text)
import re
corpus = []
for i in range(0,97734):
    domains = re.sub('[.]', ' ', dataset['Domain'][i]);
    domains = domains.lower()
    domains = domains.split()
    domains = ' '.join(domains)
    corpus.append(domains)
print("Done with creating corpus")
    
# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(analyzer='char', ngram_range=(2, 2))
X = cv.fit_transform(corpus).toarray()
Y = dataset.iloc[:,1].values
print("Done with creating X and Y")
obs, feats = X.shape

# Spliting the dataset into the Training and Test Set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
print("Dataset are split into Training and Test")

# ------ Making the ANN model -------

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
print("Keras libraries are imported")

# Initializing the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = round(feats/2), init =  'uniform', activation = 'relu', input_dim = feats))

# Adding the second hidden layer
classifier.add(Dense(output_dim = round(feats/2), init =  'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(output_dim = 1, init =  'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

print("All set for EPOCHS")
# Fitting the ANN to the Training set
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