__author__ = "Md. Ahsan Ayub"
__license__ = "GPL"
__credits__ = ["Ayub, Md. Ahsan", "Smith, Steven", "Yilmiz, Ibrahim",
               "Siraj, Ambareen"]
__maintainer__ = "Md. Ahsan Ayub"
__email__ = "mayub42@students.tntech.edu"
__status__ = "Prototype"

# Importing the libraries
import pandas as pd

#importing the data set
dataset = pd.read_csv('sample_production_data_97K.csv')
print(dataset.head())

#Set dataCount to the record count of the CSV file.
dataCount = dataset.shape[0]

# Processing the domain names (text)
import re
corpus = []
for i in range(0,dataCount):
    domains = re.sub('[.]', ' ', dataset['Domain'][i])
    domains = domains.lower()
    domains = domains.split()
    domains = ' '.join(domains)
    corpus.append(domains)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
maxFeatures = 15000;
cv = CountVectorizer(max_features=maxFeatures)
X = cv.fit_transform(corpus).toarray()
Y = dataset.iloc[:,1].values

#PCA Section
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X= sc.fit_transform(X)


from sklearn.decomposition import PCA

#Cut maxFeatures in half with PCA.
pca = PCA(n_components = int(maxFeatures/2))

principalComponents = pca.fit_transform(X)
principalDf = pd.DataFrame(data = principalComponents)

X = principalDf

# Spliting the dataset into the Training and Test Set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)



# Fitting the dataset into the Training set (Implementing Naive Bayes Tree)
from sklearn.naive_bayes import GaussianNB
naive_bayes_classifier = GaussianNB()
naive_bayes_classifier.fit(X_train, Y_train)
print("Naive Bayes | Bag of Words Model")


# ------ Evaluation -------

# Predicting the Test set results
Y_pred = naive_bayes_classifier.predict(X_test)
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