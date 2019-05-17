__author__ = "Md. Ahsan Ayub"
__license__ = "GPL"
__credits__ = ["Ayub, Md. Ahsan", "Smith, Steven", "Tinker, Paul",
               "Siraj, Ambareen"]
__maintainer__ = "Md. Ahsan Ayub"
__email__ = "mayub42@students.tntech.edu"
__status__ = "Prototype"


# Modular function to apply decision tree classifier
def LR_classifier(X, Y, numFold):
    
    # Intilization of the figure
    myFig = plt.figure(figsize=[12,10])
    
    # Stratified K-Folds cross-validator
    cv = StratifiedKFold(n_splits=numFold,random_state=None, shuffle=False)
    
    # Initialization of the logistic regression classifier
    classifier = LogisticRegression(random_state=0)
    
    acc_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    
    i = 1
    for train, test in cv.split(X, Y):
        
        # Spliting the dataset
        X_train, X_test, Y_train, Y_test = X[train], X[test], Y[train], Y[test]
        
        # Fitting the classifier into training set
        classifier = classifier.fit(X_train, Y_train)
        
        # Breakdown of statistical measure based on classes
        Y_pred = classifier.predict(X_test)
        print(classification_report(Y_test, Y_pred, digits=4))
        
        # Compute the model's performance
        acc_scores.append(accuracy_score(Y_test, Y_pred))
        
        if(len(np.unique(Y)) > 2):
            f1_scores_temp = []
            f1_scores_temp.append(f1_score(Y_test, Y_pred, average=None))
            f1_scores.append(np.mean(f1_scores_temp))
            del f1_scores_temp
            
            precision_scores_temp = []
            precision_scores_temp.append(precision_score(Y_test, Y_pred, average=None))
            precision_scores.append(np.mean(precision_scores_temp))
            del precision_scores_temp
            
            recall_scores_temp = []
            recall_scores_temp.append(recall_score(Y_test, Y_pred, average=None))
            recall_scores.append(np.mean(recall_scores_temp))
            del recall_scores_temp
        
        else:
            f1_scores.append(f1_score(Y_test, Y_pred, average='binary'))
            precision_scores.append(precision_score(Y_test, Y_pred, average='binary'))
            recall_scores.append(recall_score(Y_test, Y_pred, average='binary'))
        
        
        if(len(np.unique(Y)) == 2):
            from sklearn.preprocessing import label_binarize
            Y_train = label_binarize(Y_train, classes = [i for i in range(len(np.unique(Y_train)))])
            Y_test = label_binarize(Y_test, classes = [i for i in range(len(np.unique(Y_test)))])
        
            probas_ = classifier.predict_proba(X_test)
            fpr, tpr, thresholds = roc_curve(Y_test, probas_[:, 1])
            tprs.append(interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            plt.plot(fpr, tpr, lw=1, color='black', alpha=0.5,
                     label='ROC fold %d (AUC = %0.3f)' % (i, roc_auc))
            print("Iteration ongoing inside LR method - KFold step: ", i)
            i += 1
    
    if(len(np.unique(Y)) == 2):    
        plt.plot([0,1],[0,1],linestyle = '--',lw = 1, alpha=0.5, color = 'black')
        
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        plt.plot(mean_fpr, mean_tpr, color='black',
                 label=r'Mean ROC (AUC = %0.3f)' % (mean_auc),
                 lw=2, alpha=0.8)
        
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate', fontsize=18, weight='bold')
        plt.ylabel('True Positive Rate', fontsize=18, weight='bold')
        plt.title('Receiver Operating Characteristic (ROC) Curve\nLogistic Regression with Bigram Model', fontsize=20, fontweight='bold')
        plt.legend(loc="lower right",fontsize=14)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.show()
        
        fileName = 'Logistic_Regression_ROC_' + str(numFold) + '_Fold.eps'
        # Saving the figure
        myFig.savefig(fileName, format='eps', dpi=1200)
    
    
    # Statistical measurement of the model
    print("Accuracy: ", np.mean(acc_scores))
    print("Precision: ", np.mean(precision_scores))
    print("Recall: ", np.mean(recall_scores))
    print("F1: ", np.mean(f1_scores))
    if(len(np.unique(Y)) == 2):
        print(acc_scores)
        print(precision_scores)
        print(recall_scores)
        print(f1_scores)
    
    
# Modular function to apply decision tree classifier
def DT_classifier(X, Y, numFold):
    
    # Intilization of the figure
    myFig = plt.figure(figsize=[12,10])
    
    # Stratified K-Folds cross-validator
    cv = StratifiedKFold(n_splits=numFold,random_state=None, shuffle=False)
    
    # Initialization of the decision tree classifier
    classifier = tree.DecisionTreeClassifier()
    
    acc_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    
    i = 1
    for train, test in cv.split(X, Y):
        
        # Spliting the dataset
        X_train, X_test, Y_train, Y_test = X[train], X[test], Y[train], Y[test]
        
        # Fitting the classifier into training set
        classifier = classifier.fit(X_train, Y_train)
        
        # Breakdown of statistical measure based on classes
        Y_pred = classifier.predict(X_test)
        print(classification_report(Y_test, Y_pred, digits=4))
        
        # Compute the model's performance
        acc_scores.append(accuracy_score(Y_test, Y_pred))
        if(len(np.unique(Y)) > 2):
            f1_scores_temp = []
            f1_scores_temp.append(f1_score(Y_test, Y_pred, average=None))
            f1_scores.append(np.mean(f1_scores_temp))
            del f1_scores_temp
            
            precision_scores_temp = []
            precision_scores_temp.append(precision_score(Y_test, Y_pred, average=None))
            precision_scores.append(np.mean(precision_scores_temp))
            del precision_scores_temp
            
            recall_scores_temp = []
            recall_scores_temp.append(recall_score(Y_test, Y_pred, average=None))
            recall_scores.append(np.mean(recall_scores_temp))
            del recall_scores_temp
        else:
            f1_scores.append(f1_score(Y_test, Y_pred, average='binary'))
            precision_scores.append(precision_score(Y_test, Y_pred, average='binary'))
            recall_scores.append(recall_score(Y_test, Y_pred, average='binary'))
        
        if(len(np.unique(Y)) == 2):
            probas_ = classifier.predict_proba(X_test)
            fpr, tpr, thresholds = roc_curve(Y_test, probas_[:, 1])
            tprs.append(interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
        
            aucs.append(roc_auc)
            plt.plot(fpr, tpr, lw=1, color='black', alpha=0.5,
                     label='ROC fold %d (AUC = %0.3f)' % (i, roc_auc))
            print("Iteration ongoing inside DT method - KFold step: ", i)
            i += 1
        
    if(len(np.unique(Y)) == 2):
        plt.plot([0,1],[0,1],linestyle = '--',lw = 1, alpha=0.5, color = 'black')
        
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        plt.plot(mean_fpr, mean_tpr, color='black',
                 label=r'Mean ROC (AUC = %0.3f)' % (mean_auc),
                 lw=2, alpha=0.8)
        
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate', fontsize=18, weight='bold')
        plt.ylabel('True Positive Rate', fontsize=18, weight='bold')
        plt.title('Receiver Operating Characteristic (ROC) Curve\nDecision Tree with Bigram Model', fontsize=20, fontweight='bold')
        plt.legend(loc="lower right",fontsize=14)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.show()
        
        fileName = 'Decision_Tree_ROC_' + str(numFold) + '_Fold.eps'
        # Saving the figure
        myFig.savefig(fileName, format='eps', dpi=1200)
    
    # Statistical measurement of the model
    print("Accuracy: ", np.mean(acc_scores))
    print("Precision: ", np.mean(precision_scores))
    print("Recall: ", np.mean(recall_scores))
    print("F1: ", np.mean(f1_scores))
    if(len(np.unique(Y)) == 2):
        print(acc_scores)
        print(precision_scores)
        print(recall_scores)
        print(f1_scores)

# Modular function to apply decision tree classifier
def RF_classifier(X, Y, numFold):
    
    # Intilization of the figure
    myFig = plt.figure(figsize=[12,10])
    
    # Stratified K-Folds cross-validator
    cv = StratifiedKFold(n_splits=numFold,random_state=None, shuffle=False)
    
    # Initialization of the random forest classifier
    classifier = RandomForestRegressor(n_estimators = 100, random_state = 0)
    
    acc_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    
    i = 1
    for train, test in cv.split(X, Y):
        # Spliting the dataset
        X_train, X_test, Y_train, Y_test = X[train], X[test], Y[train], Y[test]
        
        # Fitting the classifier into training set
        classifier = classifier.fit(X_train, Y_train)
        
        # Breakdown of statistical measure based on classes
        Y_pred = classifier.predict(X_test)
        print(classification_report(Y_test, Y_pred, digits=4))
        
        # Compute the model's performance
        acc_scores.append(accuracy_score(Y_test, Y_pred))
        
        if(len(np.unique(Y)) > 2):
            f1_scores_temp = []
            f1_scores_temp.append(f1_score(Y_test, Y_pred, average=None))
            f1_scores.append(np.mean(f1_scores_temp))
            del f1_scores_temp
            
            precision_scores_temp = []
            precision_scores_temp.append(precision_score(Y_test, Y_pred, average=None))
            precision_scores.append(np.mean(precision_scores_temp))
            del precision_scores_temp
            
            recall_scores_temp = []
            recall_scores_temp.append(recall_score(Y_test, Y_pred, average=None))
            recall_scores.append(np.mean(recall_scores_temp))
            del recall_scores_temp
        
        else:
            f1_scores.append(f1_score(Y_test, Y_pred, average='binary'))
            precision_scores.append(precision_score(Y_test, Y_pred, average='binary'))
            recall_scores.append(recall_score(Y_test, Y_pred, average='binary'))
        
        if(len(np.unique(Y)) == 2):
            probas_ = classifier.predict_proba(X_test)
            fpr, tpr, thresholds = roc_curve(Y_test, probas_[:, 1])
            tprs.append(interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
        
            aucs.append(roc_auc)
            plt.plot(fpr, tpr, lw=1, color='black', alpha=0.5,
                     label='ROC fold %d (AUC = %0.3f)' % (i, roc_auc))
            print("Iteration ongoing inside RF method - KFold step: ", i)
            i += 1
        
    if(len(np.unique(Y)) == 2):
        plt.plot([0,1],[0,1],linestyle = '--',lw = 1, alpha=0.5, color = 'black')
        
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        plt.plot(mean_fpr, mean_tpr, color='black',
                 label=r'Mean ROC (AUC = %0.3f)' % (mean_auc),
                 lw=2, alpha=0.8)
        
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate', fontsize=18, weight='bold')
        plt.ylabel('True Positive Rate', fontsize=18, weight='bold')
        plt.title('Receiver Operating Characteristic (ROC) Curve\nRandom Forest with Bigram Model', fontsize=20, fontweight='bold')
        plt.legend(loc="lower right",fontsize=14)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.show()
        
        fileName = 'Random_Forest_ROC_' + str(numFold) + '_Fold.eps'
        # Saving the figure
        myFig.savefig(fileName, format='eps', dpi=1200)
    
    # Statistical measurement of the model
    print("Accuracy: ", np.mean(acc_scores))
    print("Precision: ", np.mean(precision_scores))
    print("Recall: ", np.mean(recall_scores))
    print("F1: ", np.mean(f1_scores))
    if(len(np.unique(Y)) == 2):
        print(acc_scores)
        print(precision_scores)
        print(recall_scores)
        print(f1_scores)
    
    
# Modular function to apply artificial neural network 
def ANN_classifier(X, Y, batchSize, epochCount):
    
    # Spliting the dataset into the Training and Test Set
    from sklearn.model_selection import train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42, stratify=Y)
    
    # Initializing the ANN
    classifier = Sequential()
    
    # Adding the input layer and the first hidden layer
    classifier.add(Dense(output_dim = round(X.shape[1]/2), init =  'uniform', activation = 'relu', input_dim = X.shape[1]))
    
    # Adding the second hidden layer
    classifier.add(Dense(output_dim = round(X.shape[1]/2), init =  'uniform', activation = 'relu'))
    
    # Adding the output layer
    classifier.add(Dense(output_dim = 1, init =  'uniform', activation = 'sigmoid'))
    
    # Compiling the ANN
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    # Callback to stop if validation loss does not decrease
    callbacks = [EarlyStopping(monitor='val_loss', patience=2)]

    # Fitting the ANN to the Training set
    classifier.fit(X_train,
                   Y_train,
                   callbacks=callbacks,
                   validation_split=0.2,
                   batch_size = batchSize,
                   epochs = epochCount,
                   shuffle=True)
    
    # ------ Evaluation -------

    print("ANN using Bigram Model")
        
    # Predicting the Test set results
    Y_pred = classifier.predict_classes(X_test)
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
    

# Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Libraries relevant to performance metrics
from sklearn.metrics import roc_curve, auc, classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold
from scipy import interp

# Libraries relevant to supervised learning 
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping

#importing the data set
dataset = pd.read_csv('Dataset/master_dataset.csv')
print(dataset.head())

# Compute the length of the dataset
totalRecords = len(dataset.index)

# One Hot Encode the TLD column
df = dataset.copy(deep=True)
df = df[['TLD']]
df = pd.get_dummies(df,prefix=['TLD'])

# Concating the one hot encodded dataframe to main dataframe
dataset = pd.concat([dataset, df], axis=1)
dataset = dataset.drop(columns=['TLD'])

# Processing the domain names (text)
import re
corpus = []
for i in range(0,totalRecords):
    domains = re.sub('[.]', ' ', dataset['domain'][i])
    domains = domains.lower()
    domains = domains.split()
    domains = ' '.join(domains)
    corpus.append(domains)
    
# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(analyzer='char', ngram_range=(2, 2)) #bigram initialization
X = cv.fit_transform(corpus).toarray()  # X obtains the corups
Y_class = dataset.iloc[:,dataset.columns.get_loc("class")].values
Y_family = dataset.iloc[:,dataset.columns.get_loc("family_id")].values.astype(int)

# Drop two Y columns from the dataset as well as the domain string column from the dataset
dataset = dataset.drop(columns=['class', 'family_id', 'domain'])

# Concat and create the X properly for the last time
X_temp = dataset.iloc[:,:].values
X = np.column_stack([X, X_temp])

# Clear the memories
del dataset
del df
del X_temp
del corpus

print("Data are processed and ready for classification.")

# Calling the logistic regression classifier for binary classification with
# 5-fold cross validation
LR_classifier(X, Y_class, 5)

# Calling the logistic regression classifier for malware family detection
#  with 5-fold cross validation
LR_classifier(X, Y_family, 5)

# Calling the decision tree classifier for binary classification with
# 5-fold cross validation
DT_classifier(X, Y_class, 5)

# Calling the random forest classifier for binary classification with
# 5-fold cross validation
RF_classifier(X, Y_class, 5)

# Calling the ANN with batch_size 64 and epoch 1
ANN_classifier(X, Y_class, 64, 1)