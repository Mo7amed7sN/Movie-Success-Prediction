import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from ast import literal_eval
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import pickle
from sklearn.metrics import r2_score
from sklearn.metrics import confusion_matrix
from datetime import datetime

X = pd.read_csv('Encoded_X_testing_classification.csv')
Y = pd.read_csv('Encoded_y_testing_classification.csv')
X = X.drop(X.columns[0], axis=1)
Y = Y.drop(Y.columns[0], axis=1)

# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
prediction = None


def Dtree_adaboost():
    global prediction
    # bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=10), algorithm="SAMME.R", n_estimators=100)
    # start_time = datetime.now()
    # bdt.fit(X_train, y_train)
    # elapsed = datetime.now() - start_time
    # pickle.dump(bdt, open('adaboost_dtree', 'wb'))

    bdt = pickle.load(open('adaboost_dtree', 'rb'))
    prediction = bdt.predict(X)


def SVM():
    global prediction
    # start_time = datetime.now()
    # svm_model_linear_ovo = SVC(kernel='poly', degree=4, C=10000000).fit(X, Y)
    # elapsed = datetime.now() - start_time
    # pickle.dump(svm_model_linear_ovo, open('svm', 'wb'))

    svm_model_linear_ovo = pickle.load(open('svm', 'rb'))
    prediction = svm_model_linear_ovo.predict(X)


def Dtree():
    global prediction
    # clf = tree.DecisionTreeClassifier(max_depth=8)
    # start_time = datetime.now()
    # clf.fit(X, Y)
    # elapsed = datetime.now() - start_time
    # pickle.dump(clf, open('dtree', 'wb'))

    clf = pickle.load(open('dtree', 'rb'))
    prediction = clf.predict(X)


def KNN():
    global prediction
    # knn = KNeighborsClassifier(n_neighbors=27)
    # start_time = datetime.now()
    # knn.fit(X, Y)
    # elapsed = datetime.now() - start_time
    # pickle.dump(knn, open('knn', 'wb'))

    knn = pickle.load(open('knn', 'rb'))
    prediction = knn.predict(X)


Dtree()
print("Accuracy : "+str(float(accuracy_score(Y, prediction)) * 100))
print("confusion_matrix", confusion_matrix(Y, prediction))
