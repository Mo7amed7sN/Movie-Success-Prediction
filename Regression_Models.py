import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn import metrics
import numpy as np
import pickle

X = pd.read_csv('Encoded_X_testing_regression.csv')
Y = pd.read_csv('Encoded_y_testing_regression.csv')
X = X.drop(X.columns[0], axis=1)
Y = Y.drop(Y.columns[0], axis=1)

# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=10)
prediction = None


def Ridge_Model():
    global prediction
    cls = pickle.load(open('ridge_reg', 'rb'))
    prediction = cls.predict(X)


def linear():
    global prediction
    cls = pickle.load(open('linear_reg', 'rb'))
    # prediction = cls.predict(X)


def poly():
    global prediction
    poly_features = PolynomialFeatures(degree=1)
    cls = pickle.load(open('poly_reg', 'rb'))
    # y_train_predicted = cls.predict(X_train_poly)
    # prediction = cls.predict(poly_features.fit_transform(X))


Ridge_Model()
MSE = float(metrics.mean_squared_error(np.asarray(Y), np.asarray(prediction)))
print('Mean Square Error', MSE)
print("R2 score", abs(r2_score(Y, prediction)))
