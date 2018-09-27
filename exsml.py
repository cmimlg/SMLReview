from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
import numpy as np
import math
from math import sqrt
import GPy
from GPy.kern import *
import logging
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.grid_search import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
import os
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import SelectFromModel


# create logger for the application
logger = logging.getLogger('Example Application Logger')
ch = logging.StreamHandler()

# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)


logger.addHandler(ch)
logger.setLevel(logging.DEBUG)

def read_data():
    os.chdir("/home/admin123/Overview_BDML/example")
    fp_train = "boston_housing_train.csv"
    fp_test = "boston_housing_test.csv"
    df_train = pd.read_csv(fp_train)
    df_test = pd.read_csv(fp_test)
    col_names = df_train.columns.tolist()
    preds = list(set(col_names) - set(["MEDV"]))
    X_train = df_train[preds].as_matrix()
    X_test = df_test[preds].as_matrix()
    y_train = df_train["MEDV"].as_matrix()
    y_test = df_test["MEDV"].as_matrix()
    


    return X_train, y_train, X_test, y_test




def fit_GP(show_fit = True):
    X_train, y_train, X_test, y_test = read_data()
    y_train = np.reshape(y_train, (y_train.shape[0], 1))
    y_test = np.reshape(y_test, (y_test.shape[0], 1))
    k = RBF(input_dim = 13, ARD = True) +\
        Linear(input_dim = 13, ARD = True)
    m  = GPy.models.GPRegression(X_train, y_train, k)
    m.optimize()

    ytp = m.predict(X_test)[0]
    se = (ytp - y_test) **2
    Nt = float(X_test.shape[0])
    mse = np.sum(se)/Nt
    logger.info("GP MSE is : " + str(mse))
    if show_fit:
        plt.title("GP Fit ")
        plt.scatter(y_test, ytp)
        plt.xlabel("Observed Value")
        plt.ylabel("Predicted Value")
        plt.grid(True)
        plt.show()

    return 

def fit_LR(show_fit = True):
    X_train, y_train, X_test, y_test = read_data()
    the_alphas = np.array([0.0001, 0.001, 0.01, 0.1, 10])
    reg1 = LassoCV(alphas = the_alphas)
    reg1.fit(X_train, y_train)
    ytp = reg1.predict(X_test)
    
##    sfm = SelectFromModel(reg1)
##    sfm = sfm.fit(X_train, y_train)
##    xtrns_train  = sfm.transform(X_train)
##    xtrns_test = sfm.transform(X_test)
##    reg2 = LinearRegression()
##    reg2 = reg2.fit(xtrns_train, y_train)
##    ytp = reg2.predict(xtrns_test)
    se = (ytp - y_test) **2
    Nt = float(X_test.shape[0])
    mse = np.sum(se)/Nt
    logger.info("LASSO MSE is : " + str(mse))
    if show_fit:
        plt.title("LASSO Fit")
        plt.scatter(y_test, ytp)
        plt.xlabel("Observed Value")
        plt.ylabel("Predicted Value")
        plt.grid(True)
        plt.show()
    return 

def fit_SVR():
    X_train, y_train, X_test, y_test = read_data()
    # Fit regression models
    svr = GridSearchCV(SVR(kernel='rbf'),\
                   cv=5, param_grid={"C": [1e-1, 1e0, 1e1, 1e2],\
                                     "gamma": np.logspace(-2, 2, 10)})


    ytp = svr.fit(X_train, y_train).predict(X_test)

    se = (ytp - y_test) **2
    Nt = float(X_test.shape[0])
    rmse = sqrt(np.sum(se)/Nt)

    return rmse

def fit_DTR(depth = 5):
    X_train, y_train, X_test, y_test = read_data()
    regressor = DecisionTreeRegressor(random_state=0, max_depth = depth)
    ytp = regressor.fit(X_train, y_train).predict(X_test)
    se = (ytp - y_test) **2
    Nt = float(X_test.shape[0])
    rmse = sqrt(np.sum(se)/Nt)

    return rmse

def plot_DTR():
    os.chdir("/home/admin123/Overview_BDML/example")
    fp = "bh_test_pred.csv"
    df = pd.read_csv(fp)
    ytp = df["Pred"].as_matrix()
    yt = df["Test"].as_matrix()
    se = (ytp - yt) **2
    Nt = float(df.shape[0])
    mse = np.sum(se)/Nt
    logger.info("DT MSE is : " + str(mse))
    plt.title("Decision Tree Fit")
    plt.scatter(yt, ytp)
    plt.xlabel("Observed Value")
    plt.ylabel("Predicted Value")
    plt.grid(True)
    plt.show()

    return
