import os
from random import random, randint
from mlflow import log_metric, log_param, log_artifacts

if __name__ == "__main__":
    # Log a parameter (key-value pair)
    log_param("param1", randint(0, 100))

    # Log a metric; metrics can be updated throughout the run
    log_metric("foo", random())
    log_metric("foo", random() + 1)
    log_metric("foo", random() + 2)

    # Log an artifact (output file)
    if not os.path.exists("outputs"):
        os.makedirs("outputs")
    with open("outputs/test.txt", "w") as f:
        f.write("hello world!")
    log_artifacts("outputs")
'''
import os
import warnings
import sys

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from scipy.stats import skew, norm
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import xgboost as xgb
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import seaborn as sb
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt 
from sklearn.tree import DecisionTreeClassifier
def model_run__log_mlfow(self, df, var_dict, other_dict = {}):
    '''
    self : rf regressor model
    df   : dataframe
    var_dict : model variables dict - var_dict["independant"], var_dict["dependant"]
    other_dict : other dict if needed, set to {} default
    '''
    r_name = other_dict["run_name"] 
    with mlflow.start_run(run_name=r_name) as run:

        # get current run and experiment id
        runID = run.info.run_uuid
        experimentID = run.info.experiment_id
        feature = var_dict["independant"]
        label   = var_dict["dependant"]

        ## log of predictions
        df[label] = np.log(df[label]+1)
        X = df[feature]
        y = df[label]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25,random_state = 42)
        self._rfr.fit(X_train, y_train)
        y_pred = self._rfr.predict(X_test)

        ## self.model is a getter for the model
        mlflow.sklearn.log_model(self.model, "catboost-reg-model")
        mlflow.log_params(self.params)
        model_score = self._rfr.score(X_test , y_test)
        mae = metrics.mean_absolute_error(y_test, y_pred)
        mse = metrics.mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = metrics.r2_score(y_test, y_pred)

        # Log metrics
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        print("-" * 100)
        print("Inside MLflow Run with run_id {} and experiment_id {}".format(runID, experimentID))
        print('Mean Absolute Error    :', mae)
        print('Mean Squared Error     :', mse)
        print('Root Mean Squared Error:', rmse)
        print('R2                     :', r2)
        return (experimentID, runID)