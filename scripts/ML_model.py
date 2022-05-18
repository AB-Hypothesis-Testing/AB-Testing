# -*- coding: utf-8 -*-
import pandas as pd
from scipy.stats.stats import mode
#from scripts.ml_modelling_utils import evaluate_model, plot_roc_curve_log, plot_precision_recall_curve

"""
"""

def get_lr_model_score(model, x_train, y_train):
    return model.score(x_train, y_train)


def get_lr_params(model):
    return model.get_params()


def get_lr_features(model, x_train) -> pd.DataFrame:
    dataframe = pd.DataFrame()
    dataframe['Feature'] = x_train.columns
    dataframe['Score'] = model.coef_[0]
    dataframe.sort_values(by='Score', ascending=False, inplace=True)

    return dataframe


def get_lr_model_evaluation(model, x_valid, y_valid, x_test, y_test, show=False):
    return evaluate_model(model, x_valid, y_valid, x_test, y_test, show)


def get_lr_model_roc_curve_log(model, x_test, y_test, show=False):
    return plot_roc_curve_log(x_test, y_test, model, "Logistic Regression", show)


def get_lr_model_precision_recall_curve(model, x_test, y_test, show=False):
    return plot_precision_recall_curve(x_test, y_test, model, "Logistic Regression", show)


def get_model_evaluation(model_holder, x_valid, y_valid, x_test, y_test, show=False):
    model = get_model_best_estimator(model_holder)
    return evaluate_model(model, x_valid, y_valid, x_test, y_test, show)


def get_features(model_holder, x_train) -> pd.DataFrame:
    dataframe = pd.DataFrame()
    dataframe['Feature'] = x_train.columns
    dataframe['Score'] = model_holder.best_estimator_.feature_importances_[0]
    dataframe.sort_values(by='Score', ascending=False, inplace=True)

    return dataframe


def get_model_best_estimator(model_holder):
    return model_holder.best_estimator_


def get_model_best_score(model_holder):
    return model_holder.best_score_


def get_score_df(model_holder, top: int = 5, by: str = 'mean_test_score'):
    score_df = pd.DataFrame(model_holder.cv_results_)
    score_df = score_df.nlargest(top, by)

    return score_df


def get_best_model_parameters(model_holder):
    return model_holder.best_params_


def get_model_roc_curve_log(model_holder, x_test, y_test, label, show=False):
    model = get_model_best_estimator(model_holder)
    return plot_roc_curve_log(x_test, y_test, model, label, show)


def get_model_precision_recall_curve(model_holder, x_test, y_test, label, show=False):
    model = get_model_best_estimator(model_holder)
    return plot_precision_recall_curve(x_test, y_test, model, label, show)


if __name__ == '__main__':
    print("Works")
