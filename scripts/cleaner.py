import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import sys


class Cleaner:
    """
    - this class is responsible for performing 
    Exploratory Data Analysis
    """

    def __init__(self, df):
        """initialize the eda class"""
        self.df = df

    def descriptive_stats(self, describe=False, info=False, size=False):
        """
        expects: 
            - boolean
        returns:
            - summary
        """
        summary = None
        if describe:
            summary = self.df.describe()
        elif info:
            summary = self.df.info
        elif size:
            summary = self.df.shape
        return summary

    def has_missing_values(self):
        """
        expects:
            -   nothing
        returns:
            -   boolean
        """
        has_missing_values = False
        if True in self.df.isnull().any().to_list():
            has_missing_values = True
        return has_missing_values

    def get_df(self):
        """
        - returns the dataframes
        """
        return self.df


if __name__ == '__main__':
    file_path = sys.argv[1]
    df = pd.read_csv(file_path)
    cleaner = cleaner(df)
    cleaner_df = cleaner.get_df()
    cleaner_df.to_csv("../data/eda.csv", index=False)