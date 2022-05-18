import pandas as pd
import numpy as np


def transform_data(df):
    '''
        segment data into exposed and control groups
        consider that SmartAd runs the experment hourly, group data into hours. 
            Hint: create new column to hold date+hour and use df.column.map(lambda x:  pd.Timestamp(x,tz=None).strftime('%Y-%m-%d:%H'))
        create two dataframes with bernouli series 1 for posetive(yes) and 0 for negative(no)
            Hint: Given engagement(sum of yes and no until current observation as an array) and success (yes countas an array), the method generates random binomial distribution
                #Example
                engagement = np.array([5, 3, 3])
                yes = np.array([2, 0, 3])
                Output is "[1] 1 0 1 0 0 0 0 0 1 1 1", showing a binary array of 5+3+3 values
                of which 2 of the first 5 are ones, 0 of the next 3 are ones, and all 3 of
                the last 3 are ones where position the ones is randomly distributed within each group.
    '''

    def get_bernouli_series(engagment_list, success_list):
        bernouli_series = []

        for engagment, success in zip(engagment_list, success_list):
            no_list = (engagment - success) * [0]
            yes_list = (success) * [1]
            series_item = yes_list + no_list
            random.shuffle(series_item)
            bernouli_series += series_item
        return np.array(bernouli_series)

    clean_df = df.query("not (yes == 0 & no == 0)")

    exposed = clean_df[clean_df['experiment'] == 'exposed']
    control = clean_df[clean_df['experiment'] == 'control']

    # group data into hours.
    control['hour'] = control['hour'].astype('str')
    control['date_hour'] = pd.to_datetime(control['date'] + " " + control['hour'] + ":00:00")
    control['date_hour'] = control['date_hour'].map(lambda x:  pd.Timestamp(x, tz=None).strftime('%Y-%m-%d:%H'))

    exposed['hour'] = exposed['hour'].astype('str')
    exposed['date_hour'] = pd.to_datetime( exposed['date'] + " " + exposed['hour'] + ":00:00")
    exposed['date_hour'] = exposed['date_hour'].map( lambda x:  pd.Timestamp(x, tz=None).strftime('%Y-%m-%d:%H'))

    # create two dataframes with bernouli series 1 for positive(yes) and 0 for negative(no)
    cont = exposed.groupby('date_hour').agg({'yes': 'sum', 'no': 'count'})
    cont = cont.rename(columns={'no': 'total'})
    control_bernouli = get_bernouli_series(
        cont['total'].to_list(), cont['yes'].to_list())

    exp = exposed.groupby('date_hour').agg({'yes': 'sum', 'no': 'count'})
    exp = exp.rename(columns={'no': 'total'})
    exposed_bernouli = get_bernouli_series(
        exp['total'].to_list(), exp['yes'].to_list())

    return control_bernouli, exposed_bernouli