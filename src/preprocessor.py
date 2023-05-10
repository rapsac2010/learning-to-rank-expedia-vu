################################################################
###### Imports
################################################################

import configparser
from helpers.helper_functions import *
from helpers.helper_classes import *
import pandas as pd
import numpy as np
from joblib import dump, load

################################################################
###### Helper functions
################################################################

def construct_target_df(df):
    df.loc[:, 'target'] = np.zeros(len(df))
    df.loc[df['click_bool'] == 1, 'target'] = 1
    df.loc[df['booking_bool'] == 1, 'target'] = 5
    return df.drop(['click_bool', 'booking_bool'], axis = 1)

def construct_datetime(df):
    df_out = df
    df_out['date_time'] = pd.to_datetime(df_out['date_time'])
    df_out['month'] = df_out['date_time'].dt.month
    df_out['day'] = df_out['date_time'].dt.day
    df_out['hour'] = df_out['date_time'].dt.hour
    return df_out

def drop_missing_cols_thresholded(df, threshold = 0.8):
    missing_cols = df.columns[df.isna().any()].tolist()
    missing_cols = [col for col in missing_cols if df[col].isna().sum() / len(df) > threshold]
    return df.drop(missing_cols, axis = 1), missing_cols

def drop_cols(df, cols):

    # check which cols are in df
    cols_present = [col for col in cols if col in df.columns]

    # If cols and cols_present mismatch, print warning and which cols are not present
    if len(cols) != len(cols_present):
        print('Warning: not all columns to be dropped are present in df')
        print('Missing columns: ', [col for col in cols if col not in df.columns])
    
    return df.drop(cols_present, axis=1)

################################################################
###### Data loading
################################################################

# Read config.ini file
config = configparser.ConfigParser()
config.read('src/config.ini')

# Read dataframes from parquet
print('Loading data...')
df = pd.read_parquet(config['PATH']['DATA_DIR'] + '/training_set.parquet', engine = 'fastparquet')
df_test = pd.read_parquet(config['PATH']['DATA_DIR'] + '/test_set.parquet', engine = 'fastparquet')

# Construct target for training set
print('Preprocessing...')
df = construct_target_df(df)

# Construct datetime features for both sets
df = construct_datetime(df)
df_test = construct_datetime(df_test)

# Drop columns with more than 80% missing values
df, missing_cols = drop_missing_cols_thresholded(df, threshold = 0.99)
df_test = drop_cols(df_test, missing_cols)

# Fill missing values with -1
df = df.fillna(-1)
df_test = df_test.fillna(-1)

# Drop columns that leak information or are not useful (anymore)
leaky_cols = ['gross_bookings_usd', 'position', 'date_time', 'random_bool']
df = drop_cols(df, leaky_cols)
df_test = drop_cols(df_test, leaky_cols)

# To parquet
print('Saving preprocessed data...')
df.to_parquet(config['PATH']['INT_DIR'] + '/training_set_preprocessed.parquet')
df_test.to_parquet(config['PATH']['INT_DIR'] + '/test_set_preprocessed.parquet')