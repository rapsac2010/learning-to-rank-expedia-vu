################################################################
###### Imports
################################################################

import configparser
from helpers.helper_functions import *
from helpers.helper_classes import *
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from joblib import dump, load

################################################################
###### Helper functions
################################################################

def construct_target_df(df, drop = True):
    df.loc[:, 'target'] = np.zeros(len(df))
    df.loc[df['click_bool'] == 1, 'target'] = 1
    df.loc[df['booking_bool'] == 1, 'target'] = 5
    if drop:
        return df.drop(['click_bool', 'booking_bool'], axis = 1)
    else:
        return df

def construct_datetime(df):
    df_out = df
    df_out['date_time'] = pd.to_datetime(df_out['date_time'])
    df_out['month'] = df_out['date_time'].dt.month
    df_out['day'] = df_out['date_time'].dt.day
    df_out['hour'] = df_out['date_time'].dt.hour
    df_out['day_of_week'] = df_out['date_time'].dt.weekday
    df_out['is_weekend'] = df_out['date_time'].dt.weekday.isin([5,6]).astype(int)
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

def add_normalized_column(df, col, group):
    df['norm_' + str(col) + "_" + str(group)] = (
        (df[col] - df.groupby(group)[col].transform('mean')) 
        / df.groupby(group)[col].transform('std')
    )
    return df

def balanced_sampling(df, num_samples = 4, target_str = 'target'):
    # Sort by target
    df = df.sort_values(by=['srch_id', target_str], ascending=[True, False], inplace=False)
    df = df.groupby('srch_id').head(num_samples)
    df = df.sample(frac=1)
    df = df.sort_values(by=['srch_id'], ascending=[True], inplace=False)    
    return df

def create_rank_feature(df, col):
    df['rank_' + str(col)] = df.groupby('srch_id')[col].rank(ascending=False)
    return df

def construct_desire(df, subject = 'prop_id', target = 'click_bool'):
    # aggregate on prop id, average target and rename column
    desire_df = df.groupby(subject)[target].mean().reset_index().rename(columns={target: 'desire_' + target})
    df = df.merge(desire_df, on=subject, how='left')
    return df, desire_df
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

# Construct target for training set
df = construct_target_df(df, drop = False)

# Construct datetime features for both sets
df = construct_datetime(df)
df_test = construct_datetime(df_test)

# Drop columns with more than 80% missing values
df, missing_cols = drop_missing_cols_thresholded(df, threshold = 0.99)
df_test = drop_cols(df_test, missing_cols)

# Add normalized columns
columns = ['price_usd', 'prop_starrating', 'prop_review_score', 'prop_location_score1', 'prop_location_score2']
indices = ['srch_id', 'prop_id', 'prop_country_id', 'srch_destination_id', 'srch_length_of_stay', 'srch_booking_window']
for column in tqdm(columns):
    for index in indices:
        df = add_normalized_column(df, column, index)
        df_test = add_normalized_column(df_test, column, index)

# Add rank features
rank_features = ['price_usd', 'prop_starrating', 'prop_review_score', 'prop_location_score1', 'prop_location_score2']
for feature in tqdm(rank_features):
    df = create_rank_feature(df, feature)
    df_test = create_rank_feature(df_test, feature)

# difference features
for df_cur in [df, df_test]:
    df_cur['usd_diff'] = abs(df_cur['visitor_hist_adr_usd'] - df_cur['price_usd'])
    df_cur['star_diff'] = abs(df_cur['visitor_hist_starrating'] - df_cur['prop_starrating'])
    df_cur['log_price_diff'] = df_cur['prop_log_historical_price'] - np.log(df_cur['price_usd'])

# Fill distance nan with mean
df['orig_destination_distance'].fillna(df['orig_destination_distance'].mean(), inplace=True)
df_test['orig_destination_distance'].fillna(df_test['orig_destination_distance'].mean(), inplace=True)

# Fill missing values with -1
df = df.fillna(-1)
df_test = df_test.fillna(-1)

# Fill inf values with -1
# df = df.replace([np.inf, -np.inf], -1)
# df_test = df_test.replace([np.inf, -np.inf], -1)

# Drop columns that leak information or are not useful (anymore)
leaky_cols = ['gross_bookings_usd', 'position', 'date_time', 'random_bool']
df = drop_cols(df, leaky_cols)
df_test = drop_cols(df_test, leaky_cols)

# Save data
print('Saving preprocessed data...')
df.to_parquet(config['PATH']['INT_DIR'] + '/training_set_preprocessed_nodrop.parquet')
df_test.to_parquet(config['PATH']['INT_DIR'] + '/test_set_preprocessed_nodrop.parquet')
