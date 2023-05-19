from joblib import dump, load
from tqdm import tqdm
from helpers.helper_classes import *
import pandas as pd
import numpy as np
import numpy as np
from sklearn.metrics import accuracy_score
import optuna
from optuna import Trial
from sklearn.model_selection import GroupShuffleSplit

def train_test_split(df, target_str, test_size=.2):
    splitter = GroupShuffleSplit(test_size=test_size, n_splits=2, random_state = 7)
    split = splitter.split(df, groups=df['srch_id'])
    train_inds, test_inds = next(split)

    df_ideal = df.iloc[test_inds].copy().sort_values(by=['srch_id', target_str], ascending=[True, False], inplace=False)

    X = df.drop([target_str], axis=1)
    y = df[target_str]
    X_train, X_test, y_train, y_test, test_ideal = X.iloc[train_inds], X.iloc[test_inds], y.iloc[train_inds], y.iloc[test_inds], df_ideal, 


    return X_train, X_test, y_train, y_test, test_ideal[['srch_id', 'prop_id', target_str]]
    
def train_val_test_split(df, target_str, test_size=.2, val_size=.2, random_state=7):
    splitter1 = GroupShuffleSplit(test_size=test_size, n_splits=1, random_state = random_state)
    split1 = splitter1.split(df, groups=df['srch_id'])
    train_val_inds, test_inds = next(split1)

    df_train_val = df.iloc[train_val_inds]
    df_test = df.iloc[test_inds]
    df_ideal = df.iloc[test_inds].copy().sort_values(by=['srch_id', target_str], ascending=[True, False], inplace=False)

    splitter2 = GroupShuffleSplit(test_size=val_size, n_splits=1, random_state = random_state)
    split2 = splitter2.split(df_train_val, groups=df_train_val['srch_id'])
    train_inds, val_inds = next(split2)

    df_train = df_train_val.iloc[train_inds].sort_values(by=['srch_id', target_str], ascending=[True, False], inplace=False)
    df_val = df_train_val.iloc[val_inds].sort_values(by=['srch_id', target_str], ascending=[True, False], inplace=False)
    df_test.sort_values(by=['srch_id', target_str], ascending=[True, False], inplace=True)

    X_train, X_val, X_test = df_train.drop([target_str], axis=1), df_val.drop([target_str], axis=1), df_test.drop([target_str], axis=1)
    y_train, y_val, y_test = df_train[target_str], df_val[target_str], df_test[target_str]
    
    return X_train, X_val, X_test, y_train, y_val, y_test, df_ideal[['srch_id', 'prop_id', target_str]]


def construct_pred_ideal(df_in, df_ideal, y_pred):
    df = df_in.copy()
    df['pred_grades'] = y_pred
    df = df.sort_values(by=['srch_id', 'pred_grades'], ascending=[True, False], inplace=False)

    # Merge grades from ideal on srch_id and prop_id
    df = df.merge(df_ideal, on=['srch_id', 'prop_id'], how='left')

    # Return srch_id, prop_id and pred_grades
    return df[['srch_id', 'prop_id', 'pred_grades', 'target']]

def construct_pred_submission(df_in, y_pred):
    df = df_in.copy()
    df['pred_grades'] = y_pred
    df = df.sort_values(by=['srch_id', 'pred_grades'], ascending=[True, False], inplace=False)

    # Return srch_id, prop_id and pred_grades
    return df[['srch_id', 'prop_id']]

def constructs_predictions(model, data, ideal_df = None):
    y_pred = model.predict_proba(data)
    pred_grades = y_pred @ [0, 1, 5]

    if ideal_df is not None:
        pred_df = construct_pred_ideal(data, ideal_df, pred_grades)
    else:
        pred_df = construct_pred_submission(data, pred_grades)
    return pred_df


def calc_NDCG(df_ideal, df_pred, k = 5):
    # Group by 5
    df_ideal = df_ideal.groupby('srch_id').head(k)
    df_pred = df_pred.groupby('srch_id').head(k)

    assert df_ideal.shape[0] % k == 0
    assert df_pred.shape[0] % k == 0
    
    # Get grades matrices
    ideal_grades = df_ideal['target'].values.reshape(int(df_ideal.shape[0] / k), k)
    pred_grades = df_pred['target'].values.reshape(int(df_pred.shape[0] / k), k)

    discount_vec = [1/np.log2(i+2) for i in range(k)]

    # Calculate NDCG
    NDCG = (pred_grades @ discount_vec).sum() / (ideal_grades @ discount_vec).sum()

    return NDCG

def construct_desire(df, subject = 'prop_id', target = 'click_bool'):
    # aggregate on prop id, average target and rename column
    desire_df = df.groupby(subject)[target].mean().reset_index().rename(columns={target: 'desire_' + target})
    df = df.merge(desire_df, on=subject, how='left')
    return df, desire_df

def merge_and_drop(df, desire_df_click, desire_df_book, drop = True):
    df = df.merge(desire_df_click, on='prop_id', how='left')
    df = df.merge(desire_df_book, on='prop_id', how='left')
    if drop:
        df.drop(['click_bool', 'booking_bool'], axis=1, inplace=True)
    df['desire_booking_bool'].fillna(0, inplace=True)
    df['desire_click_bool'].fillna(0, inplace=True)
    return df