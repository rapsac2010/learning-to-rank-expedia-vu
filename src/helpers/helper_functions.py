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
import scienceplots
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use(['science', 'ieee']) 
plt.rcParams['figure.dpi'] = 100


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

    assert df_ideal.shape[0] % k == 0, 'Number of rows in ideal is not a multiple of k'
    assert df_pred.shape[0] % k == 0, 'Number of rows in pred is not a multiple of k'
    assert df_ideal.shape[0] == df_pred.shape[0], 'Number of rows in ideal and pred are not equal'
    
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

def merge_and_drop(df, df_on_list, drop = True):

    # iterate over tuple list
    for df_cur, key in df_on_list:
        df = df.merge(df_cur, on=key, how='left', )
    if drop:
        df.drop(['click_bool', 'booking_bool'], axis=1, inplace=True)
    
    # df = df.fillna(-1)
    return df

import scipy.stats as stats
def plot_correlation_heatmap(df, include_columns=None, exclude_columns=None, figsize=(8,6), 
                             alternative_labels=None, rotate_labels=False, 
                             title=None, savename=None, show_significance=False):
    """
    Plots a correlation heatmap of a pandas DataFrame.

    :param df: pandas DataFrame
    :param include_columns: list of columns to include, default None (includes all columns)
    :param exclude_columns: list of columns to exclude, default None (excludes no columns)
    :param alternative_labels: list of alternative labels, default None (uses original column names)
    :param show_significance: bool, default False, shows significance level on heatmap
    """
    if include_columns is not None:
        # Filter DataFrame to only include specified columns
        df = df[include_columns]
    elif exclude_columns is not None:
        # Filter DataFrame to exclude specified columns
        df = df.drop(columns=exclude_columns)
    
    # Calculate the correlation matrix
    corr_matrix = df.corr()
    # round to 2 decimals
    corr_matrix = corr_matrix.round(2)

    if show_significance:
        p_matrix = df.corr(method=lambda x, y: stats.pearsonr(x, y)[1]) - np.eye(*corr_matrix.shape)
        p_matrix = p_matrix.applymap(lambda x: ''.join(['*' for t in [0.01, 0.05, 0.1] if x<=t]))
        annot_matrix = corr_matrix.astype(str) + p_matrix
    else:
        annot_matrix = corr_matrix.astype(str)

    # Set alternative labels if provided
    if alternative_labels is not None:
        if len(include_columns) != len(alternative_labels):
            raise ValueError("Length of include_columns and alternative_labels must be equal")
        corr_matrix.columns = alternative_labels
        corr_matrix.index = alternative_labels

    # Create a heatmap using seaborn
    plt.figure(figsize=figsize)
    sns.heatmap(corr_matrix, annot=annot_matrix, fmt='', cmap='coolwarm', vmin=-1, vmax=1)
    
    if title is not None:
        plt.title(title)

    if rotate_labels:
        # Rotate x labels
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
    if savename is not None:
        plt.savefig(savename, bbox_inches='tight')
    plt.show()
