from sklearn.neighbors import KNeighborsRegressor
import optuna
import pickle
import os
import configparser
from tqdm import tqdm
from helpers.helper_functions import *
from helpers.helper_classes import *
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from wandb.lightgbm import wandb_callback, log_summary
import wandb

# Read config.ini file
config = configparser.ConfigParser()
config.read('src/config.ini')
os.chdir(config['PATH']['ROOT_DIR'])

# Load data
df = pd.read_parquet(config['PATH']['INT_DIR'] + '/training_set_preprocessed_nodrop.parquet', engine = 'fastparquet')
df_test = pd.read_parquet(config['PATH']['INT_DIR'] + '/test_set_preprocessed_nodrop.parquet', engine = 'fastparquet')

def objective(trial):

    # Define parameters for KNNRegressor
    params_knn = {
        'n_neighbors': trial.suggest_int('n_neighbors_knn', 1, 20),
        'weights': trial.suggest_categorical('weights_knn', ['uniform', 'distance']),
        'p': trial.suggest_int('p_knn', 1, 2),
    }
    params_other = {
        'val_size': trial.suggest_float('val_size', 0.1, 0.8)
    }

    X_train, X_val, X_test, y_train, y_val, y_test, test_ideal = train_val_test_split(df, 'target', test_size=.15, val_size=params_other['val_size'], random_state=7)

    _, desire_df_click = construct_desire(X_val)
    _, desire_df_book = construct_desire(X_val, target = 'booking_bool')

    prop_counts = X_val['prop_id'].value_counts()
    prop_counts.name = 'prop_counts'
    srch_dest_counts = X_val['srch_destination_id'].value_counts()
    srch_dest_counts.name = 'srch_dest_counts'
    merge_df_list = [(desire_df_click, 'prop_id'), (desire_df_book, 'prop_id'), (prop_counts, 'prop_id'), (srch_dest_counts, 'srch_destination_id')]   

    X_train = merge_and_drop(X_train, merge_df_list)
    X_test = merge_and_drop(X_test, merge_df_list)

    X_train= X_train.drop(['srch_id'], axis=1)
    X_test = X_test.drop(['srch_id'], axis=1)
    X_train.fillna(0, inplace=True)
    X_test.fillna(0, inplace=True)

    # Fill infty
    X_train = X_train.replace([np.inf, -np.inf], 0)
    X_test = X_test.replace([np.inf, -np.inf], 0)

    # Initialize wandb
    wandb.init(project='DMT-2023', group = 'knn_dmt', config = params_knn, reinit = True, allow_val_change=True)

    # Train KNNRegressor
    knn = KNeighborsRegressor(**params_knn)
    knn.fit(X_train, y_train)

    # Predict and calculate NDCG score for KNNRegressor
    y_pred_knn = knn.predict(X_test)
    df_res_knn = X_test.copy()
    df_res_knn['pred_grades'] = y_pred_knn
    df_res_knn = df_res_knn.sort_values(by=['srch_id', 'pred_grades'], ascending=[True, False], inplace=False)
    df_res_knn = df_res_knn.merge(test_ideal, on=['srch_id', 'prop_id'], how='left')
    ndcg_score_knn = calc_NDCG(test_ideal, df_res_knn)

    print(f'NDCG score for KNeighborsRegressor: {ndcg_score_knn} with params: {params_knn}')
    wandb.log({'ndcg_final': ndcg_score_knn})

    wandb.finish()

    # Return the NDCG score
    return ndcg_score_knn

# Create a study object and optimize the objective function.
study = optuna.create_study(study_name='knn_dmt', direction='maximize')
study.optimize(objective, n_trials=40)

# Extract the best hyperparameters
best_params = study.best_params
print(f'Best hyperparameters: {best_params}')

# Save best params to txt file
with open(config['PATH']['INT_DIR'] + '/knn_best_params.txt', 'w') as f:
    f.write(str(best_params))

# save study
with open(config['PATH']['INT_DIR'] + '/knn_study.pkl', 'wb') as f:
    pickle.dump(study, f)
