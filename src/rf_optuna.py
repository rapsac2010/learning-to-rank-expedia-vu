from sklearn.ensemble import RandomForestRegressor
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

# # Load data
df = pd.read_parquet(config['PATH']['INT_DIR'] + '/training_set_preprocessed_nodrop.parquet', engine = 'fastparquet')
df_test = pd.read_parquet(config['PATH']['INT_DIR'] + '/test_set_preprocessed_nodrop.parquet', engine = 'fastparquet')


def objective(trial):

    # Define parameters for RandomForestRegressor
    params_rf = {
        'n_estimators': trial.suggest_int('n_estimators_rf', 100, 900),
        'max_depth': trial.suggest_int('max_depth_rf', 1, 20),
        'min_samples_split': trial.suggest_int('min_samples_split_rf', 2, 10),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf_rf', 1, 10),
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
    X_train.fillna(0, inplace=True)
    X_test.fillna(0, inplace=True)

    # Fill infty
    X_train = X_train.replace([np.inf, -np.inf], 0)
    X_test = X_test.replace([np.inf, -np.inf], 0)

    X_train= X_train.drop(['srch_id'], axis=1)
    X_test = X_test.drop(['srch_id'], axis=1)


    # Initialize wandb
    wandb.init(project='DMT-2023', group = 'rf_optuna', config = params_rf, reinit = True, allow_val_change=True)

    # Train RandomForestRegressor
    rf = RandomForestRegressor(**params_rf)
    rf.fit(X_train, y_train)

    # Predict and calculate NDCG score for RandomForestRegressor
    y_pred_rf = rf.predict(X_test)
    df_res_rf = X_test.copy()
    df_res_rf['pred_grades'] = y_pred_rf
    df_res_rf = df_res_rf.sort_values(by=['srch_id', 'pred_grades'], ascending=[True, False], inplace=False)
    df_res_rf = df_res_rf.merge(test_ideal, on=['srch_id', 'prop_id'], how='left')
    ndcg_score_rf = calc_NDCG(test_ideal, df_res_rf)

    print(f'NDCG score for RandomForestRegressor: {ndcg_score_rf} with params: {params_rf}')
    wandb.log({'ndcg_final_rf': ndcg_score_rf})

    wandb.finish()

    # Return the NDCG score
    return ndcg_score_rf

# Create a study object and optimize the objective function.
study = optuna.create_study(study_name='rf_dmt', direction='maximize')
study.optimize(objective, n_trials=50)

# Extract the best hyperparameters
best_params = study.best_params
print(f'Best hyperparameters: {best_params}')

# Save best params to txt file
with open(config['PATH']['INT_DIR'] + '/rf_best_params.txt', 'w') as f:
    f.write(str(best_params))

# save study
with open(config['PATH']['INT_DIR'] + '/rf_best_params.pkl', 'wb') as f:
    pickle.dump(study, f)
