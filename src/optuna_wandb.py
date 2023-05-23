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
df = pd.read_parquet(config['PATH']['INT_DIR'] + '/training_set_preprocessed_nodrop.parquet', engine = 'auto')
df_test = pd.read_parquet(config['PATH']['INT_DIR'] + '/test_set_preprocessed_nodrop.parquet', engine = 'auto')

categorical_features = ['hour', 'day', 'month', 'day_of_week', 'site_id', 'visitor_location_country_id', 'prop_country_id', 'prop_id', 'srch_destination_id']

for c in categorical_features:
    df[c] = df[c].astype('category')
    df_test[c] = df_test[c].astype('category')

import optuna
import lightgbm as lgb

def objective(trial):

    params_static = {
        "objective": "lambdarank",
        "metric":"ndcg",
    }
    params_lgbm = {
        'n_estimators': trial.suggest_int('n_estimators', 250, 900), 
        'num_leaves': trial.suggest_int('num_leaves', 10, 100),
        'max_depth': trial.suggest_int('max_depth', 1, 20), 
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2), 
        'subsample': trial.suggest_float('subsample', 0.4, 1), 
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1), 
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 0.2), 
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 0.2),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'min_child_weight': trial.suggest_float('min_child_weight', 0.0001, 0.1),
        }
    params_other = {
        'val_size': trial.suggest_float('val_size', 0.3, 0.8)
    }

    X_train, X_val, X_test, y_train, y_val, y_test, test_ideal = train_val_test_split(df, 'target', test_size=.15, val_size=params_other['val_size'], random_state=7)

    _, desire_df_click = construct_desire(X_val)
    _, desire_df_book = construct_desire(X_val, target = 'booking_bool')

    prop_counts = X_val['prop_id'].value_counts()
    prop_counts.name = 'prop_counts'
    prop_counts = pd.DataFrame({'prop_id':prop_counts.index, 'count':prop_counts.values})

    srch_dest_counts = X_val['srch_destination_id'].value_counts()
    srch_dest_counts.name = 'srch_dest_counts'
    srch_dest_counts = pd.DataFrame({'srch_destination_id':srch_dest_counts.index, 'count':srch_dest_counts.values})

    merge_df_list = [(desire_df_click, 'prop_id'), (desire_df_book, 'prop_id'), (prop_counts, 'prop_id'), (srch_dest_counts, 'srch_destination_id')]   

    X_train = merge_and_drop(X_train, merge_df_list)
    X_test = merge_and_drop(X_test, merge_df_list)

    group_train = X_train.groupby('srch_id').size().values
    group_val = X_test.groupby('srch_id').size().values

    X_train_lgb = X_train.drop(['srch_id'], axis=1)
    X_test_lgb = X_test.drop(['srch_id'], axis=1)

    params_all = {**params_lgbm, **params_other}
    wandb.init(project='DMT-2023', group = 'optuna_categorical', config = params_all, reinit = True, allow_val_change=True)
    cb = wandb_callback()
    ranker = lgb.LGBMRanker(**{**params_static, **params_lgbm})

    ranker.fit(
        X=X_train_lgb,
        y=y_train,
        group=group_train,
        eval_set=[(X_train_lgb, y_train),(X_test_lgb, y_test)],
        eval_group=[group_train, group_val],
        eval_at=[5],
        callbacks=[cb],
        feature_name='auto', 
        categorical_feature = 'auto'
    )
    for c in categorical_features:
        X_test_lgb[c] = X_test_lgb[c].astype('category')

    y_pred = ranker.predict(X_test_lgb)
    df_res = X_test.copy()
    df_res['pred_grades'] = y_pred
    df_res = df_res.sort_values(by=['srch_id', 'pred_grades'], ascending=[True, False], inplace=False)
    df_res = df_res.merge(test_ideal, on=['srch_id', 'prop_id'], how='left')

    ndcg_score = calc_NDCG(test_ideal, df_res)
    
    wandb.log({'ndcg_final': ndcg_score})
    
    wandb.finish()

    return ndcg_score



# Create a study object and optimize the objective function.
study = optuna.create_study(study_name='dmt_22_5', direction='maximize')
study.optimize(objective, n_trials=120)


wandb.finish()
# Extract the best hyperparameters
best_params = study.best_params
print(f'Best hyperparameters: {best_params}')

# Save best params to txt file
with open(config['PATH']['INT_DIR'] + '/optuna_best_params_22_5.txt', 'w') as f:
    f.write(str(best_params))

# save study
import pickle
with open(config['PATH']['INT_DIR'] + '/optuna_best_params_22_5.pkl', 'wb') as f:
    pickle.dump(study, f)
