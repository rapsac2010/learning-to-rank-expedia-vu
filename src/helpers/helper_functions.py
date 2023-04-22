from joblib import dump, load
from tqdm import tqdm
from helpers.helper_classes import *
import pandas as pd
import numpy as np
import numpy as np
import pmdarima as pm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import optuna
from optuna import Trial
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization



# Placeholder function
def test():
    print("Hello World")



def KF_params(arma_model):
    p, _, q = arma_model.order
    params = arma_model.params()
    
    # if model.params() has no intercept key then set the intercept to 0
    if 'intercept' not in arma_model.params().keys():
        # Set to numpy 0
        intercept = np.array([0])
    else:
        intercept = params['intercept']
    # if p == 0 and q == 0:
    #     return None
    m = np.maximum(p, q+1)
    T = np.zeros((m, m))
    R = np.zeros(m)
    a_init = np.zeros((m, 1))
    P_init = np.eye(m) * 10e7
    R[0] = 1
    Z = np.zeros(m)
    Z[0] = 1
    d = intercept.astype(float)
    Q = np.array([params['sigma2']]).reshape(1,1)

    for i in range(p):
        T[i, 0] = params["ar.L"+str(i+1)]
    for j in range(q):
        R[j+1] = params["ma.L"+str(j+1)]
    if m > 1:
        for k in range(m-1):
            T[k, k+1] = 1
    elif p == 0 and q == 0:
        T[0,0] = 1
    
    R = R.reshape(m,1)
    Z = Z.reshape(1,m)

    # Sum AR coefficients
    sum_AR = np.sum(arma_model.arparams())
    sum_AR2 = np.sum(arma_model.arparams()**2)
    sum_MA2 = np.sum(arma_model.maparams()**2)

    # Initial state and variance
    if p > 0:
        mu = intercept / (1 - sum_AR)
        d = mu
        a_init[0] = mu
        if p > 1:
            for i in range(p-1):
                a_init[i+1] = np.sum(arma_model.arparams()[i+1] * mu)
        P_init[0,0] = params['sigma2'] * (sum_MA2 + 1) / (1 - sum_AR2)

    return {'T': T, 'R': R, 'Z': Z, 'Q': Q, 'd': d, 'a_init': a_init, 'P_init': P_init}


def get_ARMA(df, variable, verbose = 0):
    print("Obtaining ARMA parameters for variable: ", variable)
    KF_dict = {}
    for person in tqdm(df['id'].unique()):

        # Dataframe for variable of current person
        idx_mood = np.logical_and(df['id'] == person , df['variable'] == variable)
        df_mood = df[idx_mood].copy()['value']
        model = pm.auto_arima(df_mood, suppress_warnings=True, seasonal=False, stepwise=True, d = 0, stationary = True, with_intercept = True)

        # Extract the best (p, d, q) orders
        if verbose:
            p, d, q = model.order
            print(f"person: {person}, order: {model.order}")
            print(model.summary())
            print(model.arparams())
            params = model.params()

        # Get KF parameters
        KF_dict[person] = KF_params(model)
    return KF_dict

def create_NA(df_in, variable, verbose = 0):
    print("Creating missing values for variable: ", variable)
    
    df = df_in.copy()
    for person in tqdm(df['id'].unique()):
        idx_cur = np.logical_and(df['id'] == person , df['variable'] == variable)
        df_cur = df[idx_cur]
        df_cur

        df_time = df[np.logical_and(df['id'] == person , df['variable'] == 'mood')]
        # Get first date
        first_date = df_time['time'].min().date()

        # Last date
        last_date = df_time['time'].max().date()

        # Iterate over dates by day
        for date in pd.date_range(start=first_date, end=last_date, freq='D'):
            # Get all rows for this date
            idx_date = df_cur['time'].dt.date == date.date()
            df_date = df_cur[idx_date].copy()

            if len(df_date) == 5:
                continue

            # check for observation between 9.00 and 12.00
            hour_sets = [[9,12], [12,15], [15,18], [18,21], [21,24]]

            for hour_set in hour_sets:
                idx_cur = np.logical_and(df_date['time'].dt.hour >= hour_set[0], df_date['time'].dt.hour < hour_set[1])
                if len(df_date[idx_cur]) == 0:
                    # Set hour of date
                    cur_date = date.replace(hour=hour_set[0])
                    # Create new row
                    new_row = pd.DataFrame({'id': person, 'time': cur_date, 'variable': variable, 'value': np.nan}, index=[0])
                    df = pd.concat([df, pd.DataFrame(new_row)], ignore_index=True)
    return df

def impute_KF(df_in, variable, KF_res_dict, verbose = 0):
    print("Imputing missing values for variable: ", variable)

    df = df_in.copy()
    comp_df_dict = {}
    for person in tqdm(df['id'].unique()):
        idx_person = np.logical_and(df['id'] == person, df['variable'] == variable)
        KF_cur = KF_res_dict[person]
        df_person = df[idx_person].copy()

        # sort by time
        df_person.sort_values('time', inplace=True)
        
        # df_mood.values to floats
        df_person['value'] = df_person['value'].astype(float)

        KF = KalmanFilter(y=df_person['value'].values,
                        a_init=KF_cur['a_init'],
                        P_init = KF_cur['P_init'],
                        H = np.array([0]).reshape(1,1),
                        Q = KF_cur['Q'],
                        R = KF_cur['R'],
                        d = KF_cur['d'])

        res = KF.run_smoother(Z = KF_cur['Z'], T = KF_cur['T'])

        smoothed_var = (res['a_smooth'][:, 0] + KF_cur['d']).flatten()
        smoothed_var[smoothed_var < 0] = 0
        smoothed_var[smoothed_var > 10] = 10

        # DF with one column with the smoothed values, and one column with the true values
        df_person['smoothed'] = smoothed_var
        df.iloc[idx_person, df.columns.get_loc('value')] = smoothed_var
        comp_df_dict[person] = df_person
    return df, comp_df_dict

def imputation_linear(df_in, variable, verbose = 0):
    df = df_in.copy()
    comp_df_dict = {}
    for person in tqdm(df['id'].unique()):
        idx_person = np.logical_and(df['id'] == person, df['variable'] == variable)
        df_person = df[idx_person].copy()

        # sort by time
        df_person.sort_values('time', inplace=True)
        
        # df_mood.values to floats
        df_person['value'] = df_person['value'].astype(float)

        time_df = df_person.copy()
        time_df.index = time_df['time']
        time_df['value'] = time_df['value'].interpolate(method='time', limit_direction='both')
        # Linear interpolate df nan values
        time_df.index = df_person.index
        df_person['smoothed'] = time_df['value']
        df.iloc[idx_person, df.columns.get_loc('value')] = df_person['smoothed']
        comp_df_dict[person] = df_person
        
    return df, comp_df_dict

def impute_ARMA(df, variable, verbose = 0):
    # Obtain KF_res_dict
    KF_res_dict = get_ARMA(df, variable, verbose = verbose)

    # Construct NA's in df
    df = create_NA(df, variable, verbose = verbose)

    # Impute missing values
    df, compare_df = impute_KF(df, variable, KF_res_dict, verbose = verbose)

    return df, compare_df

def impute_linear(df, variable, verbose = 0):
    df = create_NA(df, variable, verbose = verbose)
    df, compare_df = imputation_linear(df, variable, verbose = verbose)
    return df, compare_df

def drop_partial_obs(df_in, threshold = 6):
    df_cur = df_in.copy()
    len_start = len(df_cur)

    for person in tqdm(df_cur['id'].unique()):
        df_person = df_cur[df_cur['id'] == person]
        df_person = df_person.sort_values(by='time')

        # initialize while loop over days
        for direction in [1, -1]:
            i = 0 if direction == 1 else -1
            day = df_person['time'].dt.date.unique()[i]
            n_variables = 0

            # while loop over days
            while n_variables < threshold:
                df_day = df_person[df_person['time'].dt.date == day]
                n_variables = len(df_day['variable'].unique())

                if n_variables < threshold:
                    # Drop observations for this day
                    df_cur.drop(df_day.index, axis=0, inplace=True)

                i += direction
                day = df_person['time'].dt.date.unique()[i]

    len_end = len(df_cur)
    print(f"Removed {len_start - len_end} observations")
    return df_cur


def find_split_ts(df, split_ratios, date_colname='time', plot_data=False):
    split_points = np.cumsum(split_ratios) * 100
    dates = df[date_colname].unique()
    dates.sort()
    x = []
    y = []
    for date in dates:
        x.append(date)
        y.append(len(df[df[date_colname] <= date]) / len(df) * 100)



    split_dates = []
    for split in split_points:
        for i in range(len(x)):
            if y[i] >= split:
                split_dates.append(x[i])
                break

    if plot_data:
        plt.figure(figsize=(5, 3))
        for split, split_date in zip(split_points, split_dates):
            plt.axhline(split, linestyle='--')
            plt.axvline(split_date)
        plt.title(f'Cumulative distribution of data. Splits at {split_points}%')
        plt.plot(x, y)

    return split_dates

def train_test_ts(X, y, test_size = 0.2):
    # Find split date
    split_date = find_split_ts(X, test_size)
    
    # Split data
    X_train, X_test = X[X['time'] < split_date], X[X['time'] >= split_date]
    y_train, y_test = y[X['time'] < split_date], y[X['time'] >= split_date]
    
    # Drop time column
    X_train, X_test = X_train.drop('time', axis = 1), X_test.drop('time', axis = 1)
    
    return X_train, X_test, y_train, y_test

import numpy as np


def train_val_test_ts(X, y, split_ratios=(0.6, 0.2, 0.2), drop_time=True):
    # Find split dates
    train_split, val_split = find_split_ts(X, split_ratios)

    # Split data
    X_train, X_val, X_test = X[X['time'] < train_split], X[(X['time'] >= train_split) & (X['time'] < val_split)], X[X['time'] >= val_split]
    y_train, y_val, y_test = y[X['time'] < train_split], y[(X['time'] >= train_split) & (X['time'] < val_split)], y[X['time'] >= val_split]

    # Drop time column
    if drop_time:
        X_train, X_val, X_test = X_train.drop('time', axis=1), X_val.drop('time', axis=1), X_test.drop('time', axis=1)

    return X_train, X_val, X_test, y_train, y_val, y_test

def get_best_pipeline(best_trial, X_train):
    classifier_name = best_trial.params['classifier']

    preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), list(X_train.columns))
    ])

    if classifier_name == 'RandomForest':
        classifier_obj = RandomForestClassifier(n_estimators=best_trial.params['n_estimators'], max_depth=best_trial.params['max_depth'])
    elif classifier_name == 'GradientBoosting':
        classifier_obj = GradientBoostingClassifier(n_estimators=best_trial.params['n_estimators'], learning_rate=best_trial.params['learning_rate'], max_depth=best_trial.params['max_depth'])
    elif classifier_name == 'NaiveBayes':
        classifier_obj = GaussianNB()
    elif classifier_name == 'SVC':
        classifier_obj = SVC(C=best_trial.params['C'], kernel=best_trial.params['kernel'], decision_function_shape='ovr')
    else:  # KNN
        classifier_obj = KNeighborsClassifier(n_neighbors=best_trial.params['n_neighbors'])

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', classifier_obj)
    ])

    return pipeline

# Count number of days for each person and print
def count_days(df, id_name = 'id'):
    for person in df[id_name].unique():
        print(f"Person {person} has { len( df[df[id_name] == person] )} days")

def shape_lstm(X, y, seq_length, return_naive = False):
    import numpy as np

    X_lstm = []
    y_lstm = []
    naive_pred = []
    
    # Iterate over persons:
    for person in X['remainder__id'].unique():
        # Get all days for this person
        X_person = X[X['remainder__id'] == person].drop(columns = ['remainder__id']).values
        y_person = y[X['remainder__id'] == person].drop(columns = ['remainder__id']).values

        i = 0
        while i < len(X_person):
            # If days are less than seq_length, pad with zeros at the beginning of X
            if len(X_person) < seq_length:
                n_missing = seq_length - len(X_person)
                X_padded = np.pad(X_person, ((n_missing, 0), (0, 0)), mode='constant', constant_values=0)
                X_lstm.append(X_padded)
                y_lstm.append(y_person[-1])
                naive_pred.append(y_person[-2])
                break

            # If days are more than seq_length, create sequences of seq_length days
            elif i + seq_length <= len(X_person):
                X_lstm.append(X_person[i:i+seq_length])
                y_lstm.append(y_person[i+seq_length-1])
                naive_pred.append(y_person[i+seq_length-2])
                i += 1

            # If the remaining days are not enough to form a full sequence, stop iterating
            else:
                break
                
        # Add sequenced data for this person to the list of all sequences

    # Convert lists to numpy arrays and reshape y_lstm
    X_lstm = np.array(X_lstm).astype('float32'	)
    y_lstm = np.array(y_lstm).reshape(-1, 1).astype('float32')

    if return_naive:
        return X_lstm, y_lstm, naive_pred
    return X_lstm, y_lstm


# Define the LSTM model
def build_lstm_model_regression(input_shape, hunits = 64):
    model = Sequential()
    model.add(LSTM(hunits, input_shape=input_shape, return_sequences=False))
    model.add(Dense(1, activation='linear'))

    return model