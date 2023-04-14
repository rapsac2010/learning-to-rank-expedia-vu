from joblib import dump, load
from tqdm import tqdm
from helpers.helper_classes import *
import pandas as pd
import numpy as np
import numpy as np
import pmdarima as pm

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

        # Linear interpolate df nan values
        df_person['imputed'] = df_person['value'].interpolate(method='linear', limit_direction='both')
        df.iloc[idx_person, df.columns.get_loc('value')] = df_person['imputed']
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