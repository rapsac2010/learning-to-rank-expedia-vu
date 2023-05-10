import pandas as pd
import configparser
import os


# Check if file exists
if not os.path.isfile('../data/training_set.parquet'):
    print("Training set not found, converting csv to parquet")
    # Read csv file
    df = pd.read_csv('../data/training_set_VU_DM.csv')

    # Convert to parquet
    df.to_parquet('../data/training_set.parquet')
else:
    print("Training set found, skipping conversion")

if not os.path.isfile('../data/test_set.parquet'):
    print("Test set not found, converting csv to parquet")
    # Read csv file
    df = pd.read_csv('../data/test_set_VU_DM.csv')

    # Convert to parquet
    df.to_parquet('../data/test_set.parquet')
else:
    print("Test set found, skipping conversion")
    