import pandas as pd
import configparser
import os

config = configparser.ConfigParser()
config.read('src/config.ini')

# Check if file exists
if not os.path.isfile(config['PATH']['DATA_DIR'] + '/training_set.parquet'):
    print("Training set not found, converting csv to parquet")
    # Read csv file
    df = pd.read_csv(config['PATH']['DATA_DIR'] + '/training_set.csv')

    # Convert to parquet
    df.to_parquet(config['PATH']['DATA_DIR'] + '/training_set.parquet')
else:
    print("Training set found, skipping conversion")

if not os.path.isfile(config['PATH']['DATA_DIR'] + '/test_set.parquet'):
    print("Test set not found, converting csv to parquet")
    # Read csv file
    df = pd.read_csv(config['PATH']['DATA_DIR'] + '/test_set.csv')

    # Convert to parquet
    df.to_parquet(config['PATH']['DATA_DIR'] + '/test_set.parquet')
else:
    print("Test set found, skipping conversion")
    