import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import yaml
import logging


logger = logging.getLogger('data_ingestion')
logger.setLevel("DEBUG")


console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler("error.log")
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_params(params_path: str) -> float:
    try:
        global test_size
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        test_size = params['data_ingestion']['test_size']
        logger.debug('test size retrieved')
        return test_size
    except FileNotFoundError:
        logger.error('file not found')
        raise
    except KeyError as e:
        print(f"Error: Missing key in the params file: {e}")
        raise
    except yaml.YAMLError as exc:
        print(f"Error reading YAML file: {exc}")
        raise


def read_data(url: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(url)
        return df
    except pd.errors.EmptyDataError:
        print("Error: The provided URL points to an empty dataset.")
        raise
    except pd.errors.ParserError:
        print("Error: Error parsing the CSV file.")
        raise
    except Exception as e:
        print(f"An unexpected error occurred while reading the data: {e}")
        raise


def process_data(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df.drop(columns=['tweet_id'], inplace=True)
        final_df = df[df['sentiment'].isin(["happiness", "sadness"])]
        final_df['sentiment'].replace({'happiness': 1, 'sadness': 0}, inplace=True)
        return final_df
    except KeyError as e:
        print(f"Error: Missing expected column in the DataFrame: {e}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred while processing the data: {e}")
        raise


def save_data(data_path, train_data, test_data):
    try:
        os.makedirs(data_path, exist_ok=True)
        train_data.to_csv(os.path.join(data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(data_path, "test.csv"), index=False)
    except OSError as e:
        print(f"Error: An OS error occurred while saving the data: {e}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred while saving the data: {e}")
        raise


def main():
    try:
        test_size = load_params('params.yaml')
        df = read_data("https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv")
        final_df = process_data(df)
        train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=42)
        data_path = os.path.join("data", "raw")
        save_data(data_path, train_data, test_data)
    except Exception as e:
        print(f"An error occurred in the main function: {e}")
        raise


if __name__ == '__main__':
    main()
