import logging

import pandas as pd
from src.data_cleaning import DataCleaning, DataPreProcessStrategyForTesting

def get_data_for_test():
    try:
        data_path = r"F:/machine learning/House Prices/data/test.csv"
        # data_path="/mnt/f/machine learning/House Prices/data/test.csv"
        # data_path = "/app/data/test.csv"
        df = pd.read_csv(data_path)
        preprocess_strategy = DataPreProcessStrategyForTesting()
        data_cleaning = DataCleaning(df, preprocess_strategy)
        df = data_cleaning.handle_data()
        result = df.to_json(orient="split")
        return result
    except Exception as e:
        logging.error(e)
        raise e
    
if __name__ == "__main__":
    res = get_data_for_test()
    print(res)