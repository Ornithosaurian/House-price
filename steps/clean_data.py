import logging
from zenml import step

import pandas as pd
from typing_extensions import Annotated
from typing import Tuple

from src.data_cleaning import DataCleaning, DataPreProcessStrategyForTrain, DataDivideStrategy


@step
def clean_df(df: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"]
]:
    """
    Cleans data and divides it into train and split

    Args:
        df: Raw data 
    Returns:
        X_train: Training data
        X_test: Testing data
        y_train: Training labels
        y_test: Testing labels
    """
    try:
        process_strategy = DataPreProcessStrategyForTrain()
        data_cleaning = DataCleaning(df, process_strategy)
        processed_data = data_cleaning.handle_data()

        divide_strategy = DataDivideStrategy()
        data_cleaning = DataCleaning(processed_data, divide_strategy)
        X_train, X_test, y_train, y_test = data_cleaning.handle_data()
        logging.info("Data cleaning complete")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logging.error(f"Error in cleaning data step: {e}")
        raise e