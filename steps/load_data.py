import logging
import pandas as pd
from zenml import step

class LoadData:
    """
    Load data from data_path.
    """
    def __init__(self, data_path: str):
        """
        Args:
        data_path: path to the data
        """
        self.data_path = data_path

    def get_data(self):
        """
        Load data from data_path.
        """
        logging.info(f"Load data from: {self.data_path}")
        return pd.read_csv(self.data_path)
    
@step
def load_df(data_path:str) -> pd.DataFrame:
    """
    Load data from data_path.

    Args:
        data_path: path to the data
    Returns:
        pd.DataFrame: the load data
    """
    try:
        load_data = LoadData(data_path)
        df = load_data.get_data()
        return df
    except Exception as e:
        logging.error(f"Error while load data: {e}")
        raise e