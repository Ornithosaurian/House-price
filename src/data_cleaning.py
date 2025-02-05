import logging
from abc import ABC, abstractmethod
from typing import Union
import numpy as np
import pandas as pd
import pickle
import os

from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split

KEEP_DATA =  [
                    "OverallQual",
                    "GrLivArea",
                    "GarageCars",
                    "GarageArea",
                    "TotalBsmtSF",
                    "1stFlrSF",
                    "FullBath",
                    "TotRmsAbvGrd",
                    "YearBuilt",
                    "YearRemodAdd",
                    "GarageFinish",
                    "KitchenQual",
                    "BsmtQual",
                    "ExterQual",
                ]

class DataStrategy(ABC):
    """
    Abstract class defining strategy for handling data
    """

    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass

class DataPreProcessStrategyForTrain(DataStrategy):
    """
    Strategy for preprocessing data for training.
    """
    
    def __init__(self, encoder_path="F:\machine learning\House Prices\models\ordinal_encoder.pkl", 
                 scaler_path="F:\machine learning\House Prices\models\standard_scaler.pkl"):
        self.encoder_path = encoder_path
        self.scaler_path = scaler_path
        self.encoder = self._load_or_train_encoder()
        self.scaler = self._load_or_train_scaler()

    def _load_or_train_encoder(self):
        """
        Load the OrdinalEncoder if it exists, otherwise train and save a new one.
        """
        if os.path.exists(self.encoder_path):
            logging.info("Loading existing OrdinalEncoder...")
            with open(self.encoder_path, "rb") as f:
                return pickle.load(f)
        else:
            logging.info("Training new OrdinalEncoder...")
            return OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        
    def _load_or_train_scaler(self):
        """
        Load the StandardScaler if it exists, otherwise train and save a new one.
        """
        if os.path.exists(self.scaler_path):
            logging.info("Loading existing StandardScaler...")
            with open(self.scaler_path, "rb") as f:
                return pickle.load(f)
        else:
            logging.info("Training new StandardScaler...")
            return StandardScaler()

    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess data for training.
        """
        logging.info("Preprocessing data for training...")
        try:
            keep_column = KEEP_DATA.copy()
            keep_column.append("SalePrice")

            transform_data = data[keep_column].copy()

            categorical_cols = transform_data.select_dtypes(include=["object"]).columns
            numerical_cols = transform_data.select_dtypes(include=["number"]).columns

            for col in categorical_cols:
                transform_data[col].fillna("NA")
            
            for col in numerical_cols:
                mean_value = transform_data[col].mean()  
                transform_data[col] = transform_data[col].fillna(mean_value)

            if not os.path.exists(self.encoder_path):
                transform_data[categorical_cols] = self.encoder.fit_transform(transform_data[categorical_cols])

                with open(self.encoder_path, "wb") as file:
                    pickle.dump(self.encoder, file)
                logging.info("OrdinalEncoder trained and saved.")
            else:
                transform_data[categorical_cols] = self.encoder.transform(transform_data[categorical_cols])
                logging.info("Used existing OrdinalEncoder.")
            
            q = transform_data['TotalBsmtSF'].quantile(0.99)
            transform_data = transform_data[transform_data['TotalBsmtSF'] < q]
            q = transform_data['SalePrice'].quantile(0.999)
            transform_data = transform_data[transform_data['SalePrice'] < q]

            data_cleaned = transform_data.reset_index(drop=True)

            target = data_cleaned['SalePrice']
            inputs = data_cleaned.drop(['SalePrice'], axis=1)       

            if not os.path.exists(self.scaler_path):
                inputs_scaled = self.scaler.fit_transform(inputs)

                with open(self.scaler_path, "wb") as file:
                    pickle.dump(self.scaler, file)
                logging.info("StandardScaler trained and saved.")
            else:
                inputs_scaled = self.scaler.transform(inputs)
            
            log_target = np.log(target)

            inputs_scaled_df = pd.DataFrame(inputs_scaled, columns=inputs.columns)

            prepared_data = pd.concat([inputs_scaled_df, log_target], axis=1)

            prepared_data = prepared_data.reset_index(drop=True)

            return prepared_data

        except Exception as e:
            logging.error(f"Error in preprocessing data for traning: {e}")
            raise e
        
class DataPreProcessStrategyForTesting(DataStrategy):
    """
    Strategy for preprocessing data for testing.
    """

    def __init__(self, encoder_path="F:\machine learning\House Prices\models\ordinal_encoder.pkl", 
                 scaler_path="F:\machine learning\House Prices\models\standard_scaler.pkl"):
        self.encoder_path = encoder_path
        self.scaler_path = scaler_path
        self.encoder = self._load_or_train_encoder()
        self.scaler = self._load_or_train_scaler()

    def _load_or_train_encoder(self):
        """
        Load the OrdinalEncoder if it exist.
        """
        if os.path.exists(self.encoder_path):
            logging.info("Loading existing OrdinalEncoder...")
            with open(self.encoder_path, "rb") as f:
                return pickle.load(f)
        else:
            logging.error("OrdinalEncoder does not exist.")
        
    def _load_or_train_scaler(self):
        """
        Load the StandardScaler if it exist.
        """
        if os.path.exists(self.scaler_path):
            logging.info("Loading existing StandardScaler...")
            with open(self.scaler_path, "rb") as f:
                return pickle.load(f)
        else:
            logging.error("StandardScaler does not exist.")

    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess data for testing.
        """
        logging.info("Preprocessing data for testing...")
        try:
            transform_data = data[KEEP_DATA].copy()
            categorical_cols = transform_data.select_dtypes(include=["object"]).columns
            numerical_cols = transform_data.select_dtypes(include=["number"]).columns
            
            for col in categorical_cols:
                transform_data[col].fillna("NA")
            
            for col in numerical_cols:
                mean_value = transform_data[col].mean()  
                transform_data[col] = transform_data[col].fillna(mean_value)

            if os.path.exists(self.encoder_path):
                transform_data[categorical_cols] = self.encoder.transform(transform_data[categorical_cols])
                logging.info("Used existing OrdinalEncoder.")
            else:
                logging.error("OrdinalEncoder does not exist.")

            q = transform_data['TotalBsmtSF'].quantile(0.99)
            transform_data = transform_data[transform_data['TotalBsmtSF'] < q]

            data_cleaned = transform_data.reset_index(drop=True)

            if not os.path.exists(self.scaler_path):
                inputs_scaled = self.scaler.fit_transform(data_cleaned)

                with open(self.scaler_path, "wb") as file:
                    pickle.dump(self.scaler, file)
                logging.info("StandardScaler trained and saved.")
            else:
                inputs_scaled = self.scaler.transform(data_cleaned)

            prepared_data = pd.DataFrame(inputs_scaled, columns=KEEP_DATA)

            return prepared_data
            
        except Exception as e:
            logging.error(f"Error in preprocessing data for testing: {e}")
            raise e
        
class DataDivideStrategy(DataStrategy):
    """
    Strategy for dividing data into train and test
    """
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        """
        Divide data into train and test
        """
        logging.info("Divide data")
        try:
            y = data["SalePrice"]
            X = data.drop("SalePrice", axis=1)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error(f"Error in dividing data: {e}")
            raise e
        
class DataCleaning:
    """
    Class for cleaning data which use different strategies
    """
    def __init__(self, data: pd.DataFrame, strategy: DataStrategy):
        self.data = data
        self.strategy = strategy
    
    def handle_data(self) -> Union[pd.DataFrame, pd.Series]:
        """
        Handle data
        """
        try:
            return self.strategy.handle_data(self.data)
        except Exception as e:
            logging.error(f"Error in handling data: {e}")
            raise e
        
if __name__ == "__main__":
    data = pd.read_csv("F:\machine learning\House Prices\data\\train.csv")
    print(data.head())
    data_cleaning = DataCleaning(data, DataPreProcessStrategyForTrain())
    data = data_cleaning.handle_data()
    print(data.columns)

    divide_strategy = DataDivideStrategy()
    data_cleaning = DataCleaning(data, divide_strategy)
    X_train, X_test, y_train, y_test = data_cleaning.handle_data()
    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)


    data = pd.read_csv("F:\machine learning\House Prices\data\\test.csv")
    print(data['GarageCars'].isnull().sum())
    data_cleaning = DataCleaning(data, DataPreProcessStrategyForTesting())
    data = data_cleaning.handle_data()
    print(data.isnull().sum())