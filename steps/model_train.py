import logging
import pandas as pd
import mlflow

from zenml import step
from zenml.client import Client
from sklearn.base import RegressorMixin
from src.model_dev import LinearRegressionModel
from .config import ModelNameConfig

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def train_model(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    config: ModelNameConfig
) -> RegressorMixin:
    """
    Trains model om the load data

    Args:
        X_train: Training data
        y_train: Training labels
    """
    try:
        model = None
        if config.name_of_model == "LinearRegression":
            model = LinearRegressionModel()
            mlflow.sklearn.autolog()
            trained_model = model.train(X_train, y_train)
            logging.info(f"Model-{config.name_of_model} successfully trained")
            # mlflow.sklearn.log_model(model, "linear_regression_model")
            return trained_model
        else:
            raise ValueError(f"Model {config.name_of_model} not supported")
    except Exception as e:
        logging.error(f"Error in training model: {e}")
        raise e