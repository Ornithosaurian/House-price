import logging
from typing import Tuple
import mlflow

import pandas as pd
from zenml.client import Client
from zenml import step
from sklearn.base import RegressorMixin
from typing_extensions import Annotated


from src.evaluation import MSE, RMSE, R2

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def evaluate_model(model: RegressorMixin, 
                   X_test: pd.DataFrame,
                   y_test: pd.DataFrame
                   ) -> Tuple[
                       Annotated[float, "r2"],
                       Annotated[float, "rmse"]
                   ]:
    """
    Evaluates model on load data
    Args:
        model: model for evaluation
        X_test: Testing data
        y_test: Testing labels
    """
    try:
        prediction = model.predict(X_test)

        mse_class = MSE()
        mse = mse_class.calculate_score(y_test, prediction)
        mlflow.log_metric("mse", mse)

        rmse_class = RMSE()
        rmse = rmse_class.calculate_score(y_test, prediction)
        mlflow.log_metric("rmse", rmse)

        r2_class = R2()
        r2 = r2_class.calculate_score(y_test, prediction)
        mlflow.log_metric("r2", r2)

        logging.info("Evaluating model complete")
        return r2, rmse
    except Exception as e:
        logging.error(f"Error in evaluating step: {e}")
        raise e