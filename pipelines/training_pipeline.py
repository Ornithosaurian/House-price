from zenml import pipeline
from steps.load_data import load_df
from steps.clean_data import clean_df
from steps.model_train import train_model
from steps.evaluation import evaluate_model
from steps.config import ModelNameConfig, get_docker_settings

docker_settings = get_docker_settings()

@pipeline(settings=docker_settings)
def train_pipeline(data_path: str):
    df = load_df(data_path)
    X_train, X_test, y_train, y_test = clean_df(df)
    config = ModelNameConfig()
    model = train_model(X_train, y_train, config)
    r2_score, rmse = evaluate_model(model, X_test, y_test)