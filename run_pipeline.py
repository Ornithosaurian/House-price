from pipelines.training_pipeline import train_pipeline
from zenml.client import Client

if __name__ == "__main__":
    #Run pipeline
    stack = Client().active_stack
    
    experiment_tracker = stack.experiment_tracker
    print(experiment_tracker.get_tracking_uri())
    # data_path = "/app/data/train.csv"
    data_path = r"F:/machine learning/House Prices/data/train.csv"
    # data_path="/mnt/f/machine learning/House Prices/data/train.csv"
    train_pipeline(data_path=data_path)