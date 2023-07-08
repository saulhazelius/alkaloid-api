"""
Performs XGB training and logs model with MLFlow. 
"""
import json
import logging
from data.preprocessing import read_data, resample
from xgboost import XGBClassifier
import mlflow
from mlflow.xgboost import autolog 

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def open_file(path):
    """Returns the open data file from path.
    """
    f = open(path)

    return f


def load_model_definition(model_path):
    """Reads XGB hyperparameters info."""
    with open(model_path) as w:
        model_def = json.load(w)

    return model_def


def train(model_definition, X, y):
    """Trains and logs the XGB model using the 
       hyperparameters from the model definition
       and the processed data."""

    model_name = model_definition["model"]
    params = model_definition["params"]
    n_samples = len(X)

    if model_name == "XGBClassifier":
        
        logger.info(f"Loaded model: {model_name}")
        logger.info(f"Number of samples for training: {n_samples}")

        autolog()
        with mlflow.start_run():
            
            model = XGBClassifier(**params)
            model.fit(X, y)

    else:
        logger.warning("Please verify model name and definition")


if __name__ == '__main__':
    MODEL_CONFIG_PATH = "../configs/model_config.json"
    FILE_PATH = "./data/alkaloids.csv"

    data_file = open_file(FILE_PATH)
    model = load_model_definition(MODEL_CONFIG_PATH)
    X, y = read_data(data_file)
    X_res, y_res = resample(X, y)
    train(model, X_res, y_res)
