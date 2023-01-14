"""
Performs XGB training and logs model with MLFlow. 
"""
import json
import logging
from data.preprocessing import processed_data
from xgboost import XGBClassifier
import mlflow
from mlflow.xgboost import autolog 

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

MODEL_CONFIG_PATH = "../configs/model_config.json"


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
    
    model = load_model_definition(MODEL_CONFIG_PATH)
    X, y = processed_data
    train(model, X, y)
