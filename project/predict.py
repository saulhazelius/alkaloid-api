"""
Script for local alkaloid prediction.
"""
import logging
import numpy as np
from xgboost import XGBClassifier
from preprocess import create_input, get_max_len

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def prediction(model_path, inpt, max_len):
    """Predicts alkaloid or non-alkaloid existence for input arrays.
    If input size > MAX_LEN automatically predicts non-alkaloid."""
    if len(inpt[0]) > max_len:

        predicted_value = np.array([0])
        logger.info(
            "Number of 13C spectra greater than max spectra for alkaloids."
            "Model not loaded for prediction."
        )

    else:

        xgb_model = XGBClassifier()
        xgb_model.load_model(model_path)  # Load model
        logger.info(f"Model for prediction: {xgb_model}")
        predicted_value = xgb_model.predict(inpt)

    return predicted_value


if __name__ == "__main__":
    MODEL_PATH = "../models/model.xgb"
    PREDICT_CONFIG_PATH = "../configs/predict_config.json"
    max_len = get_max_len(PREDICT_CONFIG_PATH)
    json_file = "spectra.json"
    created_input = create_input(json_file, max_len)

    logger.info(f"Array of spectra: {created_input}")

    model_path = MODEL_PATH

    predicted_value = prediction(model_path, created_input, max_len)
    logger.info(
        f"Alkaloid prediction: {True if predicted_value == 1 else False}"
    )
