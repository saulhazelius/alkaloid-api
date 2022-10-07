import json
import numpy as np
import logging
from xgboost import XGBClassifier

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

PREDICT_CONFIG_PATH = "/configs/predict_config.json"
MODEL_PATH = "/opt/model.xgb"


def get_max_len(max_len_path):
    """Max length for array padding in create_input function"""
    with open(max_len_path) as w:
        max_len_def = json.load(w)

    max_len = max_len_def["max_len"]

    return max_len


def create_input(event, max_len):
    """Returns a np array for inference. Takes the values from the JSON file,
    then checks for valid input values and convert them to a np.array from the
    loaded dictionary. The resultant array is padded to the max_len of spectra
    for alkaloids"""
    json_dict = event

    values_type = set([type(val) for val in json_dict.values()])
    for value_type in values_type:

        if value_type != int and value_type != float:
            logging.warning(
                f"Please set JSON input values to float or int types. Type found: {value_type}"
            )

    inpt = np.array(sorted([val for val in json_dict.values()]))
    if max_len - len(inpt) >= 0:

        inp_padded = np.pad(
            inpt, (0, max_len - len(inpt)), constant_values="-999.0"
        )

        return inp_padded.reshape(1, -1)

    else:

        return inpt.reshape(1, -1)


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
        predicted_value = xgb_model.predict(inpt)

    return predicted_value


def lambda_handler(event, context):
    max_len = get_max_len(PREDICT_CONFIG_PATH)   
    created_input = create_input(event['body'], max_len)

    model_path = MODEL_PATH
    predicted_value = int(prediction(model_path, created_input, max_len)[0])
    alkaloid = True if predicted_value == 1 else False
    return {
        'statusCode': 200,
        'body': json.dumps(
            {
                "predicted_label": predicted_value,
                "alkaloid_presence": alkaloid
            }
        )
    }

