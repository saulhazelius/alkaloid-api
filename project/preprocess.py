"""
Preprocessing functions for prediction.
"""

import json
import numpy as np
import logging


def get_max_len(max_len_path):
    """Max length for array padding in create_input function"""
    with open(max_len_path) as w:
        max_len_def = json.load(w)

    max_len = max_len_def["max_len"]

    return max_len


def create_input(json_file, max_len):
    """Returns a np array for inference. Takes the values from the JSON file,
    then checks for valid input values and convert them to a np.array from the
    loaded dictionary. The resultant array is padded to the max_len of spectra
    for alkaloids.
    """
    try:
        with open(json_file) as nmr:
            json_dict = json.load(nmr)

    except ValueError as e:
        logging.warning(f"Invalid JSON: {e}")

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


PREDICT_CONFIG_PATH = "../configs/predict_config.json"
max_len = get_max_len(PREDICT_CONFIG_PATH)
json_file = "spectra.json"
created_input = create_input(json_file, max_len)
