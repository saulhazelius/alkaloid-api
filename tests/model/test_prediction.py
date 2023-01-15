import pytest
import json
import sys
sys.path.append('../../project')
from project.predict import prediction
from project.preprocess import create_input

spec_pos = 'spectra_pos.json'
spec_large = 'spectra_large.json'
MODEL_PATH = '../../models/model.xgb'


@pytest.fixture
def get_max_len():
    """Max length for array padding in create_input function"""
    with open('../../configs/predict_config.json') as w:
        max_len_def = json.load(w)

    max_len = max_len_def["max_len"]

    return max_len


def test_prediction_pos(get_max_len):
    inpt = create_input(spec_pos, get_max_len)
    pred = prediction(MODEL_PATH, inpt, get_max_len)

    assert pred == 1


def test_prediction_large(get_max_len):
    inpt = create_input(spec_large, get_max_len)
    pred = prediction(MODEL_PATH, inpt, get_max_len)

    assert pred == 0 
