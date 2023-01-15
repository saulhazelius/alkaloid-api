import pytest
import json
from train.data.preprocessing import read_data, resample
from project.preprocess import create_input

spec_wrong = 'spectra_wrong.json'
spec_inv = 'spectra_invalid.json'
spec = 'spectra.json'


@pytest.fixture
def get_max_len():
    """Max length for array padding in create_input function"""
    with open('../../configs/predict_config.json') as w:
        max_len_def = json.load(w)

    max_len = max_len_def["max_len"]

    return max_len


def test_read_data(get_max_len):
    file = open('../../train/data/alkaloids.csv')
    X, y = read_data(file)

    assert X.shape[1] == get_max_len

    
def test_resample():
    file = open('../../train/data/alkaloids.csv')
    X, y = read_data(file)
    _, y_res = resample(X, y)
    y_min = y_res[y_res == 1]
    y_maj = y_res[y_res == 0]

    assert round(len(y_min) / len(y_maj), 1) == 0.1

   
def test_create_input_raises(get_max_len):
    with pytest.raises(TypeError):
        create_input(spec_wrong, get_max_len)


def test_create_input_fails(get_max_len):
    with pytest.raises(UnboundLocalError):
        create_input(spec_inv, get_max_len)


def test_create_input(get_max_len):
    inpt = create_input(spec, get_max_len)
    
    assert inpt.shape == (1, 37)
