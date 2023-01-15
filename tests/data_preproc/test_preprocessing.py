from train.data.preprocessing import read_data, resample


def test_read_data():
    file = open('../../train/data/alkaloids.csv')
    X, y = read_data(file)

    assert X.shape[1] == 37

    
def test_resample():
    file = open('../../train/data/alkaloids.csv')
    X, y = read_data(file)
    _, y_res = resample(X, y)
    y_min = y_res[y_res == 1]
    y_maj = y_res[y_res == 0]

    assert round(len(y_min) / len(y_maj), 1) == 0.1
