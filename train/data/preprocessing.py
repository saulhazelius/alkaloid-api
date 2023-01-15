"""
Reads data from alkaloids and performs over- and under-sampling.
"""
import numpy as np
from imblearn.combine import SMOTEENN


def read_data(file):
    """Returns a np array from file. Each row from the original file
       is padded with -999.0 in order to have the same length as the
       row with the maximun number of NMR.
    """
    X = []
    y = []

    for line in file:
        spectra = [float(nmr) for nmr in line.split(", ")[:-1]]
        X.append(np.array(spectra))
        y.append(np.array(int(line.split(", ")[-1].strip())))

    X = np.array(X, dtype=object)
    y = np.array(y)

    return X, y


def resample(X, y):
    """Performs resampling with the SMOTEENN method using
       a min/Maj ratio of 0.1.
    """
    resampler = SMOTEENN(sampling_strategy=0.1, random_state=0)
    X_res, y_res = resampler.fit_resample(X, y)

    return X_res, y_res
