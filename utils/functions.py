import pickle
import numpy as np


def mean_squared_error(x_true, x_pred):
    """
        Compute MSE
    """
    return np.mean((np.array(x_true) - np.array(x_pred))**2)


def save_object(obj, filename_object):
    """
        Save pickle object 
    """
    with open(filename_object, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
    print('Object saved:', filename_object)


def load_object(filename_object):
    """
        Load pickle object 
    """
    with open(filename_object, 'rb') as input:
        obj = pickle.load(input)
    print("Object loaded:", filename_object)
    return obj


def sort_lists(X, Y):
    """
    returns two sorted lists
    """

    T = sorted(zip(X, Y), key=lambda x: x[0])
    X = np.array([x for x, _ in T])
    Y = np.array([y for _, y in T])
    return X, Y
