import numpy.random as npr
import random
import torch
import test_functions

def test_function3(x):
    return test_functions.test_function(x + 1)

def test_function2(x):
    return x + 1