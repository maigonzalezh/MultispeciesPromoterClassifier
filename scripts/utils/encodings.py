import numpy as np
import pandas as pd


def one_hot(text, char_set='acgt'):
    one_hot = np.zeros((len(text), len(char_set)), dtype=np.int8)
    for i, char in enumerate(text):
        one_hot[i, char_set.index(char)] = 1
    return one_hot
