import pandas as pd
import numpy as np
from scripts.utils.encodings import one_hot

def parse_data(data: pd.DataFrame, is2d: bool = True):
    sequences = data['Sequence'].values
    labels = data['label'].values
    one_hot_sequences = np.array([one_hot(seq) for seq in sequences])

    # for models that require 2D input
    if is2d:
        one_hot_sequences = one_hot_sequences.copy().reshape(len(one_hot_sequences), -1)

    return one_hot_sequences, labels
