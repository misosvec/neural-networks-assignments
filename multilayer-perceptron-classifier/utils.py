import numpy as np

def onehot_encode(x):
    unique = np.unique(x)
    label_to_int = {label: idx for idx, label in enumerate(unique)}
    onehot = np.zeros((len(x), len(unique)))
    for i, label in enumerate(x):
        label_int = label_to_int[label]
        onehot[i, label_int] = 1
    return onehot