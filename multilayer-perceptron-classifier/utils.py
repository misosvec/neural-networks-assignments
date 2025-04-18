import numpy as np

def onehot_encode(x):
    unique = np.unique(x)
    label_to_int = {label: idx for idx, label in enumerate(unique)}
    onehot = np.zeros((len(x), len(unique)))
    for i, label in enumerate(x):
        label_int = label_to_int[label]
        onehot[i, label_int] = 1
    return onehot

def add_bias(X):
    if X.ndim == 1:
        return np.concatenate((X, [1]))
    else:
        pad = np.ones((1, X.shape[1]))
        return np.concatenate((X, pad), axis=0)

def compute_accuracy(y_true, y_pred):
    true_labels = np.argmax(y_true, axis=0)
    pred_labels = np.argmax(y_pred, axis=0)
    accuracy = np.mean(true_labels == pred_labels)
    return accuracy