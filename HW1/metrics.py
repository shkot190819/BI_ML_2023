import numpy as np


def binary_classification_metrics(y_pred, y_true):
    """
    Computes metrics for binary classification
    Arguments:
    y_pred, np array (num_samples) - model predictions
    y_true, np array (num_samples) - true labels
    Returns:
    precision, recall, f1, accuracy - classification metrics
    """

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score

    TP, TN, FP, FN = 0, 0, 0, 0
    for i in range(len(y_true)):
        if y_pred[i] == 1:
            if y_true[i] == 1:
                TP += 1
            else: 
                FP += 1
        else:
            if y_true[i] == 0:
                TN += 1
            else:
                FN +=1
    accuracy = (TP+TN)/(TP+TN+FP+FN)
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    f1 = 2 * (precision*recall/(precision+recall))
    return accuracy, precision, recall, f1



def multiclass_accuracy(y_pred, y_true):
    """
    Computes metrics for multiclass classification
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true labels
    Returns:
    accuracy - ratio of accurate predictions to total samples
    """

    n_test = len(y_true)
    TP_TN = 0
    for i in range(n_test):
        if y_true[i] == y_pred[i]: 
            TP_TN +=1
    return TP_TN/n_test 


def r_squared(y_pred, y_true):
    r2 = 1-sum((y_true-y_pred)**2)/sum((y_true-np.mean(y_true))**2)
    return r2

def mse(y_pred, y_true):
    mse = np.sum((y_pred - y_true)**2)/len(y_pred)
    return mse


def mae(y_pred, y_true):
    mae = np.sum(np.absolute(y_pred - y_true))/len(y_pred)
    return mae
    