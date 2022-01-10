import json
import numpy as np
import os
import random

FILE_PATH = os.path.dirname(__file__)


def _confusion_matrix(labels: np.array, predictions: np.array, num_labels: int) -> np.array:
    """
    Calculat Confusion Matrix for Single Multilabel Classification
    Parameters
    ----------
    labels : np.array
    predictions : np.array
    num_labels : int

    Returns
    -------
    np.array:
        Confusion Matrix
    """
    # https://www.researchgate.net/figure/Confusion-matrix-for-multi-class-classification-The-confusion-matrix-of-a_fig7_314116591
    confusion_matrix = np.zeros((num_labels, num_labels))

    for label, prediction in zip(labels, predictions):
        pred = int(prediction) if type(prediction) == np.int64 or type(prediction) == np.float64 else prediction

        if pred in label:
            confusion_matrix[pred][pred] += 1
        else:
            if len(label) != 1:
                label = random.sample(label, 1)
            confusion_matrix[label[0]][pred] += 1

    return confusion_matrix


def custom_precision(labels: np.array, predictions: np.array, exp_path: str, exp_name: str):
    """
    Calculate custom precision value.

    Parameters
    ----------
    exp_name : str
        experiment name
    exp_path : str
        path to experiment folder
    labels : np.array
    predictions : np.array

    Returns
    -------
    float:
        average precision
    """
    label_list = []
    for label in labels:
        for lab in label:
            label_list.append(lab)
    num_labels = len(set(label_list))

    # https://www.researchgate.net/figure/Confusion-matrix-for-multi-class-classification-The-confusion-matrix-of-a_fig7_314116591
    confusion_matrix = _confusion_matrix(labels, predictions, num_labels)

    precisions = []
    for label in range(num_labels):
        true_positive = confusion_matrix[label, label]
        false_positive = np.sum(confusion_matrix[:, label]) - confusion_matrix[label, label]
        if (true_positive + false_positive) != 0:
            precisions.append(true_positive / (true_positive + false_positive))
        else:
            precisions.append(0)
            print('Precision:', label)

    if exp_path:
        file_name = 'result_precison_single' + exp_name + '.json'
        with open(os.path.join(exp_path, file_name), "w") as fp:
            json.dump(precisions, fp)

    return np.mean(precisions)


def custom_recall(labels: np.array, predictions: np.array):
    """
    Calculate custom precision value.

    Parameters
    ----------
    labels : np.array
    predictions : np.array

    Returns
    -------
    float:
        average precision
    """
    label_list = []
    for label in labels:
        for lab in label:
            label_list.append(lab)
    num_labels = len(set(label_list))

    # https://www.researchgate.net/figure/Confusion-matrix-for-multi-class-classification-The-confusion-matrix-of-a_fig7_314116591
    confusion_matrix = _confusion_matrix(labels, predictions, num_labels)

    recalls = []
    for label in range(num_labels):
        true_positive = confusion_matrix[label, label]
        false_negative = np.sum(confusion_matrix[label, :]) - confusion_matrix[label, label]
        if (true_positive + false_negative) != 0:
            recalls.append(true_positive / (true_positive + false_negative))
        else:
            print('Recall:', label)
            continue

    return np.mean(recalls)


def custom_f1(labels: np.array, predictions: np.array) -> float:
    recall = custom_recall(labels, predictions)
    precision = custom_precision(labels, predictions, '', '')

    return 2*(recall * precision) / (recall + precision)
