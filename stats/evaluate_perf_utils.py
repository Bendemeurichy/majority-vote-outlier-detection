import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score


def evaluate_performance(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    performance_metrics = {
        "True Negatives": tn,
        "False Positives": fp,
        "False Negatives": fn,
        "True Positives": tp,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
    }

    return performance_metrics
