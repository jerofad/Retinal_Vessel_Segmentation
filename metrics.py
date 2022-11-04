import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score
from sklearn.metrics import auc, plot_precision_recall_curve, roc_auc_score




def iou(outputs: torch.Tensor, labels: torch.Tensor):
    SMOOTH = 1e-6
    outputs = (outputs.squeeze(1) > 0.5).int()
    labels = (labels > 0.5).int()
    intersection = torch.sum((outputs & labels).float())
    union = torch.sum((outputs | labels).float())

    iou = (intersection + SMOOTH) / (union + SMOOTH)
    return iou



def calculate_metrics(y_true, y_pred):
    """ Ground truth """
    y_true = y_true.cpu().numpy()
    y_true = y_true > 0.5  # threshold
    y_true = y_true.astype(np.uint8)
    y_true = y_true.reshape(-1)

    """ Prediction """
    y_pred = y_pred.cpu().numpy()
    y_pred = y_pred > 0.5
    y_pred = y_pred.astype(np.uint8)
    y_pred = y_pred.reshape(-1)

    score_jaccard = jaccard_score(y_true, y_pred)
    score_f1 = f1_score(y_true, y_pred)
    score_recall = recall_score(y_true, y_pred)
    score_precision = precision_score(y_true, y_pred)
    score_acc = accuracy_score(y_true, y_pred)
    score_roc_auc = roc_auc_score(y_true, y_pred)

    return [score_jaccard, score_f1, score_recall, score_precision, score_acc, score_roc_auc]


def mask_parse(mask):
    mask = np.expand_dims(mask, axis=-1)    ## (512, 512, 1)
    mask = np.concatenate([mask, mask, mask], axis=-1)  ## (512, 512, 3)
    return mask