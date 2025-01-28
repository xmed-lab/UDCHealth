# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : metrics.py
# Time       ：8/3/2024 9:14 am
# Author     ：XXXX
# version    ：python 
# Description：
"""

from typing import Dict, List, Optional, Any
import os

import numpy as np
import sklearn.metrics as sklearn_metrics
from pyhealth.medcode import ATC
import pyhealth.metrics.calibration as calib
from pyhealth.metrics import ddi_rate_score
from pyhealth import BASE_CACHE_PATH as CACHE_PATH

from typing import Dict, List, Optional

import numpy as np
import sklearn.metrics as sklearn_metrics

import pyhealth.metrics.calibration as calib
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import sklearn.metrics as sklearn_metrics

import pyhealth.metrics.calibration as calib
import pyhealth.metrics.prediction_set as pset
from config import config
from collections import Counter


def group_rec(y, p, pred, patient_ids, group_p):
    # jaccard, f1-score, precision, recall, roc_auc, pr_auc
    # print(group_p)
    # print("==========")
    # print(patient_ids) # ['85424', '2613', '4931', '90917']
    # print(group_p)
    # print("A", patient_ids)
    patient_ids = np.array([group_p[id] for id in patient_ids]) # rare type
    zero_indices = np.where(patient_ids == '0%-20%')[0]
    one_indices = np.where(patient_ids == '20%-40%')[0]
    two_indices = np.where(patient_ids == '40%-60%')[0]
    three_indices = np.where(patient_ids == '60%-80%')[0]
    four_indices = np.where(patient_ids == '80%-100%')[0]
    group_indices = {'0%-20%':zero_indices, '20%-40%':one_indices, '40%-60%':two_indices, '60%-80%':three_indices, '80%-100%':four_indices}
    # print(group_indices)
    out = {}
    for key, lis in group_indices.items():
        y_g, p_g, pred_g = y[lis], p[lis], pred[lis]
        # print(y_g.shape, p_g.shape)
        roc_auc_samples = sklearn_metrics.roc_auc_score(
            y_g, p_g, average="samples"
        )
        pr_auc_samples = sklearn_metrics.average_precision_score(
                y_g, p_g, average="samples"
            )
        f1_samples = sklearn_metrics.f1_score(y_g, pred_g, average="samples",  zero_division=1)
        jaccard_samples = sklearn_metrics.jaccard_score(
            y_g, pred_g, average="samples", zero_division=1
        )
        out[key] = [jaccard_samples, f1_samples, pr_auc_samples, roc_auc_samples]
    return out






def topk_precision(y, p, k):
    """Computes precision at k for multilabel classification."""
    ret_lst = []
    for i in range(y.shape[0]):
        predictions = np.argsort(p[i, :])[-k:]
        true_labels = np.nonzero(y[i, :])[0]
        n_correct = np.in1d(true_labels, predictions, assume_unique=True).sum()  # 直接计算precision
        # pdb.set_trace()

        ret_lst.append(n_correct / min(len(true_labels), k))

    return np.mean(ret_lst)


#
# def topk_precision_group(y, p, k, grouped_y):
#     ret_lst = []
#     total_counter = Counter()
#     correct_counter = Counter()
#     prec_counter = Counter()
#
#     for i in range(y.shape[0]):
#         predictions = np.argsort(p[i, :])[-k:]
#         true_labels = np.nonzero(y[i, :])[0]
#         n_correct = np.in1d(true_labels, predictions, assume_unique=True).sum()  # 直接计算precision
#         for l in predictions:
#             total_counter[l] += 1
#             correct_counter[l] += np.in1d(l, true_labels, assume_unique=True).sum() # 以病为单位
#
#         ret_lst.append(n_correct / min(len(true_labels), k))
#
#     n_groups = len(grouped_y)
#     total_labels = [0] * n_groups  # 每个组别。
#
#     for i, group in enumerate(grouped_y):
#         for l in group:
#             prec_counter[l] = correct_counter[l] / total_counter[l] if total_counter[l] > 0 else 0
#             total_labels[i] += prec_counter[l]
#     result = [total_labels[i]/len(group) for i, group in enumerate(grouped_y)]
#
#
#     return np.mean(ret_lst), result
#

def topk_acc(y, p, k, grouped_y):
    """Computes top-k accuracy for multilabel classification."""
    total_counter = Counter()
    correct_counter = Counter()

    for i in range(y.shape[0]):
        true_labels = np.nonzero(y[i, :])[0]  # 真实的[1,0,1,0,1]->[32,33,46]
        predictions = np.argsort(p[i, :])[-k:]  # topk [10,9,8]
        for l in true_labels:
            total_counter[l] += 1 # 真正出现的次数
            correct_counter[l] += np.in1d(l, predictions, assume_unique=True).sum()  # 预测的次数，如果存在则加1

    y_grouped = grouped_y  # {'10':[32,33,34]}
    n_groups = len(y_grouped)
    total_labels = [0] * n_groups  # 每个组别。
    correct_labels = [0] * n_groups
    for i, group in enumerate(y_grouped): # 以组为单位计数
        for l in group:
            correct_labels[i] += correct_counter[l]
            total_labels[i] += total_counter[l]

    acc_at_k_grouped = [x / float(y) for x, y in zip(correct_labels, total_labels)]  # grouped
    acc_at_k = sum(correct_labels) / float(sum(total_labels))  # all

    return acc_at_k, acc_at_k_grouped


def regression_metrics_fn(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: Optional[List[str]] = None,
    aux_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """Computes metrics for regression.

    User can specify which metrics to compute by passing a list of metric names.
    The accepted metric names are:
        - mse: mean squared error
        - rmse: root mean squared error
        - mae: mean absolute error
        - r2: R squared

    If no metrics are specified, mse is computed by default.

    This function calls sklearn.metrics functions to compute the metrics. For
    more information on the metrics, please refer to the documentation of the
    corresponding sklearn.metrics functions.

    Args:
        y_true: True target values of shape (n_samples,).
        y_pred: Predicted target values of shape (n_samples,).
        metrics: List of metrics to compute. Default is ["mse"].

    Returns:
        Dictionary of metrics whose keys are the metric names and values are
            the metric values.

    Examples:
        >>> from pyhealth.metrics import regression_metrics_fn
        >>> y_true = np.array([1.0, 2.0, 3.0])
        >>> y_pred = np.array([1.1, 2.1, 3.1])
        >>> regression_metrics_fn(y_true, y_pred, metrics=["mse"])
        {'mse': 0.01}
    """
    if metrics is None:
        metrics = ["mse"]

    output = {}
    for metric in metrics:
        if metric == "mse":
            mse = sklearn_metrics.mean_squared_error(y_true, y_pred)
            output["mse"] = mse
        elif metric == "rmse":
            rmse = np.sqrt(sklearn_metrics.mean_squared_error(y_true, y_pred))
            output["rmse"] = rmse
        elif metric == "mae":
            mae = sklearn_metrics.mean_absolute_error(y_true, y_pred)
            output["mae"] = mae
        elif metric == "r2":
            r2 = sklearn_metrics.r2_score(y_true, y_pred)
            output["r2"] = r2
        else:
            raise ValueError(f"Unknown metric for regression: {metric}")
    return output



def multilabel_metrics_fn(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metrics: Optional[List[str]] = None,
    threshold: float = config['THRES'],
    y_predset: Optional[np.ndarray] = None,
    aux_data: Optional[Dict[str, Any]] = None,
    patient_ids: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Computes metrics for multilabel classification.

    User can specify which metrics to compute by passing a list of metric names.
    The accepted metric names are:
        - roc_auc_micro: area under the receiver operating characteristic curve,
          micro averaged
        - roc_auc_macro: area under the receiver operating characteristic curve,
          macro averaged
        - roc_auc_weighted: area under the receiver operating characteristic curve,
          weighted averaged
        - roc_auc_samples: area under the receiver operating characteristic curve,
          samples averaged
        - pr_auc_micro: area under the precision recall curve, micro averaged
        - pr_auc_macro: area under the precision recall curve, macro averaged
        - pr_auc_weighted: area under the precision recall curve, weighted averaged
        - pr_auc_samples: area under the precision recall curve, samples averaged
        - accuracy: accuracy score
        - f1_micro: f1 score, micro averaged
        - f1_macro: f1 score, macro averaged
        - f1_weighted: f1 score, weighted averaged
        - f1_samples: f1 score, samples averaged
        - precision_micro: precision score, micro averaged
        - precision_macro: precision score, macro averaged
        - precision_weighted: precision score, weighted averaged
        - precision_samples: precision score, samples averaged
        - recall_micro: recall score, micro averaged
        - recall_macro: recall score, macro averaged
        - recall_weighted: recall score, weighted averaged
        - recall_samples: recall score, samples averaged
        - jaccard_micro: Jaccard similarity coefficient score, micro averaged
        - jaccard_macro: Jaccard similarity coefficient score, macro averaged
        - jaccard_weighted: Jaccard similarity coefficient score, weighted averaged
        - jaccard_samples: Jaccard similarity coefficient score, samples averaged
        - ddi: drug-drug interaction score (specifically for drug-related tasks, such as drug recommendation)
        - hamming_loss: Hamming loss
        - cwECE: classwise ECE (with 20 equal-width bins). Check :func:`pyhealth.metrics.calibration.ece_classwise`.
        - cwECE_adapt: classwise adaptive ECE (with 20 equal-size bins). Check :func:`pyhealth.metrics.calibration.ece_classwise`.

    The following metrics related to the prediction sets are accepted as well, but will be ignored if y_predset is None:
        - fp: Number of false positives.
        - tp: Number of true positives.


    If no metrics are specified, pr_auc_samples is computed by default.

    This function calls sklearn.metrics functions to compute the metrics. For
    more information on the metrics, please refer to the documentation of the
    corresponding sklearn.metrics functions.

    Args:
        y_true: True target values of shape (n_samples, n_labels).
        y_prob: Predicted probabilities of shape (n_samples, n_labels).
        metrics: List of metrics to compute. Default is ["pr_auc_samples"].
        threshold: Threshold to binarize the predicted probabilities. Default is 0.5.

    Returns:
        Dictionary of metrics whose keys are the metric names and values are
            the metric values.

    Examples:
        >>> from pyhealth.metrics import multilabel_metrics_fn
        >>> y_true = np.array([[0, 1, 1], [1, 0, 1]])
        >>> y_prob = np.array([[0.1, 0.9, 0.8], [0.05, 0.95, 0.6]])
        >>> multilabel_metrics_fn(y_true, y_prob, metrics=["accuracy"])
        {'accuracy': 0.5}
    """
    if metrics is None:
        metrics = ["pr_auc_samples"]
    prediction_set_metrics = ['tp', 'fp']

    y_pred = y_prob.copy()
    y_pred[y_pred >= threshold] = 1
    y_pred[y_pred < threshold] = 0

    # if config['PCF_MODEL'] in ['COGNet', 'VITA']: # 这种推理其实也只是近似，因为原版是通过串形decode的
    #     y_pred = y_prob.copy() # B，M
    #     # 找到每行排名前45的索引
    #     topk_values, topk_indices = torch.topk(y_pred, 45, dim=1) # 一般是生成45个
    #     # 创建一个与x相同大小的零张量
    #     result = torch.zeros_like(x)
    #     # 将前45个元素的位置设置为1
    #     result.scatter_(1, topk_indices, 1)
    #     y_pred = result
    # else:
    #     y_pred = y_prob.copy()
    #     y_pred[y_pred >= threshold] = 1
    #     y_pred[y_pred < threshold] = 0

    output = {}
    for metric in metrics:
        if metric == "roc_auc_micro":
            roc_auc_micro = sklearn_metrics.roc_auc_score(
                y_true, y_prob, average="micro"
            )
            output["roc_auc_micro"] = roc_auc_micro
        elif metric =='avg_drug':
            output['avg_drug'] = np.mean(y_pred.sum(1))
        elif metric == "roc_auc_macro":
            roc_auc_macro = sklearn_metrics.roc_auc_score(
                y_true, y_prob, average="macro"
            )
            output["roc_auc_macro"] = roc_auc_macro
        elif metric == "roc_auc_weighted":
            roc_auc_weighted = sklearn_metrics.roc_auc_score(
                y_true, y_prob, average="weighted"
            )
            output["roc_auc_weighted"] = roc_auc_weighted
        elif metric == "roc_auc_samples":
            roc_auc_samples = sklearn_metrics.roc_auc_score(
                y_true, y_prob, average="samples"
            )
            output["roc_auc_samples"] = roc_auc_samples
        elif metric == "pr_auc_micro":
            pr_auc_micro = sklearn_metrics.average_precision_score(
                y_true, y_prob, average="micro"
            )
            output["pr_auc_micro"] = pr_auc_micro
        elif metric == "pr_auc_macro":
            pr_auc_macro = sklearn_metrics.average_precision_score(
                y_true, y_prob, average="macro"
            )
            output["pr_auc_macro"] = pr_auc_macro
        elif metric == "pr_auc_weighted":
            pr_auc_weighted = sklearn_metrics.average_precision_score(
                y_true, y_prob, average="weighted"
            )
            output["pr_auc_weighted"] = pr_auc_weighted
        elif metric == "pr_auc_samples":
            pr_auc_samples = sklearn_metrics.average_precision_score(
                y_true, y_prob, average="samples"
            )
            output["pr_auc_samples"] = pr_auc_samples
        elif metric == "accuracy":
            accuracy = sklearn_metrics.accuracy_score(y_true.flatten(), y_pred.flatten())
            output["accuracy"] = accuracy
        elif metric == "f1_micro":
            f1_micro = sklearn_metrics.f1_score(y_true, y_pred, average="micro") # Diag不准
            output["f1_micro"] = f1_micro
        elif metric == "f1_macro":
            f1_macro = sklearn_metrics.f1_score(y_true, y_pred, average="macro")
            output["f1_macro"] = f1_macro
        elif metric == "f1_weighted":
            f1_weighted = sklearn_metrics.f1_score(y_true, y_pred, average="weighted")
            output["f1_weighted"] = f1_weighted
        elif metric == "f1_samples":
            f1_samples = sklearn_metrics.f1_score(y_true, y_pred, average="samples",  zero_division=1)
            output["f1_samples"] = f1_samples
        elif metric == "precision_micro":
            precision_micro = sklearn_metrics.precision_score(
                y_true, y_pred, average="micro"
            )
            output["precision_micro"] = precision_micro
        elif metric == "precision_macro":
            precision_macro = sklearn_metrics.precision_score(
                y_true, y_pred, average="macro"
            )
            output["precision_macro"] = precision_macro
        elif metric == "precision_weighted":
            precision_weighted = sklearn_metrics.precision_score(
                y_true, y_pred, average="weighted"
            )
            output["precision_weighted"] = precision_weighted
        elif metric == "precision_samples":
            precision_samples = sklearn_metrics.precision_score(
                y_true, y_pred, average="samples"
            )
            output["precision_samples"] = precision_samples
        elif metric == "recall_micro":
            recall_micro = sklearn_metrics.recall_score(y_true, y_pred, average="micro")
            output["recall_micro"] = recall_micro
        elif metric == "recall_macro":
            recall_macro = sklearn_metrics.recall_score(y_true, y_pred, average="macro")
            output["recall_macro"] = recall_macro
        elif metric == "recall_weighted":
            recall_weighted = sklearn_metrics.recall_score(
                y_true, y_pred, average="weighted"
            )
            output["recall_weighted"] = recall_weighted
        elif metric == "recall_samples":
            recall_samples = sklearn_metrics.recall_score(
                y_true, y_pred, average="samples"
            )
            output["recall_samples"] = recall_samples
        elif metric == "jaccard_micro":
            jaccard_micro = sklearn_metrics.jaccard_score(
                y_true, y_pred, average="micro"
            )
            output["jaccard_micro"] = jaccard_micro
        elif metric == "jaccard_macro":
            jaccard_macro = sklearn_metrics.jaccard_score(
                y_true, y_pred, average="macro"
            )
            output["jaccard_macro"] = jaccard_macro
        elif metric == "jaccard_weighted":
            jaccard_weighted = sklearn_metrics.jaccard_score(
                y_true, y_pred, average="weighted"
            )
            output["jaccard_weighted"] = jaccard_weighted
        elif metric == "jaccard_samples":
            jaccard_samples = sklearn_metrics.jaccard_score(
                y_true, y_pred, average="samples",  zero_division=1
            )
            output["jaccard_samples"] = jaccard_samples
        elif metric == "hamming_loss":
            hamming_loss = sklearn_metrics.hamming_loss(y_true, y_pred)
            output["hamming_loss"] = hamming_loss
        elif metric == "ddi":
            ddi_adj = np.load(os.path.join(CACHE_PATH, 'ddi_adj.npy'))
            y_pred = [np.where(item)[0] for item in y_pred]
            output["ddi_score"] = ddi_rate_score(y_pred, ddi_adj)
        elif metric == "topk_acc": # 这里的topk和rec还是有点不同
             all_acc, all_k_acc = topk_acc(y_true, y_prob, k=aux_data['topk'], grouped_y=aux_data['y_grouped'])
             output['topk_acc'] = all_acc
             output['topk_acc_grouped'] = all_k_acc
        elif metric == "topk_precision":
            visit_precision = topk_precision(y_true, y_prob, k=aux_data['topk'])
            # all_prec, all_k_prec = topk_precision_group(y_true, y_prob, k=aux_data['topk'], grouped_y=aux_data['y_grouped'])
            output['topk_precision'] = visit_precision # 这里有点奇怪啊
            # output['topk_precision_grouped'] = all_k_prec
            # output['topk_prec'] = all_prec


        elif metric == "group_rec":
            output['rec_grouped'] = group_rec(y_true, y_prob, y_pred, patient_ids, group_p=aux_data['p_grouped'])

        elif metric in {"cwECE", "cwECE_adapt"}:
            output[metric] = calib.ece_classwise(
                y_prob,
                y_true,
                bins=20,
                adaptive=metric.endswith("_adapt"),
                threshold=0.0,
            )
        elif metric in prediction_set_metrics:
            if y_predset is None:
                continue
            if metric == 'tp':
                output[metric] = (y_true * y_predset).sum(1).mean()
            elif metric == 'fp':
                output[metric] = ((1-y_true) * y_predset).sum(1).mean()
        else:
            raise ValueError(f"Unknown metric for multilabel classification: {metric}")

    return output
