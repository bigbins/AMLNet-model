from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np
import json
from prettytable import PrettyTable


class _StreamMetrics(object):
    def __init__(self):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def update(self, gt, pred):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def get_results(self):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def to_str(self, metrics):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def reset(self):
        """ Overridden by subclasses """
        raise NotImplementedError()  
class StreamSegMetrics(_StreamMetrics):
    """
    Stream Metrics for Semantic Segmentation Task
    """
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def update(self , label_preds,label_trues):
        label_trues = label_trues.detach().cpu().numpy().reshape(-1)
        label_preds = label_preds.detach().cpu().numpy().reshape(-1)
        for p, t in zip(label_preds.flatten(),label_trues.flatten()):
            self.confusion_matrix[p, t] += 1

    
    @staticmethod
    def to_str(results):
        return json.dumps(results, indent = 2)

    def get_results(self,labels):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
        table = PrettyTable()
        table.field_names = ["", 'Count', 'TP', "Precision", 'FDR', "Recall(TPR)", "Specificity(TNR)", 'Accuracy', 'F1']
        for i in range(self.n_classes):
            TP = hist[i, i]
            FP = np.sum(hist[i, :]) - TP
            FN = np.sum(hist[:, i]) - TP
            TN = np.sum(hist) - TP - FP - FN
            Count = np.sum(hist[:,i])
            Precision = round(TP / (TP + FP), 3) if TP + FP != 0 else 0.
            # Precision = round(TP / np.sum(self.matrix[i, :]), 3) if TP + FP != 0 else 0.
            FDR = round(1 - Precision, 3)
            Recall = round(TP / (TP + FN), 3) if TP + FN != 0 else 0.
            Specificity = round(TN / (TN + FP), 3) if TN + FP != 0 else 0.
            Accuracy = round((TP + TN) / (TP + FP + FN + TN), 3) if TP + TN + FP + FN != 0 else 0.
            F1 = round(2 * Precision * Recall / (Precision + Recall), 3)

            # table.add_row([self.labels[i], Count, TP, Precision, FDR, Recall, Specificity, Accuracy, F1])
            table.add_row([labels[i], Count, TP, Precision, FDR, Recall, Specificity, Accuracy, F1])
        print(table)

        print('the sum of images{}'.format(hist.sum()))
        # calculate accuracy
        sum_TP = 0
        for i in range(self.n_classes):
            sum_TP += hist[i, i]
        acc = sum_TP / np.sum(hist)
        print("the sum of TP{}".format(sum_TP))
        print("the model accuracy is ", acc)


    def get_results_topn(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
        recall = np.diag(hist) / hist.sum(axis=0)
        recall_avg = np.nanmean(recall)
        precision = np.diag(hist) / hist.sum(axis=1)
        precision_avg = np.nanmean(precision)
        """******"""
        return {
                "Overall Acc":  acc,
                "recall":list(recall),
                "recall_avg": recall_avg,
                "precision": list(precision),
                "precision_avg": precision_avg
            }
        
    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))
        
def accuracy(output, target, topk=(1,), metric=None,metric_top2=None,metric_top3=None):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        
        if metric != None:
            metric.update(pred[0],target)
        if metric_top2 !=None:
            metric_top2.update(pred[1],target)
        if metric_top3 !=None:
            metric_top3.update(pred[2],target)
        return res
        """**********"""