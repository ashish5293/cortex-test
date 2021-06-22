"""
@name: rec_metric.py
@author: Graydon Snider, Mohammad Jeihoonian
Updated on Jan 2020
"""
from app.models.cbcf.helpers.metrics import *


class ValidationMetrics(object):

    def __init__(self, actual: list, predict: list, k_val: int):

        self.actual = actual
        self.predict = predict
        self.k_val = k_val

    def grouping_score_ranked(self):

        methods = {'maf_at_k': avg_f1_at_k}

        metrics = dict()

        for key in methods:

            metrics.update({key: [methods.get(key)(self.actual, self.predict, self.k_val)]})

        return metrics
