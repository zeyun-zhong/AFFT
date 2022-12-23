"""Implementation of metrictracker in training process"""

import numpy as np
from typing import Dict
from common.utils import is_dist_avail_and_initialized
import torch
import torch.distributed as dist


class MeanTopKRecallMeter(object):
    """adapted from RULSTM"""
    def __init__(self, name, num_classes: int, k=5, string_format='{:.3f}'):
        self.name = name
        self.num_classes = num_classes
        self.k = k
        self.string_format = string_format

    def reset(self):
        self.tps = np.zeros(self.num_classes)
        self.nums = np.zeros(self.num_classes)

    def update(self, logits_labels_dict, n=1):
        del n # not used here
        scores = logits_labels_dict['logits']
        labels = logits_labels_dict['labels']
        tp = (np.argsort(scores, axis=1)[:, -self.k:] == labels.reshape(-1, 1)).max(1)
        for l in np.unique(labels):
            self.tps[l] += tp[labels == l].sum()
            self.nums[l] += (labels == l).sum()

    def synchronize_between_processes(self):
        if not is_dist_avail_and_initialized():
            return
        tps_all = torch.tensor(self.tps, device='cuda')
        nums_all = torch.tensor(self.nums, device='cuda')
        dist.barrier()
        dist.all_reduce(tps_all)
        dist.all_reduce(nums_all)
        self.tps = tps_all
        self.nums = nums_all

    @property
    def value(self):
        tps = self.tps[self.nums > 0]
        nums = self.nums[self.nums > 0]
        recalls = tps / nums
        if len(recalls) > 0:
            return recalls.mean() * 100
        else:
            return None

    def to_string(self):
        return self.string_format.format(self.value)


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self, name, string_format='{:.3f}'):
        self.name = name
        self.string_format = string_format

    def reset(self):
        self.val, self.avg, self.sum, self.count = 0, 0, 0, 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n

    def synchronize_between_processes(self):
        if not is_dist_avail_and_initialized():
            return
        count_all = torch.tensor(self.count, device='cuda')
        sum_all = torch.tensor(self.sum, device='cuda')
        dist.barrier()
        dist.all_reduce(count_all)
        dist.all_reduce(sum_all)
        self.count = count_all
        self.sum = sum_all

    @property
    def value(self):
        """returns the current floating average"""
        self.avg = self.sum / self.count
        return self.avg

    def to_string(self):
        return self.string_format.format(self.value)


class MetricTracker:
    """Interface of all metrics, tracks multiple metrics"""
    def __init__(self, num_classes: Dict[str, int]):
        self.training_metrics = {}
        self.validation_metrics = {}
        self.num_classes = num_classes
        self.training_prefix = 'train_'
        self.validation_prefix = 'val_'

    def _get_num_classes(self, name):
        num_classes = None
        for key, value in self.num_classes.items():
            if key in name:
                num_classes = value
        if num_classes is None:
            raise ValueError('Name of the mt5r metric muss contain action, verb or noun.')
        return num_classes

    def add_metric(self, name, is_training=None):
        meter = AverageMeter(name)
        if 'mt5r' in name:
            num_classes = self._get_num_classes(name)
            meter = MeanTopKRecallMeter(name, num_classes)

        # reset the meter
        meter.reset()

        if is_training is None:
            self.training_metrics[name] = meter
            self.validation_metrics[name] = meter
        elif is_training:
            self.training_metrics[name] = meter
        else:
            self.validation_metrics[name] = meter

    def update(self, metric_dict: Dict, batch_size: int, is_training: bool):
        if is_training:
            metrics = self.training_metrics
            prefix = self.training_prefix
        else:
            metrics = self.validation_metrics
            prefix = self.validation_prefix

        for key, value in metric_dict.items():
            key = prefix + key
            if key not in metrics:
                self.add_metric(key, is_training)
            metrics[key].update(value, batch_size)

    def synchronize_between_processes(self, is_training):
        if is_training:
            metrics = self.training_metrics
        else:
            metrics = self.validation_metrics

        for key in metrics:
            metrics[key].synchronize_between_processes()

    def reset(self):
        """reset all metrics at the beginning of each training epoch"""
        for name in self.training_metrics:
            self.training_metrics[name].reset()
        for name in self.validation_metrics:
            self.validation_metrics[name].reset()

    def get_all_data(self, is_training):
        """returns the current values of all tracked metrics"""
        if is_training:
            metrics = self.training_metrics
        else:
            metrics = self.validation_metrics
        data = {}
        for key in metrics:
            data[key] = metrics[key].value
        return data

    def get_data(self, metric_name, is_training):
        """returns the current value of the metric"""
        if is_training:
            return self.training_metrics[metric_name].value
        else:
            return self.validation_metrics[metric_name].value

    def to_string(self, is_training):
        """returns the string of all values"""
        if is_training:
            result = '\33[0;36;40m' + 'Training:    '
            metrics = self.training_metrics
        else:
            result = '\33[0;32;40m' + 'Validation:  '
            metrics = self.validation_metrics

        for key in metrics:
            result += metrics[key].name + ': ' + metrics[key].to_string() + '   '
        return result + '\033[0m'
