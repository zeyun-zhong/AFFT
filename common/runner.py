"""Implementation of a training iteration"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Union, Callable, Optional

from common import utils

CLS_MAP_PREFIX = 'cls_map_'
PAST_LOGITS_PREFIX = 'past_'


class MultiDimCrossEntropy(nn.CrossEntropyLoss):
    """Will reshape the flatten initial dimensions and then incur loss"""

    def forward(self, inp, tgt,
                one_hot: bool = False,
                ignore_index: Union[torch.Tensor, None] = None):
        """
        Args:
            inp: (*, C)
            tgt: (*, )
            one_hot: whether the labels are already one-hotted
            ignore_index: index of inputs to be ignored
        """
        inp = inp.reshape(-1, inp.size(-1))
        tgt = tgt.reshape(-1,) if not one_hot else tgt.reshape(-1, tgt.size(-1))

        if ignore_index is not None:
            assert one_hot, "Target should be one-hotted."
            ignore_index = ignore_index.reshape(-1,)
            keep_index = ~ignore_index
            inp = inp[keep_index]
            tgt = tgt[keep_index]

        res = super().forward(inp, tgt)
        return res


class BasicLossAccuracy(nn.Module):
    """Computes acc1, acc5 and the three loss values
        1. loss for future action prediction,
        2. loss for past action prediction and
        3. loss for past feature regression.
    """

    def __init__(self):
        super().__init__()
        kwargs = {'ignore_index': -1}
        kwargs['reduction'] = 'none'  # to get batch level output
        self.cls_criterion = MultiDimCrossEntropy(**kwargs)
        self.reg_criterion = torch.nn.MSELoss()

    def forward_future_action(self, logits, tgt_val, mixup_enable, losses, metrics,
                              acc1_key, acc5_key, mt5r_key, loss_key, key_suffix=''):
        """Computes accuracy, mt5r and loss value of future action"""
        loss_future_action = self.cls_criterion(logits, tgt_val, one_hot=mixup_enable)

        sequence_index = 0
        if mixup_enable:
            # we add up the top1 and top2 predictions and labels
            _top_max_k_vals, top_max_k_inds = torch.topk(
                tgt_val, 2, dim=1, largest=True, sorted=True
            )
            idx_top1 = torch.arange(tgt_val.shape[0]), \
                       torch.tensor(sequence_index).repeat(tgt_val.shape[0]), \
                       top_max_k_inds[:, 0]
            idx_top2 = torch.arange(tgt_val.shape[0]), \
                       torch.tensor(sequence_index).repeat(tgt_val.shape[0]), \
                       top_max_k_inds[:, 1]
            preds = logits.detach().clone()
            preds[idx_top1] += preds[idx_top2]
            preds[idx_top2] = 0.0
            labels = top_max_k_inds[:, 0]
        else:
            preds = logits.detach()
            labels = tgt_val.clone()

        if len(labels.shape) == 1:  # single frame prediction
            labels = labels.unsqueeze(dim=-1)

        metrics[mt5r_key + key_suffix] = {
            'logits': preds[:, sequence_index, :].cpu().numpy(),
            'labels': labels[:, sequence_index].cpu().numpy()
        }

        dataset_max_classes = preds.size(-1)
        acc1, acc5 = utils.accuracy(preds, labels, topk=(1, min(5, dataset_max_classes)))

        losses[loss_key + key_suffix] = loss_future_action
        metrics[acc1_key + key_suffix] = acc1
        metrics[acc5_key + key_suffix] = acc5

    def forward_past_action(self, past_logits, past_target, mixup_enable,
                            losses, loss_key, past_target_ignore_index=None, key_suffix=''):
        frames_to_keep = None
        if mixup_enable:
            assert past_logits.shape == past_target.shape
            assert past_target_ignore_index is not None
            if frames_to_keep is not None:
                past_target_ignore_index = past_target_ignore_index[:, frames_to_keep]
            loss_past_action = self.cls_criterion(
                past_logits, past_target, one_hot=mixup_enable, ignore_index=past_target_ignore_index
            )
        else:
            past_target = past_target.squeeze(-1)  # this assumes the last dimension is 1
            assert past_logits.shape[:-1] == past_target.shape
            loss_past_action = self.cls_criterion(past_logits, past_target)

        losses[loss_key + key_suffix] = loss_past_action

    def forward(self, outputs, target, target_subclips,
                mixup_enable: bool = False, target_subclips_ignore_index: Union[Dict, None] = None):
        """
        Args:
            outputs['logits'] torch.Tensor (B, num_classes) or
                (B, T, num_classes)
                Latter in case of dense prediction
            target: {type: (B) or (B, T')}; latter in case of dense prediction
            target_subclips: {type: (B, #clips, T)}: The target for each input frame
            mixup_enable (bool): whether the targets are already one-hotted
            target_subclips_ignore_index: index of inputs to be ignored
        """
        losses = {}
        metrics = {}
        for tgt_type, tgt_val in target.items():
            # --------FUTURE ACTION PREDICTION-------
            for modk in outputs[f'logits/{tgt_type}']:
                logits = outputs[f'logits/{tgt_type}'][modk]

                # metric keys
                acc1_key, acc5_key = f'acc1_{tgt_type}_{modk}', f'acc5_{tgt_type}_{modk}'
                mt5r_key = f'mt5r_{tgt_type}_{modk}'
                loss_key = f'cls_{tgt_type}_{modk}'

                assert len(logits.shape) == 3  # Includes temporal dimension (B, T, C), even if T == 1
                self.forward_future_action(
                    logits, tgt_val, mixup_enable, losses, metrics,
                    acc1_key, acc5_key, mt5r_key, loss_key
                )

            # --------PAST ACTION PREDICTION-------
            past_logits_key = f'{PAST_LOGITS_PREFIX}logits/{tgt_type}'
            if past_logits_key in outputs and target_subclips is not None:
                for modk in outputs[past_logits_key]:
                    past_logits = outputs[past_logits_key][modk]
                    loss_key = f'past_cls_{tgt_type}_{modk}'
                    past_target_ignore_index = None if target_subclips_ignore_index is None \
                        else target_subclips_ignore_index[tgt_type]

                    self.forward_past_action(
                        past_logits, target_subclips[tgt_type], mixup_enable,
                        losses, loss_key, past_target_ignore_index
                    )

            # --------PAST FEATURE REGRESSION---------
            if 'orig_past' in outputs and 'past_futures' in outputs:
                orig_past_features = outputs['orig_past']
                updated_past_features = outputs['past_futures']

                for modk, updated_past_feature in updated_past_features.items():
                    if modk not in orig_past_features: continue
                    loss_key = f'past_reg_{modk}'
                    losses[loss_key] = self.reg_criterion(
                        updated_past_features[modk][:, 1:], orig_past_features[modk][:, 1:]
                    )

        return losses, metrics


def get_loss_wts(loss_wts: Dict, key: str) -> float:
    for k, v in loss_wts.items():
        if key.startswith(k):
            return v
    raise ValueError(f'{key} not contained in predefined loss_wts: {loss_wts}')


class Runner:
    """wrapper class of BasicLossAccuracy, runs on each batch, returns all metrics"""

    def __init__(self, model, device, loss_wts):
        super().__init__()
        self.model = model
        self.device = device
        self.loss_acc_fn = BasicLossAccuracy()
        self.loss_wts = loss_wts

    def _basic_preproc(self, data):
        if not isinstance(data, dict):
            video, target = data
            # Make a dict so that later code can use it
            data = {}
            data['video'] = video
            data['target'] = target
            data['idx'] = -torch.ones_like(target)
        return data

    @staticmethod
    def _reduce_loss(losses, loss_wts):
        # reduce the losses
        losses = {key: torch.mean(val) for key, val in losses.items()}
        # weight the losses
        losses_wtd = []
        for key, val in losses.items():
            this_loss_wt = get_loss_wts(loss_wts, key)
            if this_loss_wt > 0:
                losses_wtd.append(this_loss_wt * val)
        loss = torch.sum(torch.stack(losses_wtd))
        if torch.isnan(loss):
            raise ValueError('The loss is NaN!')
        losses_metric = {k: v.item() for k, v in losses.items()}  # prevents increasing gpu memory
        losses_metric['total_loss'] = loss.item()
        return loss, losses_metric

    def __call__(self,
                 data: Union[Dict[str, torch.Tensor],  # If dict
                             Tuple[torch.Tensor, torch.Tensor]],
                 mixup_fn: Optional[Callable] = None,
                 mixup_backbone: Optional[bool] = True):
        """
        Args:
            data (dict): Dictionary of all the data from the data loader
            mixup_fn: Mixup function
            mixup_backbone: Whether to mixup the inputs or the backbone outputs
        """
        data, timings = data  # Getting timings from dataloader
        data = self._basic_preproc(data)
        feature_dict = {mod: tens.to(self.device, non_blocking=True) for mod, tens in data["data_dict"].items()}
        target = {}
        target_subclips = {}
        for key in data['target'].keys():
            target[key] = data['target'][key].to(self.device, non_blocking=True)
        if 'target_subclips' in data:
            for key in data['target_subclips'].keys():
                target_subclips[key] = data['target_subclips'][key].to(self.device, non_blocking=True)
        else:
            target_subclips = None

        kwargs = {}
        kwargs['mixup_fn'] = None
        kwargs['target'] = target
        kwargs['target_subclips'] = target_subclips
        kwargs['target_subclips_ignore_index'] = None

        # the target will be one-hotted after mixup
        if mixup_fn is not None:
            # mixup the inputs
            if not mixup_backbone:
                feature_dict, target, target_subclips, target_subclips_ignore_index = \
                    mixup_fn(feature_dict, target, target_subclips)
                kwargs['target'] = target
                kwargs['target_subclips'] = target_subclips
                kwargs['target_subclips_ignore_index'] = target_subclips_ignore_index
            # mixup the backbone outputs
            else:
                kwargs['mixup_fn'] = mixup_fn

        outputs, outputs_targets = self.model(feature_dict, **kwargs)

        losses, metrics = self.loss_acc_fn(
            outputs,
            outputs_targets['target'],
            outputs_targets['target_subclips'],
            mixup_enable=(mixup_fn is not None),
            target_subclips_ignore_index=outputs_targets['target_subclips_ignore_index'],
        )
        loss, losses_metric = self._reduce_loss(losses, self.loss_wts)
        metrics.update(losses_metric)
        metrics.update(timings)
        return loss, metrics
