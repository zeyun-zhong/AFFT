"""
Implementation of mixup with ignore class, since some sequences donot have gt labels
Mixup: Beyond Empirical Risk Minimization (https://arxiv.org/abs/1710.09412)
"""

from typing import Dict, Sequence, Union
import torch


def batch_wo_ignore_cls(target_subclips: torch.Tensor, ignore_cls=-1):
    target_subclips = target_subclips.squeeze(-1)  # avoid dim like (B, 1)
    assert target_subclips.ndim == 2, "Target subclips should have dimension of 2."
    batch_index = (target_subclips != ignore_cls).all(-1)
    return batch_index


def convert_to_one_hot(
    targets: torch.Tensor,
    num_class: int,
    label_smooth: float = 0.0,
) -> torch.Tensor:
    """
    This function converts target class indices to one-hot vectors,
    given the number of classes.
    Args:
        targets (torch.Tensor): Index labels to be converted.
        num_class (int): Total number of classes.
        label_smooth (float): Label smooth value for non-target classes. Label smooth
            is disabled by default (0).
    """
    assert (
        torch.max(targets).item() < num_class
    ), "Class Index must be less than number of classes"
    assert 0 <= label_smooth < 1.0, "Label smooth value needs to be between 0 and 1."

    targets = targets.squeeze(-1)  # avoids dim like (B, 1)

    non_target_value = label_smooth / num_class
    target_value = 1.0 - label_smooth + non_target_value
    one_hot_targets = torch.full(
        (*targets.shape, num_class),
        non_target_value,
        dtype=None,
        device=targets.device,
    )
    one_hot_targets.scatter_(-1, targets.unsqueeze(-1), target_value)
    return one_hot_targets


def _mix_labels(
    labels: torch.Tensor,
    num_classes: int,
    lam: float = 1.0,
    label_smoothing: float = 0.0,
    one_hot: bool = False,
):
    """
    This function converts class indices to one-hot vectors and mix labels, given the
    number of classes.
    Args:
        labels (torch.Tensor): Class labels.
        num_classes (int): Total number of classes.
        lam (float): lamba value for mixing labels.
        label_smoothing (float): Label smoothing value.
    """
    if one_hot:
        labels1 = labels
        labels2 = labels.flip(0)
    else:
        labels1 = convert_to_one_hot(labels, num_classes, label_smoothing)
        labels2 = convert_to_one_hot(labels.flip(0), num_classes, label_smoothing)
    return labels1 * lam + labels2 * (1.0 - lam)


def _mix(inputs: torch.Tensor, batch_wo_ignore_index: torch.Tensor, lam: float = 1.0) -> torch.Tensor:
    """
    mix inputs of specific indexes
    :param inputs: input tensor
    :param batch_wo_ignore_index: index of batches where ignore class does occur
    :param lam: mixup lambda
    :return: mixed inputs
    """
    inputs_selected = inputs[batch_wo_ignore_index]
    inputs_flipped = inputs_selected.flip(0).mul_(1.0 - lam)
    inputs_selected.mul_(lam).add_(inputs_flipped)
    inputs[batch_wo_ignore_index] = inputs_selected
    return inputs


class MixUp(torch.nn.Module):
    """
    Mixup: Beyond Empirical Risk Minimization (https://arxiv.org/abs/1710.09412)
    """

    def __init__(
        self,
        alpha: float = 1.0,
        label_smoothing: Dict = 0.0,
        num_classes: Dict = None,
        one_hot: bool = False,
        ignore_cls=-1,
    ) -> None:
        """
        This implements MixUp for videos.
        Args:
            alpha (float): Mixup alpha value.
            label_smoothing (dict): Label smoothing value.
            num_classes (dict, int): Number of total classes.
            one_hot (bool): whether labels are already in one-hot form
            ignore_cls (int): class that will not contribute for backpropagation
        """
        super().__init__()
        self.mixup_beta_sampler = torch.distributions.beta.Beta(alpha, alpha)
        self.label_smoothing = label_smoothing
        self.num_classes = num_classes
        self.one_hot = one_hot
        self.ignore_cls = ignore_cls

    def forward(
        self,
        x_video: Dict,
        labels: Dict,
        labels_subclips: Union[Dict, None],
    ) -> Sequence[Union[Dict, None]]:
        """
        :param x_video: Dict of inputs from different modalities
        :param labels: Dict of action / (verb, noun) labels
        :param labels_subclips: Dict of action / (verb, noun) labels for past frames
        :return: mixed inputs and labels
        """
        assert next(iter(x_video.values())).size(0) > 1, "MixUp cannot be applied to a single instance."
        batch_wo_ignore_index = [...]

        # convert labels to one-hot format
        labels_out = {key: convert_to_one_hot(val, self.num_classes[key], self.label_smoothing[key])
                      for key, val in labels.items()}

        if labels_subclips is not None:
            labels_subclips_curr = next(iter(labels_subclips.values()))
            batch_wo_ignore_index = batch_wo_ignore_cls(labels_subclips_curr, self.ignore_cls)

            # convert labels_subclips to one-hot format
            labels_subclips_out = {}
            labels_subclips_ignore_index = {}
            for key, val in labels_subclips.items():
                val_tmp = val.clone()
                # we first assign those ignore classes 0, so that the code works
                # the runner will filter out these ignore classes later
                subclips_ignore_index = val == self.ignore_cls
                val_tmp[subclips_ignore_index] = 0
                labels_subclips_ignore_index[key] = subclips_ignore_index
                val_one_hot = convert_to_one_hot(val_tmp, self.num_classes[key], self.label_smoothing[key])
                labels_subclips_out[key] = val_one_hot

            if batch_wo_ignore_index.sum() <= 1:
                # we don't do mixup here, since there is only one single batch wo ignore index
                return x_video, labels_out, labels_subclips_out, labels_subclips_ignore_index

        mixup_lambda = self.mixup_beta_sampler.sample()

        # mix inputs
        x_out = {
            modk: _mix(x.clone(), batch_wo_ignore_index, mixup_lambda)
            for modk, x in x_video.items()
        }

        # mix labels
        labels_out = {
            key: _mix(val, batch_wo_ignore_index, mixup_lambda)
            for key, val in labels_out.items()
        }

        if labels_subclips is None:
            return x_video, labels_out, None, None

        # mix labels of past frames
        labels_subclips_out = {
            key: _mix(val, batch_wo_ignore_index, mixup_lambda)
            for key, val in labels_subclips_out.items()
        }

        return x_out, labels_out, labels_subclips_out, labels_subclips_ignore_index
