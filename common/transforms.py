import torch
import random


class PermuteRULSTMFeats:
    def __init__(self):
        pass

    def __call__(self, vid):
        return vid.permute(3, 0, 1, 2)


class ZeroMaskRULSTMFeats:
    """Mask random frames with zeros"""
    def __init__(self, mask_rate=0.2):
        self.mask_rate = mask_rate

    def __call__(self, vid):
        if self.mask_rate == 0:
            return vid
        num_frames = vid.size(0)
        num_masked_frames = round(num_frames * self.mask_rate)
        random_choices = random.sample(range(num_frames), num_masked_frames)
        vid[random_choices, :, :, :] = torch.zeros((num_masked_frames, vid.size(1), vid.size(2), vid.size(-1)))
        return vid
