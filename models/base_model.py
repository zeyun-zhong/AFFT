"""Implementation of the base model framework, instantiating different backbones, fusion methods and future predictor
methods using hydra.utils.instantiate"""
from itertools import repeat
from typing import Dict, Tuple
import torch
import torch.nn as nn
import hydra
from omegaconf import OmegaConf
from common import utils

CLS_MAP_PREFIX = 'cls_map_'
PAST_LOGITS_PREFIX = 'past_'


class BaseModel(nn.Module):
    def __init__(self, model_cfg: OmegaConf, num_classes: Dict[str, int],
                 class_mappings: Dict[Tuple[str, str], torch.FloatTensor]):
        super().__init__()
        self.backbone = nn.ModuleDict()

        for mod, backbone_conf in model_cfg.common.backbones.items():
            self.backbone[mod] = hydra.utils.instantiate(backbone_conf)

        self.future_predictor = hydra.utils.instantiate(model_cfg.CMFP, model_cfg=model_cfg,
                                                        num_classes=num_classes, _recursive_=False)

        # Store the class mapping as buffers
        for (src, dst), mapping in class_mappings.items():
            self.register_buffer(f'{CLS_MAP_PREFIX}{src}_{dst}', mapping)

    def forward_singlecrop(self, data_dict, **kwargs):
        """
        Args:
            video (torch.Tensor, Bx#clipsxCxTxHxW)
            target_shape: The shape of the target. Some of these layers might
                be able to use this information.
        """
        feats_past = {}
        for mod, data in data_dict.items():
            feats = self.backbone[mod](data)
            # spatial mean B*clipsxCxT
            feats = torch.mean(feats, [-1, -2])
            feats = feats.permute((0, 1, 3, 2))
            if feats.ndim == 4:
                feats = torch.flatten(feats, 1, 2)  # BxTxF, T=10
            feats_past[mod] = feats

        target = kwargs['target']
        target_subclips = kwargs['target_subclips']
        target_subclips_ignore_index = kwargs['target_subclips_ignore_index']

        # Mixup the backbone outputs if required
        if kwargs['mixup_fn'] is not None:
            mixup_fn = kwargs['mixup_fn']
            feats_past, target, target_subclips, target_subclips_ignore_index = \
                mixup_fn(feats_past, target, target_subclips)

        # Future prediction
        outputs = self.future_predictor(feats_past)
        outputs_target = {
            'target': target,
            'target_subclips': target_subclips,
            'target_subclips_ignore_index': target_subclips_ignore_index
        }

        return outputs, outputs_target

    def forward(self, video_data, *args, **kwargs):
        """
            Args: video (torch.Tensor)
                Could be (B, #clips, C, T, H, W) or
                    (B, #clips, #crops, C, T, H, W)
            Returns:
                Final features
        """
        for mod, data in video_data.items():
            if data.ndim == 6:
                video_data[mod] = [data]
            elif data.ndim == 7 and data.size(2) == 1:
                video_data[mod] = [data.squeeze(2)]
            elif data.ndim == 7:
                video_data[mod] = torch.unbind(data, dim=2)
            else:
                raise NotImplementedError('Unsupported size %s' % data.shape)

        all_mods = sorted(list(video_data.keys()))
        all_data = [video_data[mod] for mod in all_mods]
        num_crops = max([len(sl) for sl in all_data])
        all_data = [sl * (num_crops // len(sl)) for sl in all_data]
        all_crops = list(zip(*all_data))

        video_data = [{m: c for m, c in zip(mods, crops)} for mods, crops in zip(repeat(all_mods), all_crops)]

        feats = [self.forward_singlecrop(el, *args, **kwargs) for el in video_data]

        # Since we only apply mixup in training and in training we only have one single crop,
        # it's fine to just use the index 0 here
        output_targets = feats[0][1]

        # convert to dicts of lists
        feats_merged = {}
        for out_dict, _ in feats:
            for key in out_dict:
                if key not in feats_merged:
                    feats_merged[key] = {k: [v] for k, v in out_dict[key].items()}
                else:
                    for k, v in feats_merged[key].items():
                            v.append(out_dict[key][k])

        # Average over the crops
        for out_key in feats_merged:
            if out_key == 'attentions':
                # we select the attentions from the first element, as for attention analysis we only have one crop
                feats_merged[out_key] = {k: el[0] for k, el in feats_merged[out_key].items()}
                continue
            feats_merged[out_key] = {k: torch.mean(torch.stack(el, dim=0), dim=0) for k, el in
                                     feats_merged[out_key].items()}

        return feats_merged, output_targets
