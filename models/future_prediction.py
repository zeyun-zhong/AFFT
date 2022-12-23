"""Implementation of different feature anticipation modules."""

import logging
import torch
import torch.nn as nn
import transformers
import hydra
import warnings

from functools import partial
from omegaconf import OmegaConf, DictConfig
from typing import Dict, List, Tuple
import abc
from torch import Tensor

PAST_LOGITS_PREFIX = 'past_'


class CrossModalFusionPrediction(nn.Module, metaclass=abc.ABCMeta):
    """base class cross modality future predictor"""

    def __init__(self, model_cfg: OmegaConf, num_classes, instantiate: bool = True):
        super().__init__()
        assert isinstance(model_cfg.modal_dims, DictConfig), f'cfg.model.modal_dims must be a Dict!'

        self.cfg = model_cfg
        self.num_classes = num_classes
        self.latent_dim = model_cfg.common.in_features
        self.fp_inter_dim = model_cfg.common.fp_inter_dim
        self.modality_dims = model_cfg.modal_dims

        # boolean options controlling future predictor and classifier
        self.common_predictor = model_cfg.common.share_predictors
        self.common_classifier = model_cfg.common.share_classifiers
        self.modality_cls = model_cfg.common.modality_cls
        self.fusion_cls = model_cfg.common.fusion_cls

        if instantiate:
            self.mapping = self._init_mapping_layer()
            self.fuser = self._init_fuser(model_cfg)
            self.future_predictor = self._init_future_predictor(model_cfg, self.common_predictor)

        self.classifiers = self._init_classifiers(
            self.latent_dim, self.modality_dims, self.num_classes, self.common_classifier, self.cfg.dropout,
            self.modality_cls, self.fusion_cls)

    def _init_mapping_layer(self):
        mapping_layer = nn.ModuleDict()
        for mod in self.modality_dims.keys():
            # let the class defined in feature_mapping.py handle the case where in_features == out_features
            mapping_layer[mod] = hydra.utils.instantiate(
                self.cfg.mapping, in_features=self.modality_dims[mod], out_features=self.latent_dim)
            logging.info(f'Using {mapping_layer[mod]} for {mod}')
        return mapping_layer

    @staticmethod
    def _init_dimension_encoder(modality_dims, inter_dim, latent_dim):
        """replaces the encoder inside gpt2, ennabling modality specific dimension encoding"""
        del latent_dim  # not used here
        dim_encoder = nn.ModuleDict()
        for modk, mod_dim in modality_dims.items():
            dim_encoder[modk] = nn.Linear(mod_dim, inter_dim, bias=False) if mod_dim != inter_dim else nn.Identity()
        return dim_encoder

    @staticmethod
    def _init_dimension_decoder(modality_dims, inter_dim, latent_dim):
        """replaces the decoder inside gpt2, ennabling modality specific dimension decoding"""
        del latent_dim  # not used here
        dim_decoder = nn.ModuleDict()
        for modk, mod_dim in modality_dims.items():
            dim_decoder[modk] = nn.Linear(inter_dim, mod_dim, bias=False) if mod_dim != inter_dim else nn.Identity()
        return dim_decoder

    @staticmethod
    def _init_fuser(model_cfg):
        return hydra.utils.instantiate(model_cfg.fuser, _recursive_=False)

    def _init_future_predictor(self, model_cfg, common_predictor=False):
        encoder_dims = self.modality_dims

        self.dim_encoder = self._init_dimension_encoder(encoder_dims, self.fp_inter_dim, self.latent_dim)
        self.dim_decoder = self._init_dimension_decoder(encoder_dims, self.fp_inter_dim, self.latent_dim)

        if common_predictor:  # We can use a common future predictor, features are mapped.
            future_predictor = hydra.utils.instantiate(
                model_cfg.future_predictor, in_features=self.fp_inter_dim,
                dimension_mapping=False, _recursive_=False)
        else:
            future_predictor = nn.ModuleDict()
            for modk, mod_dim in model_cfg.modal_dims.items():
                future_predictor[modk] = hydra.utils.instantiate(
                    model_cfg.future_predictor, in_features=self.fp_inter_dim,
                    dimension_mapping=False, _recursive_=False)

        return future_predictor

    @staticmethod
    def _init_classifiers(latent_dim: int, modality_dims: DictConfig, num_classes: Dict, share_classifier: bool,
                          dropout: float, modality_cls: bool, fusion_cls: bool):
        assert modality_cls or fusion_cls, 'Modality-level and / or fusion classification!'

        classifiers = nn.ModuleDict()

        for cls_type, cls_dim in num_classes.items():
            mod_classifiers = nn.ModuleDict()

            # Common classifier implies common latent dim.
            common_classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(latent_dim, cls_dim)
                                              ) if share_classifier else None

            if modality_cls:
                for modk, mod_dim in modality_dims.items():
                    mod_classifiers[modk] = nn.Sequential(nn.Dropout(dropout), nn.Linear(mod_dim, cls_dim)
                                                          ) if not common_classifier else common_classifier

            if fusion_cls:
                mod_classifiers['all-fused'] = nn.Sequential(
                    nn.Dropout(dropout), nn.Linear(latent_dim, cls_dim)
                    ) if not common_classifier else common_classifier

            classifiers.update({cls_type: mod_classifiers})
        return classifiers

    @staticmethod
    def ordered_feature_list(x_d: Dict[str, Tensor], feats_order: List) -> List[Tensor]:
        """Converts multimodal feature dictionary to a list according to the given order
        used for cmfuser"""
        tensor_list = []
        for i, modk in enumerate(feats_order):
            tensor_list.append(x_d[modk])
        return tensor_list

    def feature_mapping(self, x_d: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Map features to convert to a common dimension.
        :param x_d: a dict containing multi-modal feature vectors with different sizes
        :return: a dict containing multi-modal feature vectors with same size
        """
        out = {}

        for modk, x in x_d.items():
            out[modk] = self.mapping[modk](x)
        return out

    def apply_classifier(self, input_feat, outputs_prefix=''):
        out = {}
        for classk in self.num_classes.keys():
            if classk in self.classifiers:
                out[f'{outputs_prefix}logits/{classk}'] = {
                    modk: self.classifiers[classk][modk](input_feat[modk])
                    for modk in self.classifiers[classk].keys() if modk in input_feat}
            else:
                raise ValueError(f'Classifier for {classk} does not exist.')
        return out

    @staticmethod
    def prepare_output(z, z_hat, fusions):
        """
        Uses original feature sequence and predicted feature sequence and provides a dict which contains a
        corresponding future feature sequence (original and its prediction are at the same position) as well as
        future and fusion
        """
        out = {}

        out['orig_past'] = z
        out['future'] = z_hat
        out["all-fused"] = fusions
        out['past_futures'] = {}

        B, T, C = next(iter(z.values())).shape
        for modk in out['future'].keys():
            # Future predictions are prepended by one original prediction, last is removed (no counterpart).
            out['past_futures'][modk] = torch.cat([out['orig_past'][modk][:, :1],
                                                   out['future'][modk][:, :(T - 1)]], dim=1)

            # For the future result we only use the last feature.
            out['future'][modk] = out['future'][modk][:, (T - 1):]

        for modk in out['all-fused'].keys():
            # For the future result we only use the last feature.
            out['all-fused'][modk] = out['all-fused'][modk][:, (T - 1):]

        return out

    @abc.abstractmethod
    def forward(self, x: Tensor) -> Dict[str, Dict[str, Tensor]]:
        raise NotImplementedError


class IndividualFuturePrediction(CrossModalFusionPrediction):
    """Individual modality future predictor"""

    def __init__(self, model_cfg: OmegaConf, num_classes):
        # Individual forwarding, fusion not possible.
        assert not model_cfg.common.fusion_cls

        # Set instantiate to False, since we donot need fuser and mapping layer in this case
        super().__init__(model_cfg, num_classes=num_classes, instantiate=False)

        self.future_predictor = self._init_future_predictor(model_cfg, self.common_predictor)

    def forward(self, z: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, torch.Tensor]]:
        z_hat = {}
        attentions = {}
        for modk, z_unimod in z.items():
            z_unimod_enc = self.dim_encoder[modk](z_unimod)

            if self.common_predictor:
                z_hat_unimod_enc, atts = self.future_predictor(z_unimod_enc, self.cfg.common.fp_output_len)
            else:
                z_hat_unimod_enc, atts = self.future_predictor[modk](z_unimod_enc, self.cfg.common.fp_output_len)

            z_hat_unimod = self.dim_decoder[modk](z_hat_unimod_enc)

            z_hat[modk] = z_hat_unimod
            attentions[modk] = atts

        out = self.prepare_output(z, z_hat, {})  # In this case no fusion results.

        # This is class dependant.
        feats_final = out["future"]

        out.update(self.apply_classifier(out["past_futures"], outputs_prefix=PAST_LOGITS_PREFIX))
        out.update(self.apply_classifier(feats_final))

        return out


class CMFPEarly(CrossModalFusionPrediction):
    """cross modality future predictor, early fusion version:
    feature of different modalities are fused before future prediction module"""

    def __init__(self, model_cfg: OmegaConf, num_classes):
        logger = logging.getLogger(__name__)

        if not model_cfg.common.share_classifiers:
            logger.warning("Enforcing shared classifier for early CMFP.")
            model_cfg.common.share_classifiers = True  # This is necessary for the fusion.

        if not model_cfg.common.share_predictors:
            logger.warning("Enforcing shared predictor for early CMFP.")
            model_cfg.common.share_predictors = True  # This is implied by the early fusion.

        super().__init__(model_cfg, num_classes=num_classes)

    @staticmethod
    def _init_dimension_encoder(modality_dims, inter_dim, latent_dim):
        del modality_dims  # Ignoring the modality dims, since we encode already fused features.
        dim_encoder = nn.Linear(latent_dim, inter_dim, bias=False) if latent_dim != inter_dim else nn.Identity()
        return dim_encoder

    @staticmethod
    def _init_dimension_decoder(modality_dims, inter_dim, latent_dim):
        del modality_dims  # Ignoring the modality dims, since we encode already fused features.
        dim_decoder = nn.Linear(inter_dim, latent_dim, bias=False) if latent_dim != inter_dim else nn.Identity()
        return dim_decoder

    def forward(self, feats: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, torch.Tensor]]:
        feats_order = [mod for mod in self.cfg.modal_feature_order if mod in feats]
        x_d = feats

        x_hat = self.feature_mapping(x_d)

        order_feature_func = partial(self.ordered_feature_list, feats_order=feats_order)
        z, modality_attns = self.fuser(x_hat, order_feature_func)

        # future prediction
        z_enc = self.dim_encoder(z)
        z_hat_enc, temporal_attns = self.future_predictor(z_enc, self.cfg.common.fp_output_len)
        z_hat = self.dim_decoder(z_hat_enc)

        # Since we fuse early, the fused features are the 'originals' in relation to future prediction.
        z = {"all-fused": z}
        z_hat = {"all-fused": z_hat}
        attentions = {"all-fused": {
            'modality_attns': modality_attns,
            'temporal_attns': temporal_attns
        }}

        # Early fusion: feats before future prediction are fused. Makes a copy, that way can be altered independently.
        fusion = {k: v[:] for k, v in z.items()}

        out = self.prepare_output(z, z_hat, fusion)

        # This is class dependant.
        feats_final = out["future"]

        out.update(self.apply_classifier(out["past_futures"], outputs_prefix=PAST_LOGITS_PREFIX))
        out.update(self.apply_classifier(feats_final))
        out['attentions'] = attentions

        return out


class CMFPScoreFusion(CrossModalFusionPrediction):
    def __init__(self, model_cfg: OmegaConf, num_classes):
        logger = logging.getLogger(__name__)
        # fusion_cls not needed since we directly fuse the classification scores
        assert not model_cfg.common.fusion_cls

        if not model_cfg.common.modality_cls:
            logger.warning("Enforcing modality classification for CMFPScoreFusion.")
            model_cfg.common.modality_cls = True  # This is necessary for the past feature classifier.

        super().__init__(model_cfg, num_classes=num_classes)

    def forward(self, z: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, torch.Tensor]]:
        feats_order = [mod for mod in self.cfg.modal_feature_order if mod in z]

        # future prediction
        z_hat = {}
        attentions = {}
        for modk, z_unimod in z.items():
            z_unimod_enc = self.dim_encoder[modk](z_unimod)

            if self.common_predictor:
                z_hat_unimod_enc, atts = self.future_predictor(z_unimod_enc, self.cfg.common.fp_output_len)
            else:
                z_hat_unimod_enc, atts = self.future_predictor[modk](z_unimod_enc, self.cfg.common.fp_output_len)

            z_hat_unimod = self.dim_decoder[modk](z_hat_unimod_enc)

            z_hat[modk] = z_hat_unimod
            attentions[modk] = atts

        # we concat the first frame with future frames
        z_hat_cat = {}
        for modk, z_unimod in z.items():
            z_hat_cat[modk] = torch.cat([z[modk][:, :1, :], z_hat[modk][:, :, :]], dim=1)

        # Map to common dim
        z_hat_cat = self.feature_mapping(z_hat_cat)

        order_feature_func = partial(self.ordered_feature_list, feats_order=feats_order)
        modality_attns = self.fuser(z_hat_cat, order_feature_func)

        # This is class dependant.
        out = self.prepare_output(z, z_hat, fusions={})
        logits_past = self.apply_classifier(out["past_futures"], outputs_prefix=PAST_LOGITS_PREFIX)
        logits_future = self.apply_classifier(out['future'])

        for classk in self.num_classes.keys():
            logits_past_orig = logits_past[f'{PAST_LOGITS_PREFIX}logits/{classk}']
            logits_future_orig = logits_future[f'logits/{classk}']
            logits_past_final = torch.zeros_like(next(iter(logits_past_orig.values())))
            logits_future_final = torch.zeros_like(next(iter(logits_future_orig.values())))
            for i, modk in enumerate(feats_order):
                logits_past_final += modality_attns[:, :-1, i].unsqueeze(-1) * logits_past_orig[modk]
                logits_future_final += modality_attns[:, -1:, i].unsqueeze(-1) * logits_future_orig[modk]
            out[f'{PAST_LOGITS_PREFIX}logits/{classk}'] = {'all-fused': logits_past_final}
            out[f'logits/{classk}'] = {'all-fused': logits_future_final}
        return out


class BaseFuturePredictor(nn.Module):
    """future predictor for single modality, modified from AVT"""

    def __init__(self, in_features, inter_dim=2048, n_layer=6, n_head=4,
                 embd_pdrop=0.1, resid_pdrop=0.1, attn_pdrop=0.1, output_attentions=False,
                 dimension_mapping=False):
        super().__init__()
        self.in_features = in_features
        self.output_attentions = output_attentions

        # map the original dimension to an intermediate dimension or vice versa
        if dimension_mapping:
            warnings.warn('Using dimension mapping inside GPT2 is deprecated.')
        self.encoder = nn.Linear(in_features, inter_dim, bias=False) if dimension_mapping else nn.Identity()
        self.decoder = nn.Linear(inter_dim, in_features, bias=False) if dimension_mapping else nn.Identity()

        # This already has the LayerNorm inside residual, as Naman suggested
        # causal mask will be automatically computed, not nessasary to be given, line 205 in modeling_gpt2.py
        self.gpt_model = transformers.GPT2Model(
            transformers.GPT2Config(
                n_embd=inter_dim,
                n_layer=n_layer,
                n_head=n_head,
                vocab_size=in_features,
                use_cache=True,
                embd_pdrop=embd_pdrop,
                resid_pdrop=resid_pdrop,
                attn_pdrop=attn_pdrop
                )
            )
        # Not needed, encoder will take care of it.
        del self.gpt_model.wte

    def forward(self, feats: torch.Tensor, output_len: int = 1) -> Tuple[torch.Tensor, Dict]:
        """
        :param feats: sequence of features of shape (B, T, C)
        :param output_len: number of future sequence
        :return: tuple of predicted future features of shape (B, T, C) and temporal attentions (if required)
        """
        addl_endpoints = {}
        feats = self.encoder(feats)
        past = None
        all_outputs_decoded = []
        for output_id in range(output_len):
            pred_so_far = sum([el.size(1) for el in all_outputs_decoded])
            position_ids = torch.arange(pred_so_far, pred_so_far + feats.size(1), dtype=torch.long, device=feats.device)
            outputs = self.gpt_model(inputs_embeds=feats,
                                     past_key_values=past,
                                     position_ids=position_ids,
                                     output_attentions=self.output_attentions)
            last_hidden_state = outputs.last_hidden_state
            past = outputs.past_key_values
            # For visualization later, if output_attentions is set to True
            if outputs.attentions is not None:
                # dimensions will be (batch_size, nlayers, nheads, seqlen, seqlen)
                addl_endpoints[f'gpt2_att_{output_id}'] = torch.stack(outputs.attentions).transpose(0, 1)

            all_outputs_decoded.append(self.decoder(last_hidden_state))
            feats = last_hidden_state[:, -1:, :]

        all_outputs_decoded = torch.cat(all_outputs_decoded, dim=1)  # updated past feats and future feats
        return all_outputs_decoded, addl_endpoints
