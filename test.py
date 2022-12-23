import logging
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import hydra
from omegaconf import OmegaConf, DictConfig, ListConfig
import numpy as np
import os
import h5py
from collections import defaultdict

from models.base_model import BaseModel
from datasets.data import get_dataset
from train import get_transform_val, init_model
from challenge import marginalize_verb_noun, print_accuracies_epic, LOGITS_DIR
from train import DATASET_EVAL_CFG_KEY


def store_append_h5(endpoints, output_dir, save_file_name):
    output_fpath = os.path.join(output_dir, save_file_name)
    os.makedirs(output_dir, exist_ok=True)
    with h5py.File(output_fpath, 'a') as fout:
        for key, val in endpoints.items():
            if key not in fout:
                fout.create_dataset(key, data=val, compression='gzip', compression_opts=9,
                                    chunks=True, maxshape=(None, ) + val.shape[1:])
            else:
                fout[key].resize((fout[key].shape[0] + val.shape[0], ) + val.shape[1:])
                fout[key][-val.shape[0]:, ...] = val


def save_logits(model, data_loader: DataLoader, device, logger, save_dir=None, save_file_name=None):
    """Saves logits to given path, so that the logits can be used for ensemble or any other analysis"""
    # construct kwargs for forwarding
    kwargs = {}
    kwargs['mixup_fn'] = None
    kwargs['target'] = None
    kwargs['target_subclips'] = None
    kwargs['target_subclips_ignore_index'] = None

    for idx, data in enumerate(tqdm(data_loader)):
        data, _ = data
        feature_dict = {mod: tens.to(device, non_blocking=True) for mod, tens in data["data_dict"].items()}
        outputs, outputs_target = model(feature_dict, **kwargs)

        logits_key = 'logits/action'

        logits = {}
        if len(outputs[logits_key]) == 1:  # single modality or early fusion model
            modk = next(iter(outputs[logits_key].keys()))
            logits[f'{logits_key}_{modk}'] = outputs[f'{logits_key}'][modk][:, 0, :].detach().cpu().numpy()
        else:
            fusion_key = 'all-fused'
            logging.info(f'This model consists of multiple branches. '
                         f'Saving fusion branch "{fusion_key}" only ...')
            logits[f'{logits_key}_{fusion_key}'] = \
                outputs[f'{logits_key}'][fusion_key][:, 0, :].detach().cpu().numpy()

        store_append_h5(logits, save_dir, save_file_name)
    logger.info(f'Saved logits {logits.keys()} as {save_file_name} to {save_dir}.')


def evaluate(model, dataset, data_loader: DataLoader, device):
    """
    Computes the verb, noun and action performance of overall, unseen and tail
    """
    logits_key = 'logits/action'
    logits = defaultdict(list)

    # construct kwargs for forwarding
    kwargs = {}
    kwargs['mixup_fn'] = None
    kwargs['target'] = None
    kwargs['target_subclips'] = None
    kwargs['target_subclips_ignore_index'] = None

    # forwarding
    for idx, data in enumerate(tqdm(data_loader)):
        data, _ = data
        feature_dict = {mod: tens.to(device, non_blocking=True) for mod, tens in data["data_dict"].items()}
        outputs, outputs_target = model(feature_dict, **kwargs)

        if len(outputs[logits_key]) == 1:  # single modality or early fusion model
            modk = next(iter(outputs[logits_key].keys()))
            logits[f'{logits_key}_{modk}'].append(outputs[f'{logits_key}'][modk][:, 0, :].detach().cpu().numpy())
        else:
            fusion_key = 'all-fused'
            logging.info(f'This model consists of multiple branches. '
                         f'Saving fusion branch "{fusion_key}" only ...')
            logits[f'{logits_key}_{fusion_key}'].append(
                outputs[f'{logits_key}'][fusion_key][:, 0, :].detach().cpu().numpy())

    # since we only save one entry
    logits_array = np.concatenate(next(iter(logits.values())), axis=0)

    accs, scores = marginalize_verb_noun(logits_array, dataset, to_prob=True, compute_manyshot_unseen_tail=True)
    print_accuracies_epic(accs)


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    logger = logging.getLogger(__name__)

    device = torch.device('cuda')
    transform_val = get_transform_val(cfg)
    dataset_test = get_dataset(getattr(cfg, DATASET_EVAL_CFG_KEY), cfg.data_eval, transform_val, logger)
    logger.info('Creating data loaders...')
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=cfg.eval.batch_size or cfg.train.batch_size * 4,
        num_workers=cfg.workers,
        pin_memory=True,
        shuffle=False
    )

    num_classes = {key: len(val) for key, val in dataset_test.classes.items()}
    model = BaseModel(cfg.model, num_classes=num_classes, class_mappings=dataset_test.class_mappings)

    # load pretrained weights
    assert cfg.init_from_model is not None, 'Checkpoint is required for test.'
    ckpt_paths = cfg.init_from_model
    if not isinstance(ckpt_paths, ListConfig):
        ckpt_paths = [ckpt_paths]
    ckpt_paths = [os.path.join(cfg.cwd, 'checkpoints', path) for path in ckpt_paths]
    modules_to_keep = None
    _ = init_model(model, ckpt_paths, modules_to_keep, logger)

    model = nn.DataParallel(model, device_ids=range(cfg.num_gpus))
    model = model.to(device)  # Sends model to device 0, other gpus are used automatically.

    # test
    model.eval()
    with torch.no_grad():
        if 'save_name' in cfg:
            save_dir = os.path.join(cfg.cwd, LOGITS_DIR, cfg.init_from_model.split('/')[0])
            save_logits(model, data_loader_test, device, logger, save_dir, cfg.save_name)
        else:
            evaluate(model, dataset_test, data_loader_test, device)


if __name__ == '__main__':
    main()
