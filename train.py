import logging
import operator
import time
import os
from typing import Callable, Optional

import torch
from torch.utils.data import DataLoader
from torch import nn
import hydra
import torchvision
from omegaconf import OmegaConf, DictConfig, ListConfig
import itertools
from tqdm import tqdm
import wandb
import random
import numpy as np

from models.base_model import BaseModel
from common import transforms as T
from datasets.data import get_dataset
from common.runner import Runner
from common.metric_tracking import MetricTracker
from common import utils, mixup

DATASET_TRAIN_CFG_KEY = 'dataset_train'
DATASET_EVAL_CFG_KEY = 'dataset_eval'
CKPT_FNAME = 'checkpoint.pth'
CKPT_BEST_FNAME = 'checkpoint_best.pth'


def get_transform_train(cfg):
    mod_transform = {}

    for mod in cfg.model.modal_dims.keys():
        mod_transform[mod] = [
            T.ZeroMaskRULSTMFeats(mask_rate=cfg.data_train.zero_mask_rate),
            T.PermuteRULSTMFeats()
            ]

    mod_transform = {k: torchvision.transforms.Compose(trans) for k, trans in mod_transform.items()}
    return mod_transform


def get_transform_val(cfg):
    mod_transform = {}

    for mod in cfg.model.modal_dims.keys():
        mod_transform[mod] = [T.PermuteRULSTMFeats()]

    mod_transform = {k: torchvision.transforms.Compose(trans) for k, trans in mod_transform.items()}
    return mod_transform


def init_model(model, ckpt_paths, modules_to_keep, logger):
    """Initialize model with weights from ckpt_path.
    Args:
        ckpt_paths (list[str]): A list containing checkpoint paths
        modules_to_keep (list[str]: A list containing module names which are kept
    """
    logger.debug('Initing %s with ckpt path: %s, using modules in it %s',
                 model, ckpt_paths, modules_to_keep)

    assert isinstance(ckpt_paths, list)

    state_dict_loaded = {}
    for ckpt_path in ckpt_paths:
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        if 'model' in checkpoint.keys():
            state_dict_curr = checkpoint['model']
        elif 'model_state' in checkpoint.keys():
            state_dict_curr = checkpoint['model_state']
        else:
            state_dict_curr = checkpoint
        state_dict_loaded.update(state_dict_curr)

    filtered_state_dict = {}
    if modules_to_keep:
        if not isinstance(modules_to_keep, list) and not isinstance(modules_to_keep, ListConfig):
            modules_to_keep = [modules_to_keep]
        # Keep only the elements of state_dict that match modules to keep.
        for key, val in state_dict_loaded.items():
            for mod_name in modules_to_keep:
                if key.startswith(mod_name):
                    filtered_state_dict[key] = val
        state_dict_loaded = filtered_state_dict

    # Ignore any parameters/buffers (bn mean/var) where shape does not match
    for name, param in itertools.chain(model.named_parameters(),
                                       model.named_buffers()):
        if name in state_dict_loaded and state_dict_loaded[name].shape != param.shape:
            logger.warning('Ckpt shape mismatch for %s (%s vs %s). Ignoring.',
                           name, state_dict_loaded[name].shape, param.shape)
            del state_dict_loaded[name]

    missing_keys, unexp_keys = model.load_state_dict(state_dict_loaded, strict=False)

    logger.warning('Could not init from %s: %s', ckpt_path, missing_keys)
    logger.warning('Unused keys in %s: %s', ckpt_path, unexp_keys)

    state_dict_loaded = {k: v for k, v in state_dict_loaded.items() if k not in missing_keys and k not in unexp_keys}

    return state_dict_loaded


def get_dataloader(cfg, logger, dist_info):
    transforms_train = get_transform_train(cfg)
    transforms_val = get_transform_val(cfg)
    datasets_train = [
        get_dataset(getattr(cfg, el), cfg.data_train, transforms_train, logger)
        for el in cfg.keys() if el.startswith(DATASET_TRAIN_CFG_KEY)
        ]
    if len(datasets_train) > 1:
        dataset_train = torch.utils.data.ConcatDataset(datasets_train)
    else:
        dataset_train = datasets_train[0]

    dataset_val = get_dataset(getattr(cfg, DATASET_EVAL_CFG_KEY), cfg.data_eval, transforms_val, logger)

    train_sampler, val_sampler = None, None
    if dist_info['distributed']:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset_train,
            num_replicas=dist_info['world_size'],
            rank=dist_info['rank'],
            shuffle=True
        )
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset_val,
            num_replicas=dist_info['world_size'],
            rank=dist_info['rank'],
            shuffle=False
        )

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=cfg.train.batch_size,
        sampler=train_sampler,
        num_workers=cfg.workers,
        pin_memory=True,
        shuffle=(train_sampler is None),
        prefetch_factor=2
        )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=cfg.eval.batch_size or cfg.train.batch_size * 4,
        sampler=val_sampler,
        num_workers=cfg.workers,
        pin_memory=True,
        shuffle=False,
        prefetch_factor=2
        )
    return dataset_train, data_loader_train, dataset_val, data_loader_val


def store_checkpoint(fpath, model, optimizer, lr_scheduler, epoch):
    model_without_ddp = model
    if isinstance(model, nn.parallel.DistributedDataParallel) or isinstance(model, nn.parallel.DataParallel):
        model_without_ddp = model.module
    checkpoint = {
        "model":        model_without_ddp.state_dict(),
        "optimizer":    optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "epoch":        epoch,
        }
    logging.info('Storing ckpt at epoch %f to %s', epoch, fpath)
    utils.save_on_master(checkpoint, fpath)


def create_ckpt_path(cfg):
    expt_name = cfg.experiment_name
    fusion_method = cfg.model.fuser._target_.split('.')[-1]
    fp_method = cfg.model.CMFP._target_.split('.')[-1]
    modalities = '_'.join(cfg.model.modal_dims.keys())
    experiment_name = f'{fp_method}_{fusion_method}_{modalities}'
    experiment_name += f'_{expt_name}' if expt_name is not None else ''
    ckpt_path = os.path.join(cfg.cwd, 'checkpoints', experiment_name)
    os.makedirs(ckpt_path, exist_ok=True)
    if os.path.exists(os.path.join(ckpt_path, CKPT_BEST_FNAME)):
        if utils.question("This experiment already exists. Override? (WARNING)"):
            os.remove(os.path.join(ckpt_path, CKPT_BEST_FNAME))
        else:
            raise ValueError('This experiment is already done. '
                             'Please rename the experiment name to run it again.')

    return experiment_name, ckpt_path


def prepare_params(model, lr_wd, overall_lr, overall_wd):
    """Specify modules in lr_wd with given lr and wd, other modules are assigned with overall lr and wd
    :param lr_wd: list of list containing module name, learning rate and weightd decay
    :return: list of params
    """
    ori_params = {n: p for n, p in model.named_parameters()}

    if lr_wd is None:
        return [{'params': p, 'lr': overall_lr, 'weight_decay': overall_wd, 'name': n}
                for n, p in ori_params.items()]

    params = []
    rest_params = {n: p for n, p in ori_params.items()}
    for module_names, lr, wd in lr_wd:
        if OmegaConf.get_type(module_names) != list:
            module_names = [module_names]
        modules = [
            operator.attrgetter(el)(model) if el != '__all__' else model
            for el in module_names]
        this_params = {}
        for module_name, module in zip(module_names, modules):
            this_params.update({module_name + '.' + n: p for n, p in module.named_parameters()})
        params.extend([{'params': p, 'lr': lr, 'weight_decay': wd, 'name': n}
                       for n, p in this_params.items()])

        rest_params = {n: p for n, p in rest_params.items() if n not in this_params}

    params.extend([{'params': p, 'lr': overall_lr, 'weight_decay': overall_wd, 'name': n}
                   for n, p in rest_params.items()])

    params_final = []
    for param_lr in params:
        if param_lr['lr'] != 0.0:
            params_final.append(param_lr)
        else:
            param_lr['params'].requires_grad = False
    return params_final


def run_one_epoch(runner, optimizer, lr_scheduler, dataloader, metric_tracker, is_training, grad_clip=None,
                  mixup_fn: Optional[Callable] = None, mixup_backbone: Optional[bool] = True):
    """Training or validation of one epoch"""
    dl_start_time = time.perf_counter()
    all_time = time.perf_counter()

    for idx, data in enumerate(tqdm(dataloader)):
        dl_used_time = time.perf_counter() - dl_start_time
        s_runner_time = time.perf_counter()

        # forward
        # metric values should be a dict with metric names and metric values
        loss, metrics = runner(data, mixup_fn, mixup_backbone)

        e_runner_time = time.perf_counter()
        runner_used_time = e_runner_time - s_runner_time

        for k, v in metrics.items():
            if "T " in k:  # A little hacky to provide avg timings per batch.
                metrics[k] = torch.mean(metrics[k]).numpy()

        s_backprop_time = time.perf_counter()
        if is_training:
            optimizer.zero_grad()
            loss.backward()

            # Clip the gradients if required
            if grad_clip is not None:
                params_being_optimized = []
                for param_group in optimizer.param_groups:
                    params_being_optimized += param_group['params']
                assert len(params_being_optimized) > 0
                torch.nn.utils.clip_grad_norm_(params_being_optimized, grad_clip)

            optimizer.step()

            if lr_scheduler is not None:
                lr_scheduler.step()

        e_backprop_time = time.perf_counter()
        backprop_used_time = e_backprop_time - s_backprop_time

        batch_size = dataloader.batch_size

        all_used_time = time.perf_counter() - all_time

        metrics["T DataLoader"] = dl_used_time
        metrics["T Forward"] = runner_used_time
        metrics["T Backprop"] = backprop_used_time

        metric_tracker.update(metrics, batch_size, is_training)

        if is_training and idx % 200 == 0:
            print()
            for k, v in sorted(list(metric_tracker.training_metrics.items())):
                if "T " in k:
                    print(f"{k}: {v.avg:.3f}")

        dl_start_time = time.perf_counter()
        all_time = time.perf_counter()

    # gather the stats from all processes
    metric_tracker.synchronize_between_processes(is_training)


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    logger = logging.getLogger(__name__)

    experiment_name, ckpt_path = create_ckpt_path(cfg)

    # set random seed
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    np.random.seed(cfg.seed)

    # distributed info
    dist_info = utils.init_distributed_mode(logger, dist_backend=cfg.dist_backend)
    logger.info(f'Dist info:  world size {dist_info["world_size"]}')

    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = True

    dataset_train, dataloader_train, dataset_val, dataloader_val = get_dataloader(cfg, logger, dist_info)
    num_classes = {key: len(val) for key, val in dataset_train.classes.items()}
    model = BaseModel(cfg.model, num_classes=num_classes, class_mappings=dataset_train.class_mappings)

    # load pretrained weights if possible
    modules_to_keep = cfg.train.modules_to_keep
    if cfg.init_from_model:
        if not isinstance(cfg.init_from_model, ListConfig):
            pret_ckpts = [cfg.init_from_model]
        else:
            pret_ckpts = cfg.init_from_model

        pret_ckpts = [os.path.join(cfg.cwd, 'checkpoints', path) for path in pret_ckpts]

        state_dict_loaded = init_model(model, pret_ckpts, modules_to_keep, logger)
    else:
        state_dict_loaded = {}

    if dist_info['distributed'] and utils.has_batchnorms(model):
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model.to(device)

    # set up optimizer
    # Filtering param groups is better than requires_grad=False, since it does not prevent .lower layer gradients.
    param_groups = prepare_params(model, cfg.opt.lr_wd, cfg.opt.lr, cfg.opt.wd)
    param_lr = {d["name"]: d["lr"] for d in param_groups}

    print("Parameters and training weights:")
    parnum_all = 0
    for n, p in model.named_parameters():
        parnum = sum(pa.numel() for pa in p if pa.requires_grad)
        parnum_all = parnum_all + parnum if p.requires_grad and n in param_lr and param_lr[n] > 0 else parnum_all
        print(
            f"{n:75} {utils.human_format(parnum):8} {param_lr[n] if n in param_lr else 'Frozen'} "
            f"{' (Pret)' if n in state_dict_loaded else ''}")

    print(f"All training weights: {utils.human_format(parnum_all)}")

    optimizer = hydra.utils.instantiate(cfg.opt.optimizer, param_groups)

    # set up learning rate scheduler
    main_scheduler, lr_scheduler = None, None
    if cfg.opt.scheduler is not None and cfg.opt.warmup is not None:
        main_scheduler = hydra.utils.instantiate(
            cfg.opt.scheduler, optimizer, iters_per_epoch=len(dataloader_train),
            world_size=dist_info['world_size'])
        lr_scheduler = hydra.utils.instantiate(
            cfg.opt.warmup, optimizer, main_scheduler,
            iters_per_epoch=len(dataloader_train), world_size=dist_info['world_size'])

    if dist_info['distributed']:
        logger.info('Wrapping model into DDP')
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[dist_info['gpu']], output_device=dist_info['gpu']
        )

    loss_wts = cfg.train.loss_wts
    runner = Runner(model, device, loss_wts=loss_wts)
    metric_tracker = MetricTracker(num_classes)

    # instantiate mixup if required
    mixup_fn = None
    if cfg.train.use_mixup:
        logger.info(f'Using mixup augmentation with mixupbackbone {cfg.train.mixup_backbone} '
                    f'alpha {cfg.train.mixup_alpha} and label smoothing {cfg.train.label_smoothing}')
        mixup_fn = mixup.MixUp(
            alpha=cfg.train.mixup_alpha,
            label_smoothing=cfg.train.label_smoothing,
            num_classes=num_classes)

    # start training
    best_metric_value = 0
    for epoch in range(cfg.train.num_epochs):
        if dist_info['distributed']:
            dataloader_train.sampler.set_epoch(epoch)
        lr = optimizer.param_groups[-1]['lr']
        logger.info(f'Epoch {epoch + 1} of {cfg.train.num_epochs} with lr {lr}')
        metric_tracker.reset()

        # training
        model.train()
        run_one_epoch(runner, optimizer, lr_scheduler, dataloader_train, metric_tracker, is_training=True,
                      grad_clip=cfg.opt.grad_clip, mixup_fn=mixup_fn, mixup_backbone=cfg.train.mixup_backbone)

        # validation
        model.eval()
        with torch.no_grad():
            run_one_epoch(runner, optimizer, lr_scheduler, dataloader_val, metric_tracker, is_training=False)

        if utils.is_main_process():
            # print info
            logger.info(metric_tracker.to_string(is_training=True))
            logger.info(metric_tracker.to_string(is_training=False))

            # store checkpoint
            primary_metric = metric_tracker.get_data(cfg.primary_metric, is_training=False)
            if primary_metric > best_metric_value:
                store_checkpoint(os.path.join(ckpt_path, CKPT_BEST_FNAME), model, optimizer, lr_scheduler, epoch + 1)
                best_metric_value = primary_metric

            if epoch == 0:
                wandb.init(project=cfg.project_name, name=experiment_name)
                wandb.watch(model)

            # wandb log info
            wandb.log({
                **metric_tracker.get_all_data(is_training=True),
                **metric_tracker.get_all_data(is_training=False),
                'lr': lr
                })
    if utils.is_main_process():
        wandb.run.summary[cfg.primary_metric] = best_metric_value


if __name__ == '__main__':
    main()
