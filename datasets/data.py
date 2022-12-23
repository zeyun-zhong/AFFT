import hydra


def get_dataset(dataset_cfg, data_cfg, transforms, logger):
    kwargs = {}
    kwargs['transforms'] = transforms
    kwargs['frame_rate'] = data_cfg.frame_rate
    kwargs['frames_per_clip'] = data_cfg.num_frames
    # Have to call dict() here since relative interpolation somehow doesn't work once I get the subclips object
    kwargs['frame_subclips_options'] = dict(data_cfg.frame_subclips)
    kwargs['sec_subclips_options'] = dict(data_cfg.sec_subclips)
    kwargs['load_seg_labels'] = data_cfg.load_seg_labels
    logger.info('Creating the dataset object...')
    # Not recursive since many of the sub-instantiations would need positional arguments
    _dataset = hydra.utils.instantiate(dataset_cfg, _recursive_=False, **kwargs)
    logger.info(f'Created dataset with {len(_dataset)} elts')
    return _dataset
