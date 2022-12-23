import time

from models.fusion import *
from models.future_prediction import *
from models.base_model import BaseModel
from datasets.data import get_dataset
from train import get_transform_val
from challenge import *
from models.action_recognition import *
from models.backbone import *


def ordered_feature_list(x_d: Dict[str, Tensor], feats_order: List) -> List[Tensor]:
    """Converts multimodal feature dictionary to a list according to the given order
    used for cmfuser"""
    tensor_list = []
    for i, modk in enumerate(feats_order):
        tensor_list.append(x_d[modk])
    return tensor_list


@hydra.main(config_path='conf', config_name='config')
def debug_model(cfg: DictConfig):
    ckpt_root_dir = '/home/zhong/Documents/projects/AVAA/checkpoints/'
    ckpt_path = [ckpt_root_dir + 'IndividualFuturePrediction_CMFuser_rgb/checkpoint_best.pth',
                 ckpt_root_dir + 'IndividualFuturePrediction_CMFuser_objects/checkpoint_best.pth']
    logger = logging.getLogger(__name__)

    # model configs
    model_cfg = cfg.model
    model_cfg.modal_dims = {"rgb": 1024, "objects": 352}
    model_cfg.common.share_classifiers = False
    model_cfg.common.share_predictors = False
    model_cfg.common.modality_cls = True
    model_cfg.common.fusion_cls = True
    model_cfg.CMFP = {'_target_': 'models.future_prediction.CMFPLate', 'model_cfg': 'null'}

    num_classes = {'action': 3806}
    model = BaseModel(cfg.model, num_classes=num_classes, class_mappings={})

    named_params = list(model.named_parameters())
    named_buffs = list(model.named_buffers())
    model_state = list(model.state_dict())

    modules_to_keep = ['future_predictor.future_predictor', 'future_predictor.dim_encoder',
                       'future_predictor.dim_decoder']

    params_require_grad = [p for p in model.parameters() if p.requires_grad]
    print(1)


@hydra.main(config_path="conf", config_name="config")
def debug_cmfp(cfg: DictConfig):
    cmfp_name = 'early'
    input_len = 10  # 10 seconds if fps = 1
    bs = 64

    feats = {
        'rgb': torch.randn((bs, input_len, 768)).cuda(),
        # 'objects': torch.randn((bs, input_len, 352)).cuda()
    }

    model_cfg = cfg.model
    model_cfg.modal_dims = {"rgb": 768, "objects": 352}
    model_cfg.common.fp_inter_dim = 768
    model_cfg.common_dim = 768
    model_cfg.common.fp_layers = 4

    model_cfg.common.share_classifiers = False  # may to be changed
    model_cfg.common.share_predictors = False  # may to be changed
    model_cfg.common.map_features = False  # may to be changed
    model_cfg.common.modality_cls = True  # may to be changed
    model_cfg.common.fusion_cls = False

    num_classes = {'action': 3806}

    # from models.action_recognition import CMRecognitionEarly
    # model = CMRecognitionEarly(model_cfg, num_classes)
    # model = CMFPEarly(model_cfg, num_classes, extra_cls_rgb=True)
    model = IndividualRecognition(model_cfg, num_classes)

    model.to('cuda')
    out = model(feats)

    print(1)


def debug_video_reading(model, dataset, device, logger):
    model.eval()
    dur_data = 0
    dur_infer = 0
    length = 100
    for idx in range(length):
        start_time = time.time()
        with torch.no_grad():
            data = dataset[idx]
            time1 = time.time()
            dur1 = time1 - start_time
            logger.info(f'fetch data takes {dur1}s')
            video = data['video'].to(device)
            outputs = model(video)
            dur2 = time.time() - time1
            logger.info(f'inference takes {dur2}s')
        dur_data += dur1
        dur_infer += dur2
    logger.info(f'averaged fetch data duration pro sample: {dur_data / length}s')
    logger.info(f'averaged inference duration pro sample: {dur_infer / length}')


def debug_fuser():
    feats_order = ["rgb", "objects"]
    order_feature_func = partial(ordered_feature_list, feats_order=feats_order)
    modal_dims = {'rgb': 1024, 'objects': 1024}

    # fuser = ModalTokenCMFuser(dim=1024, frame_level_token=True, temporal_sequence_length=10, modalities=modal_dims)
    # fuser = TemporalCrossAttentFuser(dim=1024, num_modals=2).cuda()
    fuser = TemporalCMFuser(dim=1024, modalities=modal_dims, frame_level_token=False, temporal_sequence_length=10).cuda()
    feats = {'rgb': torch.randn((64, 10, 1024)).cuda(),
             'objects': torch.randn((64, 10, 1024)).cuda()}
    weights = fuser(feats, order_feature_func)
    print(1)


@hydra.main(config_path="conf", config_name="config")
def debug_recognition(cfg: DictConfig):
    model_cfg = cfg.model
    model_cfg.modal_dims = {"rgb": 768}
    num_classes = {'action': 3806}

    feats = {'rgb': torch.randn((64, 3, 768))}

    model = IndividualRecognition(model_cfg, num_classes)
    y = model(feats)
    print(1)

@hydra.main(config_path="conf", config_name="config")
def debug_dataset(cfg: DictConfig):
    logger = logging.getLogger(__name__)
    cfg.dataset.epic_kitchens100.common.sample_strategy = 'random_clip'
    cfg.dataset.epic_kitchens100.common.reader_fn = {'_target_': 'datasets.reader_fns.EpicRULSTMFeatsReader',
                                                     'lmdb_path': ['${dataset.epic_kitchens100.common.rulstm_feats_dir}/rgb/']}
    transform_val = get_transform_val(cfg)
    dataset_test = get_dataset(getattr(cfg, 'dataset_eval'), cfg.data_eval, transform_val, logger)
    for i in tqdm(range(9638)):
        data = dataset_test[i]

    print(1)


def contains_list(test_list):
    for element in test_list:
         if isinstance(element, list):
              return True
    return False


def debug():
    w_tsn_10s = np.arange(0, 1, 0.25)
    w_tsn_14s = np.arange(0, 1, 0.25)
    w_tsn_18s = np.arange(0, 1, 0.25)
    w_swin_4h_8s = np.arange(0.25, 1, 0.25)
    w_swin_4h_14s = np.arange(0.25, 1, 0.25)
    w_swin_4h_16s = np.arange(0.75, 1.25, 0.25)  # important
    w_swin_4h_18s = np.arange(0.25, 1, 0.25)
    w_swin_8h_10s = np.arange(0.5, 1, 0.25)  # important
    w_swin_8h_14s = np.arange(0.5, 1, 0.25)  # important
    w_swin_8h_16s = np.arange(0.25, 1, 0.25)
    w_swin_8h_18s = np.arange(0.25, 1, 0.25)

    # weights = [w for i in range(len(ex))]
    weights = [w_tsn_10s, tsn_14s, tsn_18s, w_swin_4h_8s, w_swin_4h_14s, w_swin_4h_16s,
               w_swin_4h_18s, w_swin_8h_10s, w_swin_8h_14s, w_swin_8h_16s, w_swin_8h_18s]
    # weights = [w_tsn_10s, w_tsn_14s, w_tsn_18s]
    weights_combinations = list(itertools.product(*weights))


def debug_crossentropy_with_ignore_index():
    func = nn.CrossEntropyLoss(ignore_index=-1)
    num_class = 5
    target = torch.tensor([-1, 1, 3])
    #target1 = convert_to_one_hot(target, num_class, label_smooth=0.0)
    logits = torch.randn(3, num_class, requires_grad=True)
    target1 = torch.tensor([1, 3])
    logits1 = logits[1:]

    loss = func(logits, target)
    loss1 = func(logits1, target1)
    print(loss)
    print(loss1)


def debug_mixup_simple():
    B, num_classes = 5, {'action': 6}
    feature_dict = {
        'rgb': torch.randn((B, 5, 3)),
        'objects': torch.randn((B, 5, 3))
    }
    target = {'action': torch.tensor([0, 0, 0, 1, 1])}
    target_subclips = {'action': torch.tensor([[1, 1, 0, -1, 1],
                                                [0, 1, 1, 0, 0],
                                                [1, -1, 0, 1, 1],
                                                [1, 0, 0, 1, 1],
                                                [1, -1, 1, 1, 0]])}
    from common.mixup import MixUp
    op = MixUp(label_smoothing=0.1, num_classes=num_classes)
    x_out, labels_out, labels_subclips_out, labels_subclips_ignore_index = op(feature_dict, target, target_subclips)

    past_logits = torch.randn((B, 5, 6))
    logits = torch.randn((B, 6))

    from common.runner import MultiDimCrossEntropy
    loss_func = MultiDimCrossEntropy()

    loss = loss_func(logits, target['action'])
    past_loss = loss_func(past_logits, labels_subclips_out['action'], one_hot=True, ignore_index=labels_subclips_ignore_index['action'])

    labels = labels_out['action']
    _top_max_k_vals, top_max_k_inds = torch.topk(
        labels, 2, dim=1, largest=True, sorted=True
    )
    idx_top1 = torch.arange(labels.shape[0]), top_max_k_inds[:, 0]
    idx_top2 = torch.arange(labels.shape[0]), top_max_k_inds[:, 1]
    preds = logits.detach()
    preds[idx_top1] += preds[idx_top2]
    preds[idx_top2] = 0.0
    labels = top_max_k_inds[:, 0]

    print(1)


def debug_mixup():
    B = 8
    num_classes = 5
    feature_dict = {
        'rgb': torch.randn((B, 10, 3, 1, 224, 224)),
        'objects': torch.randn((B, 10, 352, 1, 1, 1))
    }
    target = {'action': torch.randint(0, num_classes, (B,))}
    target_subclip = {'action': torch.randint(0, num_classes, (B, 10, 1))}

    from common.mixup import MixUp

    op = MixUp(label_smoothing=0.1, num_classes=num_classes)
    a, b, c = op(feature_dict, target, target_subclip)
    print(1)


def debug_backbone():
    # model = MViTModel()
    # ckpt = torch.load('checkpoints/TIMM/MViTv2_S_in1k.pyth')
    # missing_keys, unexp_keys = model.model.load_state_dict(ckpt['model_state'], strict=False)
    model = TIMMModel(model_type='beit_base_patch16_224_in22k')
    ckpt = torch.load('checkpoints/TIMM/beit_base_patch16_224_pt22k_ft22k.pth')
    missing_keys, unexp_keys = model.model.load_state_dict(ckpt['model'], strict=False)
    print(1)


@hydra.main(config_path="conf", config_name="config")
def debug_future_embed_prediction(cfg):
    model_cfg = cfg.model
    model_cfg.modal_dims = {"rgb": 768}
    model_cfg.common.fp_inter_dim = 768
    model_cfg.common_dim = 768

    num_classes = {'action': 3806}

    from models.future_embed_prediction import FutureEmbedPrediction
    model = FutureEmbedPrediction(model_cfg, num_classes, dim=2048)
    model.to('cuda')

    input_len = 10  # 10 seconds if fps = 1
    bs = 64

    feats = {
        'rgb': torch.randn((bs, input_len, 768)).cuda(),
        # 'objects': torch.randn((bs, input_len, 352)).cuda()
    }

    out = model(feats)
    print(1)


def tmp():
    input_channel = 1
    output_channel = 128

    conv_layer = nn.Conv2d(input_channel, output_channel, kernel_size=(1, 1))
    linear_layer = nn.Linear(input_channel, output_channel)

    input_linear = torch.randn(1, 7, 7, input_channel)
    input_conv = torch.randn(1, input_channel, 7, 7)

    output_conv = conv_layer(input_conv)
    output_linear = linear_layer(input_linear)

    print(1)


if __name__ == '__main__':
    tmp()
    # debug()
    # debug_mixup_simple()
    # debug_crossentropy_with_ignore_index()
    # debug_mixup()
    # debug_recognition()
    # debug_dataset()
    # debug_cmfp()
    # debug_model()
    # debug_fuser()
    # debug_backbone()
    # debug_future_embed_prediction()
    # causal_mask = generate_square_subsequent_mask(5)
    # causal_modality_mask = causal_mask.repeat(2, 2)
    # print(causal_modality_mask)
