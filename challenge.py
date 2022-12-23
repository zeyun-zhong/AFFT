"""
Implementation of functions supporting metrics calculation, late fusion, and challenge file creation.
Most of the code are modified from AVT repo.
"""

import glob
import numpy as np
import os
import os.path as osp
import h5py
import pandas as pd
from scipy.special import softmax
import hydra
import logging
from tqdm import tqdm
import json
import subprocess
from numpyencoder import NumpyEncoder
import bisect
import argparse

from common.utils import topk_recall, topk_accuracy

EGTEA_VERSION = -1
EPIC55_VERSION = 0.1
EPIC100_VERSION = 0.2

LOGITS_DIR = 'logits'
DATASET_EVAL_CFG_KEY = 'dataset_eval'
DATASET_EVAL_CFG_KEY_SUFFIX = ''

PREFIX_H5 = 'test'

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix_h5, type=str', default='test', required=True, choices=['test', 'val'],
                        help='Prefix of h5 file to be selected')
    parser.add_argument('--models', type=str, nargs='+', required=True, help='List of models to be selected')
    parser.add_argument('--weights', type=str, nargs='+', required=True, help='List of weights for selected models')
    args = parser.parse_args()
    return args


def _get_dataset():
    with hydra.initialize(config_path='conf'):
        cfg = hydra.compose(config_name='config.yaml', return_hydra_config=True, overrides=['cwd="."'])

    # set reader fn as an empty dict, as otherwise the code will not work
    # will not affect performance, since only the action to verb / noun mapping matrix saved in dataset object will be used
    setattr(getattr(cfg, DATASET_EVAL_CFG_KEY), 'reader_fn', {})

    if 'test' in PREFIX_H5:
        if not any('test' in path for path in getattr(cfg, DATASET_EVAL_CFG_KEY).annotation_path):
            logging.warning('"epic_kitchens100/test" was not set in config.yaml,'
                            'change annotation path to load test annotations')
            # for test, we only need to change root and annotation path
            annotation_path = ['${dataset.epic_kitchens100.common.annot_dir}/EPIC_100_test_timestamps.pkl']
            setattr(getattr(cfg, DATASET_EVAL_CFG_KEY), 'annotation_path', annotation_path)

    dataset = hydra.utils.instantiate(getattr(cfg, DATASET_EVAL_CFG_KEY), _recursive_=False)
    return dataset


def allkeys(obj, keys=[]):
    """Recursively find all leaf keys in h5. """
    keys = []
    for key in obj.keys():
        if isinstance(obj[key], h5py.Group):
            keys += [f'{key}/{el}' for el in allkeys(obj[key])]
        else:
            keys.append(key)
    return keys


def gen_load_resfiles(resdir):
    resfiles = glob.glob(osp.join(resdir, PREFIX_H5 + '*h5'))
    if len(resfiles) == 0:
        raise ValueError(f'Didnt find any resfiles in {resdir}')
    for resfile in resfiles:
        output_dict = {}
        with h5py.File(resfile, 'r') as fin:
            for key in allkeys(fin):
                try:
                    output_dict[key] = fin[key][()]
                except AttributeError as err:
                    logging.warning('Unable to load %s (%s)', key, err)
        yield output_dict


def compute_accuracy(predictions, labels, classes=None):
    """
    Args:
        predictions: (B, C) logits
        labels: (B, )
        classes: OrderedDict[name (str), cls_id (int)]
    """
    if classes is not None:
        classes = list(classes.values())

    top_1, top_5 = topk_accuracy(predictions, labels, ks=(1, 5))
    mt5r = topk_recall(predictions, labels, k=5, classes=classes)
    return top_1 * 100, top_5 * 100, mt5r * 100


def epic100_unseen_tail_eval(probs, dataset):
    """ Computes metrics for unseen kitchens and tail classes
    :param probs: list of predictions verb, noun and action
    :param dataset: dataset object
    :return:
    """
    # based on https://github.com/fpv-iplab/rulstm/blob/d44612e4c351ff668f149e2f9bc870f1e000f113/RULSTM/main.py#L379
    unseen_participants_ids = pd.read_csv(
        osp.join(dataset.rulstm_annotation_dir, 'validation_unseen_participants_ids.csv'),
        names=['ids'], squeeze=True)

    tail_verbs_ids = pd.read_csv(
        osp.join(dataset.rulstm_annotation_dir, 'validation_tail_verbs_ids.csv'),
        names=['id'], squeeze=True)

    tail_nouns_ids = pd.read_csv(
        osp.join(dataset.rulstm_annotation_dir, 'validation_tail_nouns_ids.csv'),
        names=['id'], squeeze=True)

    tail_actions_ids = pd.read_csv(
        osp.join(dataset.rulstm_annotation_dir, 'validation_tail_actions_ids.csv'),
        names=['id'], squeeze=True)

    unseen_bool_idx = dataset.df.narration_id.isin(unseen_participants_ids).values
    tail_verbs_bool_idx = dataset.df.narration_id.isin(tail_verbs_ids).values
    tail_nouns_bool_idx = dataset.df.narration_id.isin(tail_nouns_ids).values
    tail_actions_bool_idx = dataset.df.narration_id.isin(tail_actions_ids).values

    # For tail
    _, _, vmt5r_tail = compute_accuracy(
        probs[0][tail_verbs_bool_idx], dataset.df.verb_class.values[tail_verbs_bool_idx])
    _, _, nmt5r_tail = compute_accuracy(
        probs[1][tail_nouns_bool_idx], dataset.df.noun_class.values[tail_nouns_bool_idx])
    _, _, amt5r_tail = compute_accuracy(
        probs[2][tail_actions_bool_idx], dataset.df.action_class.values[tail_actions_bool_idx])

    # For unseen
    _, _, vmt5r_unseen = compute_accuracy(
        probs[0][unseen_bool_idx], dataset.df.verb_class.values[unseen_bool_idx])
    _, _, nmt5r_unseen = compute_accuracy(
        probs[1][unseen_bool_idx], dataset.df.noun_class.values[unseen_bool_idx])
    _, _, amt5r_unseen = compute_accuracy(
        probs[2][unseen_bool_idx], dataset.df.action_class.values[unseen_bool_idx])

    res = {
     'vmt5r_tail': vmt5r_tail, 'nmt5r_tail': nmt5r_tail, 'amt5r_tail': amt5r_tail,
     'vmt5r_unseen': vmt5r_unseen, 'nmt5r_unseen': nmt5r_unseen, 'amt5r_unseen': amt5r_unseen
    }

    return res


def compute_accuracies_epic(probs, dataset, compute_manyshot_unseen_tail=False):
    """ Computes top1, top5 and mt5r for verb, noun and action
    :param probs: list of predictions verb, noun and action
    :param dataset: dataset object
    :return:
    """
    assert len(probs) == 3, f'Probs should contain probs for verb, noun and action'
    manyshot_classes = dataset.classes_manyshot

    vtop1, vtop5, vmt5r = compute_accuracy(probs[0], dataset.df.verb_class.values)
    vmt5r_ms, nmt5r_ms, amt5r_ms = float('nan'), float('nan'), float('nan')
    if 'verb' in manyshot_classes and compute_manyshot_unseen_tail:
        _, _, vmt5r_ms = compute_accuracy(probs[0], dataset.df.verb_class.values,
                                          classes=manyshot_classes['verb'])

    ntop1, ntop5, nmt5r = compute_accuracy(probs[1], dataset.df.noun_class.values)
    if 'noun' in manyshot_classes and compute_manyshot_unseen_tail:
        _, _, nmt5r_ms = compute_accuracy(probs[1], dataset.df.noun_class.values,
                                          classes=manyshot_classes['noun'])

    atop1, atop5, am5tr = compute_accuracy(probs[2], dataset.df.action_class.values)
    if 'action' in manyshot_classes and compute_manyshot_unseen_tail:
        _, _, amt5r_ms = compute_accuracy(probs[2], dataset.df.action_class.values,
                                          classes=manyshot_classes['action'])

    res = {'vtop1': vtop1, 'vtop5': vtop5, 'vmt5r': vmt5r, 'vmt5r_ms': vmt5r_ms,
           'ntop1': ntop1, 'ntop5': ntop5, 'nmt5r': nmt5r, 'nmt5r_ms': nmt5r_ms,
           'atop1': atop1, 'atop5': atop5, 'amt5r': am5tr, 'amt5r_ms': amt5r_ms}

    if dataset.version == EPIC100_VERSION and compute_manyshot_unseen_tail:
        res.update(epic100_unseen_tail_eval(probs, dataset))

    return res


def marginalize_verb_noun(res_action, dataset, to_prob=True, compute_manyshot_unseen_tail=False):
    """The logits are first transformed into probabilities if to_prob is true"""
    res_action_probs = softmax(res_action, axis=-1) if to_prob else res_action

    # Marginalize the other dimension, using the mapping matrices store in the dataset obj
    res_verb = np.matmul(res_action_probs, dataset.class_mappings[('verb', 'action')]).numpy()
    res_noun = np.matmul(res_action_probs, dataset.class_mappings[('noun', 'action')]).numpy()

    # compute top1, top5 and mt5r metrics
    accuracies = compute_accuracies_epic([res_verb, res_noun, res_action], dataset, compute_manyshot_unseen_tail)

    # Returning the actual scores for actions instead of the probs.
    # AVT ICCV'21 and Sener et al. ECCV'20 do the same
    scores = [res_verb, res_noun, res_action]
    return accuracies, scores


def get_epic_marginalize_verb_noun(resdir, dataset):
    """Computes scores for verb and noun based on action scores"""
    res = next(gen_load_resfiles(resdir))
    res_action = None
    for key, val in res.items():
        if key.startswith('logits/action'):
            res_action = val

    assert res_action is not None, 'Can not find logits/action in h5.'
    return marginalize_verb_noun(res_action, dataset)


def print_accuracies_epic(metrics: dict, prefix: str = ''):
    print(f"[{prefix}] Accuracies verb/noun/action: "
          f"{metrics['vtop1']:.1f} {metrics['vtop5']:.1f} "
          f"{metrics['ntop1']:.1f} {metrics['ntop5']:.1f} "
          f"{metrics['atop1']:.1f} {metrics['atop5']:.1f} ")
    print(f"[{prefix}] Mean top 5 verb/noun/action: "
          f"{metrics['vmt5r']:.1f} {metrics['nmt5r']:.1f} {metrics['amt5r']:.1f} ")
    print(f"[{prefix}] Mean top 5 many shot verb/noun/action: "
          f"{metrics['vmt5r_ms']:.1f} {metrics['nmt5r_ms']:.1f} {metrics['amt5r_ms']:.1f} ")
    if 'vmt5r_tail' in metrics:
        # assuming the others for tail/unseen will be in there too, since
        # they are all computed at one place for ek100
        print(f"[{prefix}] Mean top 5 tail verb/noun/action: "
              f"{metrics['vmt5r_tail']:.1f} {metrics['nmt5r_tail']:.1f} {metrics['amt5r_tail']:.1f} ")
        print(f"[{prefix}] Mean top 5 unseen verb/noun/action: "
              f"{metrics['vmt5r_unseen']:.1f} {metrics['nmt5r_unseen']:.1f} {metrics['amt5r_unseen']:.1f} ")


def _concat_with_uids(scores, dataset, uid_key):
    # Make a dict with the IDs from the dataset
    # There will be 3 elements in scores -- verb, noun, action
    return [
        dict(zip([str(el) for el in dataset.df[uid_key].values], scores_per_space))
        for scores_per_space in scores
    ]


def _normalize_scores(scores, p):
    """This brings the scores between 0 to 1, and normalizes by """
    res = []
    for scores_per_space in scores:
        res.append({
            uid: val / (np.linalg.norm(val, ord=p, axis=-1) + 0.000001)
            for uid, val in scores_per_space.items()
        })
    return res


def contains_list_or_tuple(test_list):
    """check whether a list contains another list"""
    for element in test_list:
         if isinstance(element, list) or isinstance(element, tuple):
              return True
    return False


def read_all_single_models(resdirs, uid_key='narration_id', normalize_before_combine=None):
    all_scores = []
    for resdir in resdirs:
        accuracies, scores = get_epic_marginalize_verb_noun(resdir, dataset)
        scores = _concat_with_uids(scores, dataset, uid_key)
        print_accuracies_epic(accuracies, prefix=resdir)

        # Normalize if required (AVT does not normalize)
        if normalize_before_combine is not None:
            scores = _normalize_scores(scores, p=normalize_before_combine)

        logging.info(f'Adding scores from {resdir}')
        all_scores.append(scores)
    return all_scores


def get_epic_marginalize_late_fuse(resdirs, weights=1.0, mp_best_weights=[], n_best=5, uid_key='narration_id'):
    """
    Args:
        normalize_before_combine: Set to non-None to normalize the features by that p-norm,
            and then combine. So the weights would have to be defined w.r.t normalized features.
    """
    if not isinstance(resdirs, list):
        resdirs = [resdirs]

    # convert weights to list of lists, make ensemble run quicker if we have to test multiple weights
    if isinstance(weights, float):
        weights = [[weights] * len(resdirs)]
    else:
        if not contains_list_or_tuple(weights) and not isinstance(weights, np.ndarray):  # list of floats
            assert len(weights) == len(resdirs)
            weights = [weights]
        else:  # list of lists
            assert all([len(weight) == len(resdirs) for weight in weights])

    logging.info(f'Loading {len(weights)} weights combinations.')

    all_scores = read_all_single_models(resdirs)

    for weight in tqdm(weights):
        combined = []
        for space_id in range(3):  # verb/noun/action
            scores_for_space = [scores[space_id] for scores in all_scores]
            # Take the union of all the UIDs we have score for
            total_uids = set.union(*[set(el.keys()) for el in scores_for_space])
            logging.info('Combined UIDs: %d. UIDs in the runs %s', len(total_uids),
                            [len(el.keys()) for el in scores_for_space])
            combined_for_space = {}
            for uid in total_uids:
                combined_for_space[uid] = []
                for run_id, scores_for_space_per_run in enumerate(scores_for_space):
                    if uid in scores_for_space_per_run:
                        combined_for_space[uid].append(
                            scores_for_space_per_run[uid] * weight[run_id])
                combined_for_space[uid] = np.sum(np.stack(combined_for_space[uid]),
                                                 axis=0)
            combined.append(combined_for_space)
        # Now to compute accuracies, need to convert back to np arrays from dict.
        # Would only work for parts that are in the dataset
        combined_np = []
        for combined_for_space in combined:
            combined_np.append(
                np.array([
                    combined_for_space[str(uid)]
                    for uid in dataset.df[uid_key].values
                ]))
        accuracies = compute_accuracies_epic(combined_np, dataset)
        print_accuracies_epic(accuracies, prefix=f'combined with {weight}')

        # update best weights list
        metric = accuracies['amt5r']
        if len(mp_best_weights) == 0 or metric > mp_best_weights[0][0]:
            try:
                bisect.insort(mp_best_weights, (metric, weight))
            except:
                print('mp_best_weights: ', mp_best_weights)
                print('metric: ', metric)
                print('weight', weight)
            if len(mp_best_weights) > n_best:
                mp_best_weights.pop(0)
    return accuracies, combined, dataset


def get_struct_outputs_per_dataset(resdirs, weights, uid_key='narration_id'):
    _, combined, dataset = get_epic_marginalize_late_fuse(
        resdirs, weights, uid_key=uid_key)

    results = {}
    action_to_verb_noun = {
        val: key
        for key, val in dataset.verb_noun_to_action.items()
    }

    for uid in tqdm(combined[0].keys(), desc='Computing res'):
        verb_res = {f'{j}': val for j, val in enumerate(combined[0][uid])}
        noun_res = {f'{j}': val for j, val in enumerate(combined[1][uid])}
        top_100_actions = sorted(np.argpartition(combined[2][uid], -100)[-100:],
                                 key=lambda x: -combined[2][uid][x])
        action_res = {
            ','.join((str(el) for el in action_to_verb_noun[j])): combined[2][uid][j]
            for j in top_100_actions
        }
        results[f'{uid}'] = {
            'verb': verb_res,
            'noun': noun_res,
            'action': action_res,
        }
    # Add in all the discarded dfs with uniform distribution
    if dataset.discarded_df is not None:
        for _, row in dataset.discarded_df.iterrows():
            if str(row[uid_key]) in results:
                continue
            results[f'{row[uid_key]}'] = {
                'verb':
                {f'{j}': 0.0
                 for j in range(len(dataset.verb_classes))},
                'noun':
                {f'{j}': 0.0
                 for j in range(len(dataset.noun_classes))},
                'action': {f'0,{j}': 0.0
                           for j in range(100)},
            }
    output_dict = {
        'version': f'{dataset.version}',
        'challenge': dataset.challenge_type,
        'results': results
    }
    return output_dict


def package_results_for_submission_ek100(resdirs, weights, sls=[1, 4, 3]):
    res = get_struct_outputs_per_dataset(resdirs, weights, uid_key='narration_id')
    res['sls_pt'] = sls[0]
    res['sls_tl'] = sls[1]
    res['sls_td'] = sls[2]

    # write it out in the first run's output dir
    output_dir = LOGITS_DIR
    print(f'Saving outputs to {output_dir}')
    os.makedirs(output_dir, exist_ok=True)
    with open(osp.join(output_dir, 'test.json'), 'w') as fout:
        json.dump(res, fout, indent=4, cls=NumpyEncoder)
    subprocess.check_output(f'zip -j {output_dir}/submit.zip {output_dir}/test.json ', shell=True)
    print('Successful!!!')


def generate_test_json(args):
    models = args.models
    weights = args.weights

    if not isinstance(models, list):
        models = [models]
    if not isinstance(weights, list):
        weights = [weights]

    resdirs = [os.path.join(LOGITS_DIR, dir) for dir in models]
    weights = [float(weight) for weight in weights]
    package_results_for_submission_ek100(resdirs, weights)


if __name__ == '__main__':
    args = parse_args()
    dataset = _get_dataset()
    generate_test_json(args)
