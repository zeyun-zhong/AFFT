"""The base video dataset loader, modified from AVT"""
import random

import torch

# used at the beginning of your program
torch.multiprocessing.set_sharing_strategy('file_system')
import torch.nn as nn
from typing import Tuple, Union, Sequence, Dict
import logging
from pathlib import Path
from collections import OrderedDict
import operator
from multiprocessing import Manager
import pandas as pd
import numpy as np
import torchvision
from omegaconf import OmegaConf
import hydra
from hydra.types import TargetConf
import multiprocessing as mp
from itertools import repeat
import time

# This is specific to EPIC kitchens
RULSTM_TSN_FPS = 30.0  # the frame rate the feats were stored by RULSTM

SAMPLE_STRAT_CNTR = 'center_clip'
SAMPLE_STRAT_RAND = 'random_clip'
SAMPLE_STRAT_FIRST = 'first_clip'
SAMPLE_STRAT_LAST = 'last_clip'
FUTURE_PREFIX = 'future'  # to specify future videos


def convert_to_anticipation(df: pd.DataFrame,
                            tau_a: float = 1,
                            tau_o: float = 10,
                            future_clip_ratios: Sequence[float] = (1.0,),
                            drop_style='correct'
                            ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """copied from AVT"""
    if tau_a == -999:
        # No anticipation, just simple recognition
        # still add the orig_start and orig_end, future etc
        # so the future prediction baseline can do the case where not future
        # is predicted.
        # This will ensure the future clip ends up being the same as current
        tau_a = df.loc[:, 'start'] - df.loc[:, 'end']
        tau_o = df.loc[:, 'end'] - df.loc[:, 'start']
    logging.debug(
        'Converting data to anticipation with tau_a=%s and '
        'tau_o=%s.', tau_a, tau_o)
    # Copy over the current start and end times
    df.loc[:, 'orig_start'] = df.start
    df.loc[:, 'orig_end'] = df.end
    # Convert using tau_o and tau_a
    df.loc[:, 'end'] = df.loc[:, 'start'] - tau_a
    df.loc[:, 'start'] = df.loc[:, 'end'] - tau_o
    # Add the future clips
    for i, future_clip_ratio in enumerate(future_clip_ratios):
        if future_clip_ratio == -999:
            # A spl number to use the exact current clip as the future
            df.loc[:, f'{FUTURE_PREFIX}_{i}_start'] = df.loc[:, 'start']
            df.loc[:, f'{FUTURE_PREFIX}_{i}_end'] = df.loc[:, 'end']
        elif future_clip_ratio > -10 and future_clip_ratio < 10:
            eff_tau_a = tau_a * future_clip_ratio
            df.loc[:, f'{FUTURE_PREFIX}_{i}_start'] = (df.loc[:, 'end'] +
                                                       eff_tau_a)
            df.loc[:, f'{FUTURE_PREFIX}_{i}_end'] = (
                    df.loc[:, f'future_{i}_start'] + tau_o)
        else:
            raise ValueError(f'Seems out of bound {future_clip_ratio}')

    # first frame seconds
    f1_sec = 1 / RULSTM_TSN_FPS
    old_df = df
    if drop_style == 'correct':
        # at least 1 frame
        df = df[df.end >= f1_sec]
    elif drop_style == 'full_context_in':
        # All frames should be in
        df = df[df.start >= f1_sec]
    elif drop_style == 'action_banks':
        # Based on their dataset_anticipation:__get_snippet_features()
        df = df[df.end >= 2]
    else:
        raise NotImplementedError(f'Unknown style {drop_style}')
    discarded_df = pd.concat([old_df, df]).drop_duplicates(subset=['uid'],
                                                           keep=False)
    df.reset_index(inplace=True, drop=True)
    return df, discarded_df


class BaseVideoDataset(torch.utils.data.Dataset):
    """modified from AVT"""
    def __init__(self,
                 df,
                 data_dir_train=None,
                 data_dir_test=None,
                 data_dir_extension=None,
                 frames_per_clip: int = 10,
                 frame_rate: float = None,
                 frame_subclips_options: Dict[str, float] = None,
                 sec_subclips_options: Dict[str, float] = None,
                 load_seg_labels: bool = False,
                 reader_fn: TargetConf = {
                     '_target_': 'datasets.reader_fns.DefaultReader'
                     },
                 transforms: Dict[str, torchvision.transforms.Compose] = None,
                 # verb, noun, action
                 label_type: Union[str, Sequence[str]] = 'action',
                 sample_strategy: str = SAMPLE_STRAT_LAST,
                 conv_to_anticipate_fn: TargetConf = None,
                 random_seed: int = 42,
                 verb_classes: dict = {},
                 noun_classes: dict = {},
                 action_classes: dict = {},
                 dummy_label: Union[list, int] = -1,
                 compute_dataset_stats: bool = False,
                 max_els=None
                 ):
        super().__init__()
        manager = Manager()
        self.data_dir_train = data_dir_train
        self.data_dir_test = data_dir_test
        self.data_dir_extension = data_dir_extension
        self.frame_subclips_options = frame_subclips_options
        self.sec_subclips_options = sec_subclips_options
        self.load_seg_labels = load_seg_labels
        self.df = df
        self.max_els = max_els

        # To be consistent with EPIC, add a uid column if not already present
        if 'uid' not in self.df.columns:
            self.df.loc[:, 'uid'] = range(1, len(self.df) + 1)
        # convert df to anticipative df
        self.discarded_df = None
        self.challenge_type = 'action_recognition'
        if conv_to_anticipate_fn is not None:
            self.df, self.discarded_df = hydra.utils.call(conv_to_anticipate_fn, self.df)
            logging.info(f'Discarded {len(self.discarded_df)} elements in anticipate conversion')
            self.challenge_type = 'action_anticipation'
        self.frames_per_clip = frames_per_clip
        self.frame_rate = frame_rate
        self.reader_fn = hydra.utils.instantiate(reader_fn)
        self.transforms = transforms
        self.label_type = label_type
        if OmegaConf.get_type(self.label_type) != list:
            self.label_type = [self.label_type]
        self.verb_classes = manager.dict(verb_classes)
        self.noun_classes = manager.dict(noun_classes)
        self.action_classes = manager.dict(action_classes)
        self.sample_strategy = sample_strategy
        self.random_seed = random_seed
        self.rng = np.random.default_rng(self.random_seed)
        self.dummy_label = dummy_label
        if isinstance(self.dummy_label, list):
            self.dummy_label = manager.list(self.dummy_label)
        # Precompute some commonly useful stats if necessary
        if compute_dataset_stats:
            classes_counts = self._compute_stats_cls_counts()
            logging.debug(f'classes counts: {classes_counts}')
            self.classes_counts = manager.dict(classes_counts)

        # select a subset if required
        self.df_before_subset = self.df
        if self.max_els is not None:
            self.df = self.df.sample(n=self.max_els, replace=False)
            self.df.reset_index(inplace=True, drop=True)

        self.path_df, self.video_fps = {}, {}
        for mod in self.reader_fn.keys():
            self.path_df[mod], self.video_fps[mod] = self.index_paths_and_fps_mp(mod)

    def index_paths_and_fps_mp(self, mod):
        file_paths = self.df["video_path"]
        file_paths = file_paths.unique()
        file_paths = list(zip(file_paths, file_paths))  # For all lmdb datasets.

        path_df = pd.DataFrame(file_paths, columns=["file", "abs_path"])
        path_df = path_df.set_index("file")
        path_df.index = path_df.index.astype(str)
        path_df["abs_path"] = [str(p[0]) for p in path_df.values.astype(str)]

        video_fps = {k: v for k, v in zip(path_df["abs_path"],
                                          map(lambda p: self.reader_fn[mod].get_frame_rate(p), path_df["abs_path"]))}

        return path_df, video_fps

    def _compute_stats_cls_counts(self):
        """
        Compute some stats that are useful, like ratio of classes etc.
        """
        all_classes_counts = {}
        for tname, tclasses in self.classes.items():
            col_name = tname + '_class'
            if col_name not in self.df:
                logging.warning('Didnt find %s column in %s', col_name, self.df)
                continue
            lbls = np.array(self.df.loc[:, col_name].values)
            # not removing the -1 labels, it's a dict so keep all of them.
            classes_counts = {
                cls_id: np.sum(lbls == cls_id)
                for _, cls_id in [('', -1)] + tclasses.items()
                }
            assert sum(classes_counts.values()) == len(self.df)
            all_classes_counts[tname] = classes_counts
        logging.debug('Found %s classes counts', all_classes_counts)
        return all_classes_counts

    @property
    def classes(self) -> OrderedDict:
        return OrderedDict([(tname, operator.attrgetter(tname + '_classes')(self)) for tname in self.label_type])

    @property
    def classes_manyshot(self) -> OrderedDict:
        """This is subset of classes that are labeled as "many shot". These were used in EPIC-55 for computing
        recall numbers. By default, using all the classes."""
        return self.classes

    @property
    def class_mappings(self) -> Dict[Tuple[str, str], torch.FloatTensor]:
        return {}

    def _sample(self, video_path: Path, fps: float, start: float, end: float, df_row: pd.DataFrame,
                frames_per_clip: int,
                frame_rate: float, sample_strategy: str, reader_fn: nn.Module, rng: np.random.Generator):
        """
        Args:
            video_path: The path to read the video from
            fps: What this video's natural FPS is.
            start, end: floats of the start and end point in seconds
        Returns:
            video between start', end'; info of the video
        """
        start = max(start, 0)  # No way can read negative time anyway
        end = max(end, 0)  # No way can read negative time anyway
        req_fps = frame_rate
        if req_fps is None:
            req_fps = fps
        nframes = int(fps * (end - start))
        frames_to_ext = int(round(frames_per_clip * (fps / req_fps)))

        # Find a point in the video and crop out
        if sample_strategy == SAMPLE_STRAT_RAND:
            start_frame = max(nframes - frames_to_ext, 0)
            if start_frame > 0:
                start_frame = rng.integers(start_frame)
        elif sample_strategy == SAMPLE_STRAT_CNTR:
            start_frame = max((nframes - frames_to_ext) // 2, 0)
        elif sample_strategy == SAMPLE_STRAT_LAST:
            start_frame = max(nframes - frames_to_ext, 0)
        elif sample_strategy == SAMPLE_STRAT_FIRST:
            start_frame = 0
        else:
            raise NotImplementedError(f'Unknown {sample_strategy}')

        new_start = start + max(start_frame / fps, 0)
        new_end = start + max((start_frame + frames_to_ext) / fps, 0)
        new_end = max(min(end, new_end), 0)
        # Start from the beginning of the video in case anticipation made it go even further back
        new_start = min(max(new_start, 0), new_end)
        args = [str(video_path), new_start, new_end, fps, df_row]
        kwargs = dict(pts_unit='sec')
        st = time.perf_counter()
        outputs = reader_fn(*args, **kwargs)
        video, _, info, reader_timings = outputs

        all_timings = {"T GetItem.GetVideo.I/O.reader": time.perf_counter() - st}
        all_timings.update(reader_timings)

        if new_start >= new_end:
            video_frame_sec = new_start * torch.ones((video.size(0),))
        else:
            video_frame_sec = torch.linspace(new_start, new_end, video.size(0))
        assert video_frame_sec.size(0) == video.size(0)

        # Subsample the video to the req_fps
        if sample_strategy == SAMPLE_STRAT_LAST:
            # From the back
            frames_to_keep = range(len(video))[::-max(int(round(fps / req_fps)), 1)][::-1]
        elif sample_strategy == SAMPLE_STRAT_RAND:  # TODO: This might be a better random shift strategy.
            # assert fps > req_fps, "Temporal frame supersampling not handled."
            frames_to_keep = range(len(video))[::-max(int(round(fps / req_fps)), 1)][::-1]
            shift = max(int(round(fps / req_fps / 3)), 1)  # we select frame randomly from the 1/3 of the desired time zone
            offset = int(round(random.random() * shift))
            frames_to_keep = [i - offset if i - offset > 0 else i for i in frames_to_keep]
        else:
            # Otherwise, this is fine
            frames_to_keep = range(len(video))[::max(int(round(fps / req_fps)), 1)]

        # Convert video to the required fps
        video_without_fps_subsample = video
        video = video[frames_to_keep]
        video_frame_sec = video_frame_sec[frames_to_keep]
        sampled_frames = torch.LongTensor(frames_to_keep)
        info['video_fps'] = req_fps

        # Pad the video with the last frame, or crop out the extra frames
        # so that it is consistent with the frames_per_clip
        vid_t = video.size(0)
        if video.ndim != 4 or (video.size(0) * video.size(1) * video.size(2) *
                               video.size(3)) == 0:
            # Empty clip if any of the dims are 0, corrupted file likely
            logging.warning(f'Corrupted video: {video_path}. Generating empty clip...')
            video = torch.zeros((frames_per_clip, 100, 100, 3), dtype=torch.uint8)
            video_frame_sec = -torch.ones((frames_per_clip,))
            sampled_frames = torch.range(0, frames_per_clip, dtype=torch.int64)
        elif vid_t < frames_per_clip:
            # Pad the first or last frame
            if sample_strategy == SAMPLE_STRAT_LAST:
                def padding_fn(T, npad):
                    return torch.cat([T[:1]] * npad + [T], dim=0)
            elif sample_strategy == SAMPLE_STRAT_RAND:
                # for fast test with random strategy, we use the same padding as for last strategy
                def padding_fn(T, npad):
                    return torch.cat([T[:1]] * npad + [T], dim=0)
            else:
                # Repeat the last frame
                def padding_fn(T, npad):
                    return torch.cat([T] + [T[-1:]] * npad, dim=0)

            npad = frames_per_clip - vid_t
            logging.debug('Too few frames read, padding with %d frames', npad)
            video = padding_fn(video, npad)
            video_frame_sec = padding_fn(video_frame_sec, npad)
            sampled_frames = padding_fn(sampled_frames, npad)
        if sample_strategy == SAMPLE_STRAT_LAST or sample_strategy == SAMPLE_STRAT_RAND:
            video = video[-frames_per_clip:]
            video_frame_sec = video_frame_sec[-frames_per_clip:]
            sampled_frames = sampled_frames[-frames_per_clip:]
        else:
            video = video[:frames_per_clip]
            video_frame_sec = video_frame_sec[:frames_per_clip]
            sampled_frames = sampled_frames[:frames_per_clip]
        return (video, video_frame_sec, video_without_fps_subsample,
                sampled_frames, info, all_timings)

    def _apply_vid_transform(self, mod, data):
        # apply only permutation and / or zeromasking for features, apply other transforms for normal videos
        if data.nelement() == 0:  # Placeholder
            return data
        if self.transforms[mod]:
            data = self.transforms[mod](data)
        return data

    def get_fps(self, fpath, mod):
        return self.video_fps[mod][self.path_df[mod].loc[str(fpath)]["abs_path"]]

    def get_abs_path(self, fpath: Path, mod):
        """
        Combine the fpath with the first root_dir it exists in.
        """
        return self.path_df[mod].loc[str(fpath)]["abs_path"]

    def _get_video(self, df_row):
        timings = {}
        ast = time.perf_counter()  # all start

        video_paths = {mod: self.get_abs_path(df_row['video_path'], mod) for mod in self.reader_fn.keys()}
        fps = {mod: self.get_fps(df_row['video_path'], mod) for mod in self.reader_fn.keys()}

        return_dict = {}

        outputs_dicts \
            = data_dict, video_frame_sec, data_dict_without_fps_subsample, frames_subsampled, info, sample_timings \
            = {}, {}, {}, {}, {}, {}

        for mod in self.reader_fn.keys():
            outputs = self._sample(video_paths[mod], fps[mod], df_row['start'], df_row['end'], df_row,
                                   self.frames_per_clip,
                                   self.frame_rate,
                                   self.sample_strategy, self.reader_fn[mod], self.rng)

            for d, v in zip(outputs_dicts, outputs):
                d[mod] = v

        sample_timings = {k + " " + sk: sample_timings[k][sk] for k in sample_timings.keys() for sk in
                          sample_timings[k].keys()}
        timings.update(sample_timings)

        timings["T GetItem.GetVideo.I/O"] = time.perf_counter() - ast  # io time

        st = time.perf_counter()  # local start

        data_dict = {mod: self._apply_vid_transform(mod, data) for mod, data in data_dict.items()}
        return_dict["data_dict"] = data_dict

        timings["T GetItem.GetVideo.Transforms"] = time.perf_counter() - st
        st = time.perf_counter()  # local start

        return_dict['video_frame_sec'] = video_frame_sec
        return_dict['video_info'] = info
        return_dict['start'] = df_row['start']
        return_dict['end'] = df_row['end']

        timings["T GetItem.GetVideo"] = time.perf_counter() - ast

        return return_dict, timings

    def _get_subclips(self, video: torch.Tensor, num_frames: int, stride: int):
        """
        Args:
            video (C, T, *): The original read video
            num_frames: Number of frames in each clip
            stride: stride to use when getting clips
        Returns:
            video (num_subclips, C, num_frames, *)
        """
        total_time = video.size(1)
        subclips = []
        # we sample from back, make sure that the last frame is the exact frame 1s before the action start
        for i in range(total_time - num_frames, 0 - num_frames, -stride)[::-1]:
            subclips.append(video[:, i:i + num_frames, ...])
        return torch.stack(subclips)

    def _get_label_from_df_row(self, df_row, tname):
        col_name = tname + '_class'
        if col_name not in df_row:
            lbl = self.dummy_label
        else:
            lbl = df_row[col_name]
        return lbl

    def _get_labels(self, df_row) -> OrderedDict:
        labels = OrderedDict()
        for tname in self.label_type:
            labels[tname] = self._get_label_from_df_row(df_row, tname)
        return labels

    def _get_vidseg_labels(self, df_row, video_frame_sec: torch.Tensor):
        """
        Args:
            video_frame_sec (#clips, T): The time point each frame in the video comes from.
        Returns:
            label for each frame
        """
        this_video_df = self.df_before_subset[self.df_before_subset.video_path == df_row.video_path]
        assert video_frame_sec.ndim == 2
        labels = OrderedDict()
        for tname in self.label_type:
            labels[tname] = -torch.ones_like(video_frame_sec, dtype=torch.long)
        for clip_id in range(video_frame_sec.size(0)):
            for t in range(video_frame_sec[clip_id].size(0)):
                cur_t = video_frame_sec[clip_id][t].tolist()
                matching_rows = this_video_df[(this_video_df.orig_start <= cur_t) & (this_video_df.orig_end >= cur_t)]
                if len(matching_rows) == 0:  # Empty dataframe, nothing labeled at this point
                    continue
                elif len(matching_rows) > 1:
                    # Apparently happens often in EK100, take the label closest to the center
                    closest_row = np.argmin(
                        np.abs(cur_t - np.array(((matching_rows.orig_end - matching_rows.orig_start) / 2.0).tolist())))
                    matching_row = matching_rows.iloc[closest_row]
                else:
                    matching_row = matching_rows.iloc[0]
                for tname in self.label_type:
                    labels[tname][clip_id][t] = self._get_label_from_df_row(matching_row, tname)
        return labels

    def __getitem__(self, idx):
        all_timings = {}
        all_start = time.perf_counter()
        df_row = self.df.loc[idx, :]

        video_dict, get_video_timings = self._get_video(df_row)
        data = video_dict['data_dict']
        all_timings.update(get_video_timings)
        st = time.perf_counter()

        for mod, mod_data in data.items():
            data[mod] = self._get_subclips(data[mod], **self.frame_subclips_options)
            label_idx = self._get_labels(df_row)
            video_dict.update({'idx': idx, 'target': label_idx, 'uid': df_row.uid})
            video_dict['video_frame_sec'][mod] = self._get_subclips(
                video_dict['video_frame_sec'][mod].unsqueeze(0), **self.sec_subclips_options).squeeze(1)

        video_frame_sec = next(iter(video_dict['video_frame_sec'].values()))

        if self.load_seg_labels:
            video_dict.update({'target_subclips': self._get_vidseg_labels(df_row, video_frame_sec)})

        all_timings["T GetItem.SubclipLabels"] = time.perf_counter() - st
        all_timings["T GetItem"] = time.perf_counter() - all_start
        return video_dict, all_timings

    def __len__(self):
        return len(self.df)
