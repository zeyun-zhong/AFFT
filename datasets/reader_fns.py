"""Implementation of reader functions, modified from AVT"""

import logging
from pathlib import Path
from typing import Union, List
import torch
import torch.nn as nn
import torchvision
from omegaconf import OmegaConf
import lmdb
import numpy as np
import pandas as pd
import time

from common.utils import get_video_info


# An abstract class to keep track of all reader type classes
class Reader(nn.Module):
    pass


class DefaultReader(Reader):
    def forward(self, video_path, start, end, fps, df_row, **kwargs):
        del df_row, fps  # Not needed here
        torchvision.set_video_backend('pyav')
        st = time.perf_counter()
        video_info = torchvision.io.read_video(video_path, start, end, **kwargs)
        timings = {"T GetItem.GetVideo.I/O.reader.pyav": time.perf_counter() - st}

        # DEBUG see what is breaking
        logging.debug('Read %s from %s', video_info[0].shape, video_path.split('/')[-1])
        return (*video_info, timings)

    @staticmethod
    def get_frame_rate(video_path: Path) -> float:
        return get_video_info(video_path, ['fps'])['fps']


class EpicRULSTMFeatsReader(Reader):
    def __init__(self,
                 lmdb_path: Union[Path, List[Path]] = None,
                 warn_if_using_closeby_frame: bool = True):
        """
        Args:
            feats_lmdb_path: LMDB path for RULSTM features. Must be
                specified if using rulstm_tsn_feat input_type. Could be a
                list, in which case it will concat all those features together.
        """
        super().__init__()
        if OmegaConf.get_type(lmdb_path) != list:
            lmdb_path = [lmdb_path]
        self.lmdb_path = lmdb_path
        self.lmdb_envs = [lmdb.open(el, readonly=True, lock=False) for el in lmdb_path]
        self.warn_if_using_closeby_frame = warn_if_using_closeby_frame

    def forward(self, *args, **kwargs):
        return self._read_rulstm_features(*args, **kwargs)

    @staticmethod
    def get_frame_rate(video_path: Path) -> float:
        del video_path
        return 30.0

    def read_representations(self, frames, env, frame_format):
        """Reads a set of representations, given their frame names and an LMDB environment.
            From https://github.com/fpv-iplab/rulstm/blob/96e38666fad7feafebbeeae94952dba24771e512/RULSTM/dataset.py#L10
        """
        features = []
        # for each frame
        for frame_id in frames:
            # read the current frame
            with env.begin() as e:
                # Need to search for a frame that has features stored, the exact frame may not have.
                # To avoid looking at the future when training/testing, (important for anticipation),
                # look only for previous to current position.
                dd = None
                search_radius = 0
                for search_radius in range(10):
                    dd = e.get(frame_format.format(frame_id - search_radius).strip().encode('utf-8'))
                    if dd is not None:
                        break
                if dd is not None and search_radius > 0:
                    if self.warn_if_using_closeby_frame:
                        logging.warning('Missing %s, but used %d instead', frame_format.format(frame_id),
                                        frame_id - search_radius)
            if dd is None:
                logging.error('Missing %s, Only specific frames are stored in lmdb :(', frame_format.format(frame_id))
                features.append(None)
            else:
                # convert to numpy array
                data = np.frombuffer(dd, 'float32')
                # append to list
                features.append(data)
        # For any frames we didn't find a feature, use a series of 0s
        features_not_none = [el for el in features if el is not None]
        assert len(features_not_none) > 0, (f'No features found in {frame_format} - {frames}')
        feature_not_none = features_not_none[0]  # any
        features = [np.zeros_like(feature_not_none) if el is None else el for el in features]
        # convert list to numpy array
        features = np.array(features)
        # Add singleton dimensions to make it look like a video, so
        # rest of the code just works
        features = features[:, np.newaxis, np.newaxis, :]
        # Make it torch Tensor to be consistent
        features = torch.as_tensor(features)
        return features

    def _read_rulstm_features(self, video_path: Path, start_sec: float, end_sec: float, fps: float,
                              df_row: pd.DataFrame, pts_unit='sec'):
        del pts_unit  # Not supported here
        # Read every single frame between the start and end, the base_video_dataset code will deal with how to sample into 4fps
        # (i.e. 0.25s steps), Rather than first computing the timestamps, just compute the
        # frame ID of the start and end, and do a arange .. that avoids any repeated frames due to quantization/floor
        time_stamps = None
        timings = {}
        start_frame = np.floor(start_sec * fps)
        end_frame = np.floor(end_sec * fps)
        frames = np.arange(end_frame, start_frame, -1).astype(int)[::-1]
        # If the frames go below 1, replace them with the lowest time pt
        assert frames.max() >= 1, (f'The dataset shouldnt have cases otherwise. {video_path} {start_sec} {end_sec} '
                                   f'{df_row} {frames} {time_stamps}')
        frames[frames < 1] = frames[frames >= 1].min()
        # Get the features
        all_feats = []
        for i, lmdb_env in enumerate(self.lmdb_envs):
            video_name = Path(video_path).stem
            lmdb_path = self.lmdb_path[i]
            st = time.perf_counter()
            if 'audio' in lmdb_path or 'poses' in lmdb_path:
                frames_new = self._convert_to_orig_video_fps(video_name, fps, frames)
                all_feats.append(self.read_representations(frames_new, lmdb_env, video_name + '_frame_{:010d}.jpg'))
            else:
                all_feats.append(self.read_representations(frames, lmdb_env, video_name + '_frame_{:010d}.jpg'))
            timings = {"T GetItem.GetVideo.I/O.reader.lmdb_read": time.perf_counter() - st}
        final_feat = torch.cat(all_feats, dim=-1)
        # Must return rgb, audio, info; so padding with empty dicts for those
        return final_feat, {}, {}, timings

    def _convert_to_orig_video_fps(self, video_name, fps, frames):
        """Convert the frames in rulstm fps to the frames in orig video fps.
        This is used for features (e.g. audio) which were extracted based on videos."""
        orig_fps = self._get_orig_video_fps(video_name)
        frames_new = frames / fps * orig_fps
        frames_new = np.rint(frames_new).astype(int)
        return frames_new

    @staticmethod
    def _get_orig_video_fps(video_name):
        length = len(video_name.split('_')[-1])
        if length == 3:  # epic 100
            return 50.0
        elif length == 2:  # epic 55
            return 59.94005994005994
        else:
            raise ValueError(f'Unkown video name format: {video_name}')
