"""The Epic Kitchens dataset loaders, this class also supports EGTEA Gaze+ dataset"""

from typing import List, Dict, Sequence, Tuple, Union
from datetime import datetime, date
from collections import OrderedDict
import pickle as pkl
import csv
import logging
from pathlib import Path
import pandas as pd
import torch

from .base_video_dataset import BaseVideoDataset, RULSTM_TSN_FPS

EGTEA_VERSION = -1  # This class also supports EGTEA Gaze+
EPIC55_VERSION = 0.1
EPIC100_VERSION = 0.2


class EPICKitchens(BaseVideoDataset):
    """EPICKitchens and EGTEA dataloader."""

    def __init__(self,
                 annotation_path: Sequence[Path],
                 action_labels_fpath: Path = None,
                 annotation_dir: Path = None,
                 rulstm_annotation_dir: Path = None,
                 version: float = EPIC55_VERSION,
                 **other_kwargs,
                 ):
        """
        Args:
            label_type (str): The type of label to return
            action_labels_fpath (Path): Path to map the verb and noun labels to
                actions. It was used in the anticipation paper, that defines
                a set of actions and train for action prediction, as opposed
                to verb and noun prediction.
            annotation_dir: Where all the other annotations are typically stored
        """
        self.version = version
        df = pd.concat([self._load_df(el) for el in annotation_path])
        df.reset_index(inplace=True, drop=True)  # to combine all of them
        self.annotation_dir = Path(annotation_dir)
        self.rulstm_annotation_dir = rulstm_annotation_dir

        # Load verb and noun classes
        epic_postfix = ''
        if self.version == EPIC100_VERSION:
            epic_postfix = '_100'
        if self.version != EGTEA_VERSION:
            verb_classes = self._load_class_names(self.annotation_dir / f'EPIC{epic_postfix}_verb_classes.csv')
            noun_classes = self._load_class_names(self.annotation_dir / f'EPIC{epic_postfix}_noun_classes.csv')
        else:
            verb_classes, noun_classes = [], []

        # Create action classes
        if action_labels_fpath is not None:
            load_action_fn = self._load_action_classes
            if self.version == EGTEA_VERSION:
                load_action_fn = self._load_action_classes_egtea
            action_classes, verb_noun_to_action = load_action_fn(action_labels_fpath)
        else:
            logging.warning('Action labels were not provided. Generating actions ...')
            action_classes, verb_noun_to_action = self._gen_all_actions(verb_classes, noun_classes)

        # Add the action classes to the data frame
        if 'action_class' not in df.columns and {'noun_class', 'verb_class'}.issubset(df.columns):
            df.loc[:, 'action_class'] = df.loc[:, ('verb_class', 'noun_class')].apply(
                lambda row: (verb_noun_to_action[(row.at['verb_class'], row.at['noun_class'])]
                             if (row.at['verb_class'], row.at['noun_class']) in verb_noun_to_action else -1), axis=1)
        elif 'action_class' not in df.columns:
            df.loc[:, 'action_class'] = -1
            df.loc[:, 'verb_class'] = -1
            df.loc[:, 'noun_class'] = -1
        num_undefined_actions = len(df[df['action_class'] == -1].index)
        if num_undefined_actions > 0:
            logging.error(f'Did not found valid action label for {num_undefined_actions}/{len(df)} samples!')

        other_kwargs['verb_classes'] = verb_classes
        other_kwargs['noun_classes'] = noun_classes
        other_kwargs['action_classes'] = action_classes

        super().__init__(df, **other_kwargs)
        self.verb_noun_to_action = verb_noun_to_action
        logging.info(f'Created EPIC {self.version} dataset with {len(self)} samples')

    @property
    def class_mappings(self) -> Dict[Tuple[str, str], torch.FloatTensor]:
        num_verbs = len(self.verb_classes)
        if num_verbs == 0:
            num_verbs = len(set([el[0] for el, _ in self.verb_noun_to_action.items()]))
        num_nouns = len(self.noun_classes)
        if num_nouns == 0:
            num_nouns = len(set([el[1] for el, _ in self.verb_noun_to_action.items()]))
        num_actions = len(self.action_classes)
        if num_actions == 0:
            num_actions = len(set([el for _, el in self.verb_noun_to_action.items()]))
        verb_in_action = torch.zeros((num_actions, num_verbs), dtype=torch.float)
        noun_in_action = torch.zeros((num_actions, num_nouns), dtype=torch.float)
        for (verb, noun), action in self.verb_noun_to_action.items():
            verb_in_action[action, verb] = 1.0
            noun_in_action[action, noun] = 1.0
        return {
            ('verb', 'action'): verb_in_action,
            ('noun', 'action'): noun_in_action
            }

    @property
    def classes_manyshot(self) -> OrderedDict:
        """
        In EPIC-55, the recall computation was done for "many shot" classes,
        and not for all classes. So, for that version read the class names as
        provided by RULSTM."""
        if self.version != EPIC55_VERSION:
            return super().classes_manyshot
        # read the list of many shot verbs
        many_shot_verbs = {
            el['verb']: el['verb_class']
            for el in pd.read_csv(self.annotation_dir / 'EPIC_many_shot_verbs.csv').to_dict('records')
            }
        # read the list of many shot nouns
        many_shot_nouns = {
            el['noun']: el['noun_class']
            for el in pd.read_csv(self.annotation_dir / 'EPIC_many_shot_nouns.csv').to_dict('records')
            }
        # create the list of many shot actions
        # an action is "many shot" if at least one between the related verb and noun are many shot
        many_shot_actions = {}
        action_names = {val: key for key, val in self.action_classes.items()}
        for (verb_id, noun_id), action_id in self.verb_noun_to_action.items():
            if (verb_id in many_shot_verbs.values()) or (noun_id in many_shot_nouns.values()):
                many_shot_actions[action_names[action_id]] = action_id
        return {
            'verb':   many_shot_verbs,
            'noun':   many_shot_nouns,
            'action': many_shot_actions,
            }

    def _load_class_names(self, annot_path: Path):
        res = {}
        with open(annot_path, 'r') as fin:
            reader = csv.DictReader(fin, delimiter=',')
            for lno, line in enumerate(reader):
                res[line['class_key' if self.version ==
                                        EPIC55_VERSION else 'key']] = lno
        return res

    @staticmethod
    def _load_action_classes(action_labels_fpath: Path) -> Tuple[Dict[str, int], Dict[Tuple[int, int], int]]:
        """
        Given a CSV file with the actions (as from RULSTM paper), construct the set of actions and mapping from verb/noun to action
        Args:
            action_labels_fpath: path to the file
        Returns:
            class_names: Dict of action class names
            verb_noun_to_action: Mapping from verb/noun to action IDs
        """
        class_names = {}
        verb_noun_to_action = {}
        with open(action_labels_fpath, 'r') as fin:
            reader = csv.DictReader(fin, delimiter=',')
            for lno, line in enumerate(reader):
                class_names[line['action']] = lno
                verb_noun_to_action[(int(line['verb']), int(line['noun']))] = int(line['id'])
        return class_names, verb_noun_to_action

    @staticmethod
    def _load_action_classes_egtea(action_labels_fpath: Path) -> Tuple[Dict[str, int], Dict[Tuple[int, int], int]]:
        """
        Given a CSV file with the actions (as from RULSTM paper), construct the set of actions and mapping from verb/noun to action
        Args:
            action_labels_fpath: path to the file
        Returns:
            class_names: Dict of action class names
            verb_noun_to_action: Mapping from verb/noun to action IDs
        """
        class_names = {}
        verb_noun_to_action = {}
        with open(action_labels_fpath, 'r') as fin:
            reader = csv.DictReader(fin, delimiter=',',
                                    # Assuming the order is verb/noun
                                    # TODO check if that is correct
                                    fieldnames=['id', 'verb_noun', 'action'])
            for lno, line in enumerate(reader):
                class_names[line['action']] = lno
                verb, noun = [int(el) for el in line['verb_noun'].split('_')]
                verb_noun_to_action[(verb, noun)] = int(line['id'])
        return class_names, verb_noun_to_action

    @staticmethod
    def _gen_all_actions(verb_classes: List[str], noun_classes: List[str]) -> Tuple[
        Dict[str, int], Dict[Tuple[int, int], int]]:
        """
        Given all possible verbs and nouns, construct all possible actions
        Args:
            verb_classes: All verbs
            noun_classes: All nouns
        Returns:
            class_names: list of action class names
            verb_noun_to_action: Mapping from verb/noun to action IDs
        """
        class_names = {}
        verb_noun_to_action = {}
        action_id = 0
        for verb_id, verb_cls in enumerate(verb_classes):
            for noun_id, noun_cls in enumerate(noun_classes):
                class_names[f'{verb_cls}:{noun_cls}'] = action_id
                verb_noun_to_action[(verb_id, noun_id)] = action_id
                action_id += 1
        return class_names, verb_noun_to_action

    def _init_df_orig(self, annotation_path):
        """Loading the original EPIC Kitchens annotations"""

        def timestr_to_sec(s, fmt='%H:%M:%S.%f'):
            # Convert timestr to seconds
            timeobj = datetime.strptime(s, fmt).time()
            td = datetime.combine(date.min, timeobj) - datetime.min
            return td.total_seconds()

        # Load the DF from annot path
        logging.info(f'Loading original EPIC pkl annotations {annotation_path}')
        with open(annotation_path, 'rb') as fin:
            df = pkl.load(fin)
        # Make a copy of the UID column, since that will be needed to gen output files
        df.reset_index(drop=False, inplace=True)

        # parse timestamps from the video
        df.loc[:, 'start'] = df.start_timestamp.apply(timestr_to_sec)
        df.loc[:, 'end'] = df.stop_timestamp.apply(timestr_to_sec)

        # original annotations have text in weird format - fix that
        if 'noun' in df.columns:
            df.loc[:, 'noun'] = df.loc[:, 'noun'].apply(lambda s: ' '.join(s.replace(':', ' ').split(sep=' ')[::-1]))
        if 'verb' in df.columns:
            df.loc[:, 'verb'] = df.loc[:, 'verb'].apply(lambda s: ' '.join(s.replace('-', ' ').split(sep=' ')))
        df = self._init_df_gen_vidpath(df)
        df.reset_index(inplace=True, drop=True)
        return df

    def _init_df_gen_vidpath(self, df):
        # generate video_path
        if self.version == EGTEA_VERSION:
            df.loc[:, 'video_path'] = df.apply(lambda x: Path(x.video_id + '.mp4'), axis=1)
        else:  # For the EPIC datasets
            df.loc[:, 'video_path'] = df.apply(lambda x: (Path(x.participant_id) / Path(x.video_id + '.MP4')), axis=1)
        return df

    def _init_df_rulstm(self, annotation_path):
        logging.info('Loading RULSTM EPIC csv annotations %s', annotation_path)
        df = pd.read_csv(annotation_path,
                         names=['uid', 'video_id', 'start_frame_30fps', 'end_frame_30fps', 'verb_class', 'noun_class',
                                'action_class'],
                         index_col=0,
                         skipinitialspace=True,
                         dtype={'uid':        str, 'video_id': str, 'start_frame_30fps': int, 'end_frame_30fps': int,
                                'verb_class': int, 'noun_class': int, 'action_class': int})
        # Make a copy of the UID column, since that will be needed to gen output files
        df.reset_index(drop=False, inplace=True)
        # Convert the frame number to start and end
        df.loc[:, 'start'] = df.loc[:, 'start_frame_30fps'].apply(lambda x: x / RULSTM_TSN_FPS)
        df.loc[:, 'end'] = df.loc[:, 'end_frame_30fps'].apply(lambda x: x / RULSTM_TSN_FPS)
        # Participant ID from video_id
        df.loc[:, 'participant_id'] = df.loc[:, 'video_id'].apply(lambda x: x.split('_')[0])
        df = self._init_df_gen_vidpath(df)
        df.reset_index(inplace=True, drop=True)
        return df

    def _load_df(self, annotation_path):
        if annotation_path.endswith('.pkl'):
            return self._init_df_orig(annotation_path)
        elif annotation_path.endswith('.csv'):
            # Else, it must be the RULSTM annotations (fps 30)
            return self._init_df_rulstm(annotation_path)
        else:
            raise NotImplementedError(annotation_path)
