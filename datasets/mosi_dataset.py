"""Implements dataloaders for AFFECT data."""
import os
import sys
from typing import *
import pickle
import h5py
import numpy as np
import torch
import pytorch_lightning as pl
from omegaconf import DictConfig
from tokenizers.implementations import BertWordPieceTokenizer

from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from utils.projection import Projection

sys.path.append(os.getcwd())
np.seterr(divide='ignore', invalid='ignore')


def drop_entry(dataset):
    """Drop entries where there's no text in the data."""
    drop = []
    for ind, k in enumerate(dataset["text"]):
        if (k == '') or (k is None):
            drop.append(ind)
    for ind, k in enumerate(dataset["vision"]):
        if k.sum() == 0:
            if ind not in drop:
                drop.append(ind)
    for ind, k in enumerate(dataset["audio"]):
        if k.sum() == 0:
            if ind not in drop:
                drop.append(ind)

    for modality in list(dataset.keys()):
        dataset[modality] = np.delete(dataset[modality], drop, 0)
    return dataset


def z_norm(dataset, max_seq_len=50):
    """Normalize data in the dataset."""
    processed = {}
    text = dataset['text'][:, :max_seq_len, :]
    vision = dataset['vision'][:, :max_seq_len, :]
    audio = dataset['audio'][:, :max_seq_len, :]
    for ind in range(dataset["text"].shape[0]):
        vision[ind] = np.nan_to_num(
            (vision[ind] - vision[ind].mean(0, keepdims=True)) / (np.std(vision[ind], axis=0, keepdims=True)))
        audio[ind] = np.nan_to_num(
            (audio[ind] - audio[ind].mean(0, keepdims=True)) / (np.std(audio[ind], axis=0, keepdims=True)))
        text[ind] = np.nan_to_num(
            (text[ind] - text[ind].mean(0, keepdims=True)) / (np.std(text[ind], axis=0, keepdims=True)))

    processed['vision'] = vision
    processed['audio'] = audio
    processed['text'] = text
    processed['labels'] = dataset['labels']
    return processed


def get_rawtext(path, data_kind, vids):
    """Get raw text, video data from hdf5 file."""
    if data_kind == 'hdf5':
        f = h5py.File(path, 'r')
    else:
        with open(path, 'rb') as f_r:
            f = pickle.load(f_r)
    text_data = []
    new_vids = []

    for vid in vids:
        text = []
        # If data IDs are NOT the same as the raw ids
        # add some code to match them here, eg. from vanvan_10 to vanvan[10]
        # (id, seg) = re.match(r'([-\w]*)_(\w+)', vid).groups()
        # vid_id = '{}[{}]'.format(id, seg)
        vid_id = int(vid[0]) if type(vid) == np.ndarray else vid
        try:
            if data_kind == 'hdf5':
                for word in f['words'][vid_id]['features']:
                    if word[0] != b'sp':
                        text.append(word[0].decode('utf-8'))
                text_data.append(' '.join(text))
                new_vids.append(vid_id)
            else:
                for word in f[vid_id]:
                    if word != 'sp':
                        text.append(word)
                text_data.append(' '.join(text))
                new_vids.append(vid_id)
        except:
            print("missing", vid, vid_id)
    return text_data, new_vids


class CMUMosiDataModule(pl.LightningDataModule):

    def __init__(self, data_path: str, batch_size: int, num_workers: int, vocab_cfg: DictConfig, train_cfg: DictConfig,
                 proj_cfg: DictConfig, **kwargs):
        super().__init__(**kwargs)
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.vocab_cfg = vocab_cfg
        self.train_cfg = train_cfg
        self.projection = Projection(vocab_cfg.vocab_path, proj_cfg.feature_size, proj_cfg.window_size)
        self.tokenizer = BertWordPieceTokenizer(**vocab_cfg.tokenizer)

    def setup(self, stage: str = None):
        with open(self.data_path, "rb") as f:
            alldata = pickle.load(f)

        processed_dataset = {'train': {}, 'test': {}, 'valid': {}}
        alldata['train'] = drop_entry(alldata['train'])
        alldata['valid'] = drop_entry(alldata['valid'])
        alldata['test'] = drop_entry(alldata['test'])

        # process = eval("_process_2") if max_pad else eval("_process_1")

        for dataset in alldata:
            processed_dataset[dataset] = alldata[dataset]

        self.train_set = Affectdataset(processed_dataset['train'], False, task=self.train_cfg.task, max_pad=False,
                                        max_pad_num=50, data_type='mosi', z_norm=self.train_cfg.z_norm,
                                       tokenizer=self.tokenizer, projection=self.projection,
                                       max_seq_len=self.train_cfg.max_seq_len)

        self.test_set = Affectdataset(processed_dataset['train'], False, task=self.train_cfg.task, max_pad=False,
                                           max_pad_num=50, data_type='mosi', z_norm=self.train_cfg.z_norm,
                                       tokenizer=self.tokenizer, projection=self.projection,
                                       max_seq_len=self.train_cfg.max_seq_len)

        self.valid_set = Affectdataset(processed_dataset['train'], False, task=self.train_cfg.task, max_pad=False,
                                           max_pad_num=50, data_type='mosi', z_norm=self.train_cfg.z_norm,
                                       tokenizer=self.tokenizer, projection=self.projection,
                                       max_seq_len=self.train_cfg.max_seq_len)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_set, self.batch_size, shuffle=True,
                          num_workers=self.num_workers, persistent_workers=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.valid_set, self.batch_size, shuffle=False,
                          num_workers=self.num_workers, persistent_workers=True)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_set, self.batch_size, shuffle=False,
                          num_workers=self.num_workers, persistent_workers=True)


class Affectdataset(Dataset):
    """Implements Affect data as a torch dataset."""

    def __init__(self, data: Dict, flatten_time_series: bool, tokenizer, projection, max_seq_len, aligned: bool = True,
                 task: str = None, max_pad=False, max_pad_num=50, data_type='mosi', z_norm=False) -> None:
        """Instantiate AffectDataset

        Args:
            data (Dict): Data dictionary
            flatten_time_series (bool): Whether to flatten time series or not
            aligned (bool, optional): Whether to align data or not across modalities. Defaults to True.
            task (str, optional): What task to load. Defaults to None.
            max_pad (bool, optional): Whether to pad data to max_pad_num or not. Defaults to False.
            max_pad_num (int, optional): Maximum padding number. Defaults to 50.
            data_type (str, optional): What data to load. Defaults to 'mosi'.
            z_norm (bool, optional): Whether to normalize data along the z-axis. Defaults to False.
        """
        self.max_seq_len = max_seq_len
        self.projection = projection
        self.tokenizer = tokenizer
        self.dataset = data
        self.flatten = flatten_time_series
        self.aligned = aligned
        self.task = task
        self.max_pad = max_pad
        self.max_pad_num = max_pad_num
        self.data_type = data_type
        self.z_norm = z_norm
        self.dataset['audio'][self.dataset['audio'] == -np.inf] = 0.0

    def __getitem__(self, ind):
        """Get item from dataset."""
        # vision = torch.tensor(vision)
        # audio = torch.tensor(audio)
        # text = torch.tensor(text)

        vision = torch.tensor(self.dataset['vision'][ind])
        audio = torch.tensor(self.dataset['audio'][ind])
        text = self.dataset['text'][ind]

        fields = text.split('\t')
        words = self.get_words(fields)
        features = self.project_features(words)
        #
        # if self.aligned:
        #     try:
        #         start = text.nonzero(as_tuple=False)[0][0]
        #         # start = 0
        #     except:
        #         print(text, ind)
        #         exit()
        #     vision = vision[start:].float()
        #     audio = audio[start:].float()
        #     text = text[start:].float()
        # else:
        try:
            vision = vision[vision.nonzero()[0][0]:].float()
        except IndexError as err:
            raise err
        audio = audio[audio.nonzero()[0][0]:].float()

        # z-normalize data
        if self.z_norm:
            vision = torch.nan_to_num(
                (vision - vision.mean(0, keepdims=True)) / (torch.std(vision, axis=0, keepdims=True)))
            audio = torch.nan_to_num((audio - audio.mean(0, keepdims=True)) / (torch.std(audio, axis=0, keepdims=True)))

        def _get_class(flag, data_type=self.data_type):
            if data_type in ['mosi', 'mosei', 'sarcasm']:
                if flag > 0:
                    return [[1]]
                else:
                    return [[0]]
            else:
                return [flag]

        tmp_label = self.dataset['labels'][ind]
        if self.data_type == 'humor' or self.data_type == 'sarcasm':
            if (self.task is None) or (self.task == 'regression'):
                if self.dataset['labels'][ind] < 1:
                    tmp_label = [[-1]]
                else:
                    tmp_label = [[1]]
        else:
            tmp_label = self.dataset['labels'][ind]

        label = torch.tensor(_get_class(tmp_label)).long() if self.task == "classification" else torch.tensor(
            tmp_label).float()

        tmp = [vision, audio[..., :70], label, tmp_label, features]
        for i in range(len(tmp) - 3):
            tmp[i] = tmp[i][:self.max_pad_num]
            tmp[i] = F.pad(tmp[i], (0, 0, 0, self.max_pad_num - tmp[i].shape[0]))

        return tmp

    def __len__(self):
        """Get length of dataset."""
        return self.dataset['vision'].shape[0]

    def project_features(self, words: List[str]) -> np.ndarray:
        encoded = self.tokenizer.encode(words, is_pretokenized=True, add_special_tokens=False)
        tokens = [[] for _ in range(len(words))]
        for index, token in zip(encoded.words, encoded.tokens):
            tokens[index].append(token)
        features = self.projection(tokens)
        padded_featrues = np.pad(features, ((0, self.max_seq_len - len(words)), (0, 0)))
        return padded_featrues

    def normalize(self, text: str) -> str:
        return text.replace('<br />', ' ')

    def get_words(self, fields: List[str]) -> List[str]:
        return [w[0] for w in self.tokenizer.pre_tokenizer.pre_tokenize_str(self.normalize(fields[0]))][:self.max_seq_len]

