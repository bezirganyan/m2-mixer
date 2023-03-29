import json
import os
from typing import List

import numpy as np
import pandas as pd
from PIL import Image
from omegaconf import DictConfig
from tokenizers.implementations import BertWordPieceTokenizer
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode

from datasets.transforms import RuinModality
from utils.projection import Projection


class MMHS150DataModule(pl.LightningDataModule):

    def __init__(self, data_dir: str, batch_size: int, num_workers: int, vocab: DictConfig, projection: DictConfig,
                 max_seq_len: int, task: str, **kwargs):
        super().__init__(**kwargs)
        self.max_seq_len = max_seq_len
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.task = task
        self.num_workers = num_workers
        self.vocab_cfg = vocab
        self.projecion = Projection(vocab.vocab_path, projection.feature_size, projection.window_size)
        self.tokenizer = BertWordPieceTokenizer(**vocab.tokenizer)

    def setup(self, stage: str = None):
        train_transforms = dict(image=T.Compose([
            T.ToTensor(),
            T.Resize(size=(256, 256), interpolation=InterpolationMode.BICUBIC, max_size=None, antialias=None),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )]))

        val_test_transforms = dict(image=T.Compose([
            T.ToTensor(),
            T.Resize(size=(256, 256), interpolation=InterpolationMode.BICUBIC, max_size=None, antialias=None),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )]))

        self.train_set = MMHS150Dataset(os.path.join(self.data_dir), stage='train', tokenizer=self.tokenizer,
                                         projection=self.projecion, max_seq_len=self.max_seq_len,
                                         transform=train_transforms,
                                         task=self.task)
        self.eval_set = MMHS150Dataset(os.path.join(self.data_dir), stage='dev', tokenizer=self.tokenizer,
                                        projection=self.projecion, max_seq_len=self.max_seq_len,
                                        transform=val_test_transforms,
                                        task=self.task)
        self.test_set = MMHS150Dataset(os.path.join(self.data_dir), stage='test', tokenizer=self.tokenizer,
                                        projection=self.projecion, max_seq_len=self.max_seq_len,
                                        transform=val_test_transforms,
                                        task=self.task)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_set, self.batch_size, shuffle=True,
                          num_workers=self.num_workers, persistent_workers=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.eval_set, self.batch_size, shuffle=False,
                          num_workers=self.num_workers, persistent_workers=True)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_set, self.batch_size, shuffle=False,
                          num_workers=self.num_workers, persistent_workers=True)


class MMHS150Dataset(Dataset):

    def __init__(self, root_dir, tokenizer, projection, max_seq_len, feat_dim=100, stage='train', task='binary',
                 transform=None):
        """
        Args:
            root_dir (string): Directory where data is.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        if stage == 'train':
            self.ref = np.loadtxt(os.path.join(root_dir, 'splits/train_ids.txt'), dtype=str)
        elif stage == 'test':
            self.ref = np.loadtxt(os.path.join(root_dir, 'splits/test_ids.txt'), dtype=str)
        elif stage == 'dev':
            self.ref = np.loadtxt(os.path.join(root_dir, 'splits/val_ids.txt'), dtype=str)
        self.len_data = self.ref.shape[0]
        self.transform = transform
        self.task = task
        self.root_dir = root_dir
        self.stage = stage
        self.tokenizer = tokenizer
        self.projection = projection
        self.max_seq_len = max_seq_len
        with open(os.path.join(self.root_dir, 'MMHS150K_GT.json')) as f:
            self.texts = json.load(f)
        global fdim
        fdim = feat_dim

    def __len__(self):
        return self.len_data

    def __getitem__(self, idx: int) -> dict:

        imagepath = os.path.join(self.root_dir, 'img_resized', f'{self.ref[idx]}.jpg')
        # Check if ocr text file exists
        textpath = os.path.join(self.root_dir, 'img_txt', f'{self.ref[idx]}.json')
        if os.path.exists(textpath):
            with open(textpath) as f:
                ocr_text = json.load(f)
            ocr_text = ocr_text['img_text']
        else:
            ocr_text = 'notavailabletext'
        text = self.texts.get(self.ref[idx], {}).get('tweet_text', 'none')
        label = np.array(self.texts[self.ref[idx]]['labels']).astype(int)
        label = (label > 0).astype(int)
        if label.sum() > 1:
            label = 1
        else:
            label = 0

        image = Image.open(imagepath).convert('RGB')
        textlen = text.count(' ') + 1
        sample = {'image': image, 'text': text, 'label': label, 'ocr': ocr_text, 'textlen': textlen}

        if self.transform:
            for m in self.transform:
                if m == 'image':
                    sample[m] = self.transform[m](sample[m])
                elif m == 'multimodal':
                    sample = self.transform[m](sample)
                else:
                    sample[m] = self.transform[m](sample[m])

        fields = sample['text'].split('\t')
        words = self.get_words(fields)
        features = self.project_features(words)
        sample['text'] = features

        fields_ocr = sample['ocr'].split('\t')
        words_ocr = self.get_words(fields_ocr)
        features_ocr = self.project_features(words_ocr)
        sample['ocr'] = features_ocr

        return sample

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
        return [w[0] for w in self.tokenizer.pre_tokenizer.pre_tokenize_str(self.normalize(fields[0]))][
               :self.max_seq_len]
