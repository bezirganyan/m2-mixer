import os
from typing import List

import numpy as np
from PIL import Image
from omegaconf import DictConfig
from tokenizers.implementations import BertWordPieceTokenizer
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import torchvision.transforms as T

from datasets.transforms import RuinModality
from utils.projection import Projection


class MMIMDBDataModule(pl.LightningDataModule):

    def __init__(self, data_dir: str, batch_size: int, num_workers: int, vocab: DictConfig, projection: DictConfig,
                 max_seq_len: int, **kwargs):
        super().__init__(**kwargs)
        self.max_seq_len = max_seq_len
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.vocab_cfg = vocab
        self.projecion = Projection(vocab.vocab_path, projection.feature_size, projection.window_size)
        self.tokenizer = BertWordPieceTokenizer(**vocab.tokenizer)

    def setup(self, stage: str = None):
        train_transforms = dict(image=T.Compose([
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )]),
            multimodal=T.RandomApply([RuinModality(p=0.3)], p=0.3))

        val_test_transforms = dict(image=T.Compose([
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )]))
        self.train_set = MMIMDBDataset(os.path.join(self.data_dir), stage='train', tokenizer=self.tokenizer,
                                       projection=self.projecion, max_seq_len=self.max_seq_len,
                                       transform=train_transforms)
        self.eval_set = MMIMDBDataset(os.path.join(self.data_dir), stage='dev', tokenizer=self.tokenizer,
                                      projection=self.projecion, max_seq_len=self.max_seq_len,
                                      transform=val_test_transforms)
        self.test_set = MMIMDBDataset(os.path.join(self.data_dir), stage='test', tokenizer=self.tokenizer,
                                      projection=self.projecion, max_seq_len=self.max_seq_len,
                                      transform=val_test_transforms)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_set, self.batch_size, shuffle=True,
                          num_workers=self.num_workers, persistent_workers=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.eval_set, self.batch_size, shuffle=False,
                          num_workers=self.num_workers, persistent_workers=True)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_set, self.batch_size, shuffle=False,
                          num_workers=self.num_workers, persistent_workers=True)


class MMIMDBDataset(Dataset):

    def __init__(self, root_dir, tokenizer, projection, max_seq_len, feat_dim=100, stage='train', transform=None):
        """
        Args:
            root_dir (string): Directory where data is.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        if stage == 'train':
            self.len_data = 15552
        elif stage == 'test':
            self.len_data = 7799
        elif stage == 'dev':
            self.len_data = 2608

        self.transform = transform
        self.root_dir = root_dir
        self.stage = stage
        self.tokenizer = tokenizer
        self.projection = projection
        self.max_seq_len = max_seq_len

        global fdim
        fdim = feat_dim

    def __len__(self):
        return self.len_data

    def __getitem__(self, idx: int) -> dict:

        imagepath = os.path.join(self.root_dir, self.stage, 'images', f'image_{idx}.jpeg')
        labelpath = os.path.join(self.root_dir, self.stage, 'labels', f'label_{idx}.npy')
        textpath = os.path.join(self.root_dir, self.stage, 'text', f'text_{idx}.txt')

        image = np.array(Image.open(imagepath).convert('RGB')).T
        label = np.load(labelpath)
        with open(textpath, 'r') as f:
            text = f.read()

        textlen = text.count(' ') + 1

        sample = {'image': image, 'text': text, 'label': label, 'textlen': textlen}
        if self.transform:
            for m in self.transform:
                if m == 'image':
                    sample[m] = self.transform[m](sample[m].T).mT
                elif m == 'multimodal':
                    sample = self.transform[m](sample)
                else:
                    sample[m] = self.transform[m](sample[m])

        fields = sample['text'].split('\t')
        words = self.get_words(fields)
        features = self.project_features(words)
        sample['text'] = features

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
