import json
import os
from typing import List

import gensim
import numpy as np
from PIL import Image
from omegaconf import DictConfig
from tokenizers.implementations import BertWordPieceTokenizer
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode

from utils.projection import Projection


class MMHS150DataModule(pl.LightningDataModule):

    def __init__(self, data_dir: str, batch_size: int, num_workers: int, vocab: DictConfig, projection: DictConfig,
                 max_seq_len: int, task: str, word_proj='pnlp', **kwargs):
        super().__init__(**kwargs)
        self.word_proj = word_proj
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
                                         task=self.task,
                                         word_proj=self.word_proj)
        self.eval_set = MMHS150Dataset(os.path.join(self.data_dir), stage='dev', tokenizer=self.tokenizer,
                                        projection=self.projecion, max_seq_len=self.max_seq_len,
                                        transform=val_test_transforms,
                                        task=self.task,
                                        word_proj=self.word_proj)
        self.test_set = MMHS150Dataset(os.path.join(self.data_dir), stage='test', tokenizer=self.tokenizer,
                                        projection=self.projecion, max_seq_len=self.max_seq_len,
                                        transform=val_test_transforms,
                                        task=self.task,
                                        word_proj=self.word_proj)

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

    def __init__(self, root_dir, tokenizer, projection=None, max_seq_len=None, feat_dim=100, stage='train', task='binary',
                 transform=None, word_proj='pnlp'):
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
        self.max_seq_len = max_seq_len
        self.word_proj = word_proj
        if self.word_proj == 'pnlp':
            self.projection = projection
        elif self.word_proj == 'word2vec':
            self.word2vec = gensim.models.KeyedVectors.load_word2vec_format('pretrained/GoogleNews-vectors-negative300.bin', binary=True)
        else:
            raise NotImplementedError(f'Word projection {self.word_proj} not implemented')
        with open(os.path.join(self.root_dir, 'MMHS150K_GT.json')) as f:
            self.texts = json.load(f)

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
            ocr_text = 'none'
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

        fields_ocr = sample['ocr'].split('\t')
        words_ocr = self.get_words(fields_ocr)

        if self.word_proj == 'pnlp':
            features_ocr = self.project_features(words_ocr)
            features = self.project_features(words)
        else:
            if (len(words_ocr) == 1) and (words_ocr[0] == 'none'):
                features_ocr = np.zeros((1, 300), dtype=np.float32)
                use_features_ocr = 0
            else:
                features_ocr = np.stack([self.word2vec[w] for w in words_ocr])
                use_features_ocr = 1
            if (len(words) == 1) and (words[0] == 'none'):
                features = np.zeros((1, 300), dtype=np.float32)
                use_features = 0
            else:
                features = np.stack([self.word2vec[w] for w in words])
                use_features = 1
            features = np.pad(features, ((0, self.max_seq_len - features.shape[0]), (0, 0)), 'constant')
            features_ocr = np.pad(features_ocr, ((0, self.max_seq_len - features_ocr.shape[0]), (0, 0)), 'constant')
        sample['ocr'] = features_ocr
        sample['text'] = features
        sample['use_features'] = use_features
        sample['use_features_ocr'] = use_features_ocr
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
        words = [w[0] for w in self.tokenizer.pre_tokenizer.pre_tokenize_str(self.normalize(fields[0]))]
        if self.word_proj == 'word2vec':
            words = [w for w in words if w in self.word2vec]
        if len(words) == 0:
            words = ['none']
        return words[:self.max_seq_len]
