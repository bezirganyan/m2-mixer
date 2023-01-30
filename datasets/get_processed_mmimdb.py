import torch
import os
import numpy as np
from omegaconf import DictConfig
from torch.utils.data import Dataset, DataLoader
import random
import string
import argparse
from IPython import embed
from torchvision.transforms import InterpolationMode
import torchvision.transforms as T
import pytorch_lightning as pl

from datasets.transforms import RuinModality

glove = []  # {w: vectors[word2idx[w]] for w in words}
all_letters = string.ascii_letters + " .,;'"
fdim = 0


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, text, label = sample['image'], sample['text'], sample['label']
        return {'image': torch.from_numpy(image.astype(np.float32)),
                # 'text': text,
                'text': torch.from_numpy(text.astype(np.float32)),
                'label': torch.from_numpy(label.astype(np.float32)),
                'textlen': sample['textlen']}


class Normalize(object):
    """Input image cleaning."""

    def __init__(self, mean_vector, std_devs):
        self.mean_vector, self.std_devs = mean_vector, std_devs

    def __call__(self, sample):
        image = sample['image']
        return {'image': self._normalize(image, self.mean_vector, self.std_devs),
                'text': sample['text'],
                'label': sample['label'], 'textlen': sample['textlen']}

    def _normalize(self, tensor, mean, std):
        """Normalize a tensor image with mean and standard deviation.
        See ``Normalize`` for more details.
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
            mean (sequence): Sequence of means for each channel.
            std (sequence): Sequence of standard deviations for each channely.
        Returns:
            Tensor: Normalized Tensor image.
        """
        if not self._is_tensor_image(tensor):
            print(tensor.size())
            raise TypeError('tensor is not a torch image. Its size is {}.'.format(tensor.size()))
        # TODO: make efficient
        for t, m, s in zip(tensor, mean, std):
            t.sub_(m).div_(s)
        return tensor

    def _is_tensor_image(self, img):
        return torch.is_tensor(img) and img.ndimension() == 3


class RandomModalityMuting(object):
    """Randomly turn a mode off."""

    def __init__(self, p_muting=0.1):
        self.p_muting = p_muting

    def __call_(self, sample):
        rval = random.random()

        im = sample['image']
        au = sample['text']
        if rval <= self.p_muting:
            vval = random.random()

            if vval <= 0.5:
                im = sample['image'] * 0
            else:
                au = sample['text'] * 0

        return {'image': im, 'text': au, 'label': sample['label'], 'textlen': sample['textlen']}


class MM_IMDB(Dataset):

    def __init__(self, root_dir='',  # /home/juanma/Documents/Data/MM_IMDB/mmimdb_np
                 transform=None,
                 stage='train',
                 feat_dim=100,
                 args=None):
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

        global fdim
        fdim = feat_dim

    def __len__(self):
        return self.len_data

    def __getitem__(self, idx):

        imagepath = os.path.join(self.root_dir, self.stage, 'image_{:06}.npy'.format(idx))
        labelpath = os.path.join(self.root_dir, self.stage, 'label_{:06}.npy'.format(idx))
        textpath = os.path.join(self.root_dir, self.stage, 'text_{:06}.npy'.format(idx))

        image = np.load(imagepath).astype(np.float32).T
        label = torch.tensor(np.load(labelpath))
        text = np.load(textpath)

        textlen = text.shape[0]

        sample = {'image': image, 'text': np.zeros(20), 'label': label, 'textlen': textlen}

        if self.transform:
            for m in self.transform:
                if m == 'image':
                    sample[m] = self.transform[m](sample[m]).float()
                elif m == 'multimodal':
                    sample = self.transform[m](sample)
                else:
                    sample[m] = self.transform[m](sample[m])

        return sample

#
# def collate_imdb(list_samples):
#     global fdim
#     max_text_len = 0
#     for sample in list_samples:
#         L = len(sample['text'])
#         if max_text_len < L:
#             max_text_len = L
#
#     list_images = len(list_samples) * [None]
#     list_text = len(list_samples) * [None]
#     list_labels = len(list_samples) * [None]
#     list_textlen = len(list_samples) * [None]
#
#     for i, sample in enumerate(list_samples):
#         text_sample_len = len(sample['text'])
#
#         text_i = sample['text'].astype(np.float32)
#         padding = np.asarray([fdim * [-10.0]] * (max_text_len - text_sample_len), np.float32)
#
#         list_images[i] = sample['image']
#         list_labels[i] = sample['label']
#         if padding.shape[0] > 0:
#             list_text[i] = torch.from_numpy(np.concatenate((text_i, padding), 0))
#         else:
#             list_text[i] = torch.from_numpy(text_i)
#         list_textlen[i] = sample['textlen']
#
#     images = torch.transpose(torch.stack(list_images), 1, 3)
#     text = torch.stack(list_text)
#     labels = torch.stack(list_labels)
#
#     return {'image': images, 'text': text, 'label': labels, 'textlen': list_textlen}


class MMIMDBExtDataModule(pl.LightningDataModule):

    def __init__(self, data_dir: str, batch_size: int, num_workers: int, vocab: DictConfig, projection: DictConfig,
                 max_seq_len: int, **kwargs):
        super().__init__(**kwargs)
        self.max_seq_len = max_seq_len
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.vocab_cfg = vocab

    def setup(self, stage: str = None):
        train_transforms = dict(image=T.Compose([
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )]),
            multimodal=T.RandomApply([RuinModality(p=0.3)], p=0.0))

        val_test_transforms = dict(image=T.Compose([
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )]))

        # train_transforms = dict(image=T.Compose([
        #     T.ToTensor(),
        #     T.Resize(size=256, interpolation=InterpolationMode.BICUBIC, max_size=None, antialias=None),
        #     T.CenterCrop(size=(224, 224)),
        #     T.Normalize(mean=torch.tensor([0.5000, 0.5000, 0.5000]), std=torch.tensor([0.5000, 0.5000, 0.5000]))]))
        # val_test_transforms = train_transforms

        self.train_set = MM_IMDB(os.path.join(self.data_dir), stage='train', transform=train_transforms)
        self.eval_set = MM_IMDB(os.path.join(self.data_dir), stage='dev', transform=val_test_transforms)
        self.test_set = MM_IMDB(os.path.join(self.data_dir), stage='test', transform=val_test_transforms)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_set, self.batch_size, shuffle=True,
                          num_workers=self.num_workers, persistent_workers=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.eval_set, self.batch_size, shuffle=False,
                          num_workers=self.num_workers, persistent_workers=True)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_set, self.batch_size, shuffle=False,
                          num_workers=self.num_workers, persistent_workers=True)


if __name__ == '__main__':
    def parse_args():
        parser = argparse.ArgumentParser(description='Modality optimization.')
        parser.add_argument('--datadir', type=str, help='data directory',
                            default='/mnt/scratch/xiaoxiang/yihang/mmimdb/')
        parser.add_argument('--small_dataset', action='store_true', default=False, help='dataset scale')
        parser.add_argument('--average_text', action='store_true', default=False, help='averaging text features')
        return parser.parse_args("")


    args = parse_args()

    dataset = MM_IMDB(root_dir=args.datadir,
                      transform=None,
                      stage='dev',
                      feat_dim=300,
                      args=args)

    embed()
