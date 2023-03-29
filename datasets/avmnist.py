from __future__ import print_function, division
import torch
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
import random
import pytorch_lightning as pl
import torchvision.transforms as T


# %%
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        tensor_dict = {}
        if sample.get('audio') is not None:
            tensor_dict['audio'] = torch.from_numpy(sample['audio'].astype(np.float32))
        if sample.get('image') is not None:
            tensor_dict['image'] = torch.from_numpy(sample['image'].astype(np.float32))
        tensor_dict['label'] = int(sample['label'])

        return tensor_dict


class Normalize(object):
    """Input image cleaning."""

    def __init__(self, mean_vector, std_devs):
        self.mean_vector, self.std_devs = mean_vector, std_devs

    def __call__(self, sample):
        tensor_dict = {}
        if sample.get('audio') is not None:
            tensor_dict['audio'] = sample['audio']
        if sample.get('image') is not None:
            image = self._normalize(sample['image'], mean=self.mean_vector, std=self.std_devs)
            tensor_dict['image'] = image
        tensor_dict['label'] = int(sample['label'])

        return tensor_dict

    def _normalize(self, tensor, mean, std):
        """Normalize a tensor image with mean and standard deviation.
        See ``Normalize`` for more details.
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
            mean (sequence): Sequence of means for each channel.
            std (sequence): Sequence of standard deviations for each channel.
        Returns:
            Tensor: Normalized Tensor image.
        """
        if not self._is_tensor_image(tensor):
            raise TypeError('tensor is not a torch image.')
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

    def __call__(self, sample):
        rval = random.random()

        im = sample['image']
        au = sample['audio']
        if rval <= self.p_muting:
            vval = random.random()

            if vval <= 0.5:
                im = sample['image'] * 0
            else:
                au = sample['audio'] * 0

        return {'image': im, 'audio': au, 'label': sample['label']}


# %%
class AVMnist(Dataset):

    def __init__(self, root_dir='./avmnist',  # args.datadir
                 transform=None,
                 stage='train',
                 modal_separate=False,
                 modal=None):
        """
        Args:
            root_dir (string): Directory where data is.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.transform = transform
        self.modal_separate = modal_separate
        self.modal = modal
        if not modal_separate:
            if stage == 'train':
                self.audio_data = np.load(os.path.join(root_dir, 'audio', 'train_data.npy'))
                self.mnist_data = np.load(os.path.join(root_dir, 'image', 'train_data.npy'))
                self.labels = np.load(os.path.join(root_dir, 'train_labels.npy'))
            else:
                self.audio_data = np.load(os.path.join(root_dir, 'audio', 'test_data.npy'))
                self.mnist_data = np.load(os.path.join(root_dir, 'image', 'test_data.npy'))
                self.labels = np.load(os.path.join(root_dir, 'test_labels.npy'))

            self.audio_data = self.audio_data[:, np.newaxis, :, :]
            self.mnist_data = self.mnist_data.reshape(self.mnist_data.shape[0], 1, 28, 28)
        else:
            if modal:
                if modal not in ['audio', 'image']:
                    raise ValueError('the value of modal is allowed')

                if stage == 'train':
                    self.data = np.load(os.path.join(root_dir, modal, 'train_data.npy'))
                    self.labels = np.load(os.path.join(root_dir, 'train_labels.npy'))
                else:
                    self.data = np.load(os.path.join(root_dir, modal, 'test_data.npy'))
                    self.labels = np.load(os.path.join(root_dir, 'test_labels.npy'))

                if modal == 'audio':
                    self.data = self.data[:, np.newaxis, :, :]
                elif modal == 'image':
                    self.data = self.data.reshape(self.data.shape[0], 1, 28, 28)

            else:
                raise ValueError('the value of modal should be given')

    def __len__(self):
        return self.mnist_data.shape[0] if not self.modal_separate else self.data.shape[0]

    def __getitem__(self, idx):
        if not self.modal_separate:
            image = self.mnist_data[idx]
            audio = self.audio_data[idx]
            label = self.labels[idx]

            sample = {'image': image, 'audio': audio, 'label': label}
        else:
            data = self.data[idx]
            label = self.labels[idx]

            sample = {self.modal: data, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample


class AVMnistDataModule(pl.LightningDataModule):

    def __init__(self, data_dir, batch_size, num_workers, p_muting=0.0, **kwargs):
        super().__init__(**kwargs)
        self.p_muting = p_muting
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: str = None):
        # transform = T.Compose([ToTensor(), RandomModalityMuting(p_muting=self.p_muting)])
        transform = T.Compose([ToTensor()])

        train_dataset = AVMnist(root_dir=self.data_dir, transform=transform, stage='train')
        val_dataset = AVMnist(root_dir=self.data_dir, transform=transform, stage='train')
        self.test_dataset = AVMnist(root_dir=self.data_dir, transform=transform, stage='test')

        train_idxs = list(range(0, 55000))
        valid_idxs = list(range(55000, 60000))

        self.train_subset = Subset(train_dataset, train_idxs)
        self.valid_subset = Subset(val_dataset, valid_idxs)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.train_subset, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.valid_subset, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.test_dataset, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers)


class AVMnistIntermediate(Dataset):

    def __init__(self, root_dir='./corrects_data',  # args.datadir
                 stage='train', modality='multi'):
        if modality == 'multi':
            self.data_image = np.load(os.path.join(root_dir, stage + '/image_vectors.npy'))
            self.data_audio = np.load(os.path.join(root_dir, stage + '/audio_vectors.npy'))
            self.data_fusion = np.load(os.path.join(root_dir, stage + '/fusion_vectors.npy'))
            self.image_labels = np.load(os.path.join(root_dir, stage + '/image_corrects.npy'))
            self.audio_labels = np.load(os.path.join(root_dir, stage + '/audio_corrects.npy'))
            self.fusion_labels = np.load(os.path.join(root_dir, stage + '/fusion_corrects.npy'))
        elif modality in ('image', 'audio', 'fusion'):
            self.data = np.load(os.path.join(root_dir, stage + '/' + modality + '_vectors.npy'))
            self.labels = np.load(os.path.join(root_dir, stage + '/' + modality + '_corrects.npy'))
        else:
            raise ValueError('Modality should be one of multi, image, audio, fusion')

    def __len__(self):
        if hasattr(self, 'data'):
            return self.data.shape[0]
        else:
            return self.data_image.shape[0]

    def __getitem__(self, idx):
        if hasattr(self, 'data'):
            data = self.data[idx]
            label = self.labels[idx]
            sample = {'data': data, 'label': label}
        else:
            image = self.data_image[idx]
            audio = self.data_audio[idx]
            fusion = self.data_fusion[idx]
            image_label = self.image_labels[idx]
            audio_label = self.audio_labels[idx]
            fusion_label = self.fusion_labels[idx]

            sample = {'image': image, 'audio': audio, 'fusion': fusion,
                      'image_label': image_label, 'audio_label': audio_label, 'fusion_label': fusion_label}

        return sample


class AVMnistIntermediateDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, num_workers, modality='multi', **kwargs):
        super().__init__()
        self.test_dataset = None
        self.valid_dataset = None
        self.train_dataset = None
        self.modality = modality
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: str = None):
        self.train_dataset = AVMnistIntermediate(root_dir=self.data_dir, stage='train', modality=self.modality)
        self.valid_dataset = AVMnistIntermediate(root_dir=self.data_dir, stage='train', modality=self.modality)
        self.test_dataset = AVMnistIntermediate(root_dir=self.data_dir, stage='test', modality=self.modality)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.valid_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.test_dataset, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers)
