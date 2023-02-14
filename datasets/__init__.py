import sys

import pytorch_lightning as pl

from .avmnist import *
from .imagenet_dataset import *
from .multimodal import *
from .pnlp import *
from .get_processed_mmimdb import *


def get_data_module(data_type: str) -> type[pl.LightningDataModule]:
    return getattr(sys.modules[__name__], data_type)
