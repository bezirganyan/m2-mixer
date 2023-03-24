import sys

from .avmnist import *
from .pnlp import *
from .get_processed_mmimdb import *


def get_data_module(data_type: str) -> type[pl.LightningDataModule]:
    return getattr(sys.modules[__name__], data_type)
