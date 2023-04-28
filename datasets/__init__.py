import sys

from .avmnist import *
from .pnlp import *
from .get_processed_mmimdb import *
from .mmimdb import *
from .multioff import *
from .get_processed_mmimdb import *
from .mmhs150 import *
from .mimic import *


def get_data_module(data_type: str) -> type[pl.LightningDataModule]:
    return getattr(sys.modules[__name__], data_type)
