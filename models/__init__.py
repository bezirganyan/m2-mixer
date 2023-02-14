import sys

import pytorch_lightning as pl

from .avmnist import *
from .convnet import *
from .mmimdb_mixer import *
from .mmimdb_gmlp import *


def get_model(model_type: str) -> type[pl.LightningModule]:
    return getattr(sys.modules[__name__], model_type)
