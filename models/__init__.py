import sys

import pytorch_lightning as pl

from .avmnist import *
from .avmnist_post import *
from .convnet import *
from .mmimdb_gmlp import *
from .mmimdb import *
from .multioff import *
from .mmhs150 import *
from .mimic import *


def get_model(model_type: str) -> type[pl.LightningModule]:
    return getattr(sys.modules[__name__], model_type)
