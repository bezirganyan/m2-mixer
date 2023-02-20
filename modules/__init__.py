import sys

from .mixer import *
from .gmpl import *
from .fusion import *
from .dynamixer import *


def get_block_by_name(**kwargs):
    thismodule = sys.modules[__name__]
    block = getattr(thismodule, kwargs['block_type'])
    return block(**kwargs)


def get_fusion_by_name(**kwargs):
    thismodule = sys.modules[__name__]
    fusion = getattr(thismodule, kwargs['fusion_function'])
    return fusion(**kwargs)
