# Please install CMU-Multimodal SDK Version 1.2.0 for this dataset
# You can find the instructions here: https://github.com/A2Zadeh/CMU-MultimodalSDK
import os

import numpy as np
from mmsdk import mmdatasdk as md

DATASET = md.cmu_mosi
DATA_PATH = '../data/mosi_raw/'

try:
    md.mmdataset(DATASET.highlevel, DATA_PATH)
except RuntimeError:
    print("High-level features have been downloaded previously.")

try:
    md.mmdataset(DATASET.raw, DATA_PATH)
except RuntimeError:
    print("Raw data have been downloaded previously.")

try:
    md.mmdataset(DATASET.labels, DATA_PATH)
except RuntimeError:
    print("Labels have been downloaded previously.")

visual_field = 'CMU_MOSI_Visual_Facet_41'
acoustic_field = 'CMU_MOSI_COVAREP'
text_field = 'CMU_MOSI_TimestampedWords'

features = [
    text_field,
    visual_field,
    acoustic_field
]

recipe = {feat: os.path.join(DATA_PATH, feat) + '.csd' for feat in features}
dataset = md.mmdataset(recipe)


def avg(intervals: np.array, features: np.array) -> np.array:
    try:
        return np.average(features, axis=0)
    except:
        return features


# first we align to words with averaging, collapse_function receives a list of functions
dataset.align(text_field, collapse_functions=[avg])


label_field = 'CMU_MOSI_Opinion_Labels'

# we add and align to lables to obtain labeled segments
# this time we don't apply collapse functions so that the temporal sequences are preserved
label_recipe = {label_field: os.path.join(DATA_PATH, label_field + '.csd')}
dataset.add_computational_sequences(label_recipe, destination=None)
dataset.align(label_field)