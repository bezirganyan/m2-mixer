import PIL.Image
import h5py
import json
# import logging
import math
import os
import numpy
import re
import sys
from collections import OrderedDict, Counter

from PIL import Image
from PIL.Image import Resampling
# from fuel.datasets import H5PYDataset
# from fuel.utils import find_in_data_path
from gensim.models.word2vec import KeyedVectors
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm


def resize_and_crop_image(input_file, output_box=(224, 224), fit=True):
    # https://github.com/BVLC/caffe/blob/master/tools/extra/resize_and_crop_images.py
    '''Downsample the image.
    '''
    img = Image.open(input_file)
    box = output_box
    # preresize image with factor 2, 4, 8 and fast algorithm
    factor = 1
    while img.size[0] / factor > 2 * box[0] and img.size[1] * 2 / factor > 2 * box[1]:
        factor *= 2
    if factor > 1:
        img.thumbnail(
            (img.size[0] / factor, img.size[1] / factor), Resampling.NEAREST)

    # calculate the cropping box and get the cropped part
    if fit:
        x1 = y1 = 0
        x2, y2 = img.size
        wRatio = 1.0 * x2 / box[0]
        hRatio = 1.0 * y2 / box[1]
        if hRatio > wRatio:
            y1 = int(y2 / 2 - box[1] * wRatio / 2)
            y2 = int(y2 / 2 + box[1] * wRatio / 2)
        else:
            x1 = int(x2 / 2 - box[0] * hRatio / 2)
            x2 = int(x2 / 2 + box[0] * hRatio / 2)
        img = img.crop((x1, y1, x2, y2))

    # Resize the image with best quality algorithm ANTI-ALIAS
    img = img.resize(box, Resampling.LANCZOS).convert('RGB')
    img = numpy.array(img)
    # img[:, :, 0] -= 103.939
    # img[:, :, 1] -= 116.779
    # img[:, :, 2] -= 123.68
    # img = img.transpose((2, 0, 1))
    # img = numpy.expand_dims(img, axis=0)
    return img


# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
mmimdb_path = '../data/mm_imdb_orig/mmimdb'

with open(os.path.join(mmimdb_path, 'list.txt'), 'r') as f:
    files = f.read().splitlines()

## Load data and define vocab ##
# logger.info('Reading json and jpeg files...')
movies = []
vocab_counts = []
img_size = (160, 256)
num_channels = 3
test_size = 0.3
dev_size = 0.1
rng_seed = [2014, 8, 6]
n_classes = 23
word2vec_path = 'GoogleNews-vectors-negative300.bin'


# clsf = VGGClassifier(model_path='vgg16.tar', synset_words='synset_words.txt')

def normalizeText(text):
    text = text.lower()
    text = re.sub(r'<br />', r' ', text).strip()
    text = re.sub(r'^https?:\/\/.*[\r\n]*', ' L ', text, flags=re.MULTILINE)
    text = re.sub(r'[\~\*\+\^`_#\[\]|]', r' ', text).strip()
    text = re.sub(r'[0-9]+', r' N ', text).strip()
    text = re.sub(r'([/\'\-\.?!\(\)",:;])', r' \1 ', text).strip()
    return text.split()

print('Loading Data...')
for i, filename in tqdm(enumerate(files), total=len(files)):
    file = os.path.join(mmimdb_path, filename)
    with open(file) as f:
        data = json.load(f)
        data['imdb_id'] = file.split('/')[-1].split('.')[0]
        # if 'plot' in data and 'plot outline' in data:
        #    data['plot'].append(data['plot outline'])
        im_file = file.replace('json', 'jpeg')
        if all([k in data for k in ('genres', 'plot')] + [os.path.isfile(im_file)]):
            plot_id = numpy.array([len(p) for p in data['plot']]).argmax()
            data['plot'] = normalizeText(data['plot'][plot_id])
            if len(data['plot']) > 0:
                vocab_counts.extend(data['plot'])
                data['cover'] = resize_and_crop_image(im_file, img_size)
                movies.append(data)
    # logger.info('{0:05d} out of {1:05d}: {2:02.2f}%'.format(
    #     i, len(files), float(i) / len(files) * 100))

print('Removing non-vocabulary words...')
vocab_counts = OrderedDict(Counter(vocab_counts).most_common())
vocab = ['_UNK_'] + [v for v in vocab_counts.keys()]
googleword2vec = KeyedVectors.load_word2vec_format(word2vec_path, binary=True)

# Define train, dev and test subsets
print('Splitting data...')
counts = OrderedDict(
    Counter([g for m in movies for g in m['genres']]).most_common())
target_names = list(counts.keys())[:n_classes]

le = MultiLabelBinarizer()
Y = le.fit_transform([m['genres'] for m in movies])
labels = numpy.nonzero(le.transform([[t] for t in target_names]))[1]

B = numpy.copy(Y)
rng = numpy.random.RandomState(rng_seed)
train_idx, dev_idx, test_idx = [], [], []
for l in labels[::-1]:
    t = B[:, l].nonzero()[0]
    t = rng.permutation(t)
    n_test = int(math.ceil(len(t) * test_size))
    n_dev = int(math.ceil(len(t) * dev_size))
    n_train = len(t) - n_test - n_dev
    test_idx.extend(t[:n_test])
    dev_idx.extend(t[n_test:n_test + n_dev])
    train_idx.extend(t[n_test + n_dev:])
    B[t, :] = 0

indices = numpy.concatenate([train_idx, dev_idx, test_idx])
nsamples = len(indices)
nsamples_train, nsamples_dev, nsamples_test = len(
    train_idx), len(dev_idx), len(test_idx)

# Obtain feature vectors and text sequences
print('Obtaining text sequences...')
sequences = []
X = []
for i, idx in enumerate(indices):
    words = movies[idx]['plot']
    X.append([w for w in words if w in googleword2vec])

del googleword2vec

output_dir = '../output'

split_dict = {
    'train': {
        'text': (0, nsamples_train),
        'images': (0, nsamples_train)},
    'dev': {
        'text': (nsamples_train, nsamples_train + nsamples_dev),
        'images': (nsamples_train, nsamples_train + nsamples_dev)},
    'test': {
        'text': (nsamples_train + nsamples_dev, nsamples),
        'images': (nsamples_train + nsamples_dev, nsamples)}
}

print('Saving data...')
for set in split_dict:
    print(f'Saving {set} data...')
    if not os.path.exists(f'{output_dir}/{set}'):
        os.makedirs(f'{output_dir}/{set}')

    for modality in split_dict[set]:
        if not os.path.exists(f'{output_dir}/{set}/{modality}'):
            os.makedirs(f'{output_dir}/{set}/{modality}')

    print(' - Saving text data...')
    for idx in tqdm(range(split_dict[set]['text'][0], split_dict[set]['text'][1])):
        with open(f'{output_dir}/{set}/text/{movies[idx]["imdb_id"]}.txt', 'w') as f:
            f.write(' '.join(X[idx]))
    print(' - Saving image data...')
    for idx in tqdm(range(split_dict[set]['images'][0], split_dict[set]['images'][1])):
        image = movies[idx]['cover'].T
        plt.imsave(f'{output_dir}/{set}/images/{movies[idx]["imdb_id"]}.jpeg', image.T)

    print(' - Saving labels...')
    with open(f'{output_dir}/{set}/labels.csv', 'w') as f:
        f.write('imdb_id,genre')
        for idx in tqdm(range(split_dict[set]['images'][0], split_dict[set]['images'][1])):
            f.write(f'\n{movies[idx]["imdb_id"]},{Y[idx]}')


# Plot distribution
# print('Plotting distribution...')
# cm = numpy.zeros((n_classes, n_classes), dtype='int')
# for i, l in enumerate(labels):
#     cm[i] = Y[Y[:, l].nonzero()[0]].sum(axis=0)[labels]
#
# cmap = plt.cm.Blues
# cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, numpy.newaxis]
# for i in range(len(target_names)):
#     cm_normalized[i, i] = 0
# plt.imshow(cm_normalized, interpolation='nearest', cmap=cmap, aspect='auto')
# for i, cas in enumerate(cm):
#     for j, c in enumerate(cas):
#         if c > 0:
#             plt.text(j - .2, i + .2, c, fontsize=4)
# plt.title('Shared labels', fontsize='smaller')
# plt.colorbar()
# tick_marks = numpy.arange(len(target_names))
# plt.xticks(tick_marks, target_names, rotation=90)
# plt.yticks(tick_marks, target_names)
# plt.tight_layout()
# plt.savefig(f'{output_dir}/distribution.pdf')
# plt.close()
