"""Handle getting raw data from mosi"""
# from mosi_split import train_fold, valid_fold, test_fold
import pickle
import sys
import os
import numpy as np
import h5py
import re
import torchtext as text
from collections import defaultdict
import pickle

sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))


def lpad(this_array, seq_len):
    """Left pad array with seq_len 0s.

    Args:
        this_array (np.array): Array to pad
        seq_len (int): Number of 0s to pad.

    Returns:
        np.array: Padded array
    """
    temp_array = np.concatenate([np.zeros(
        [seq_len] + list(this_array.shape[1:])), this_array], axis=0)[-seq_len:, ...]
    return temp_array


def detect_entry_fold(entry, folds):
    """Detect entry fold.

    Args:
        entry (str): Entry string
        folds (int): Number of folds

    Returns:
        int: Entry fold index
    """
    entry_id = entry.split("[")[0]
    for i in range(len(folds)):
        if entry_id in folds[i]:
            return i
    return None


train_fold = ['2iD-tVS8NPw', '8d-gEyoeBzc', 'Qr1Ca94K55A', 'Ci-AH39fi3Y', '8qrpnFRGt2A', 'Bfr499ggo-0', 'QN9ZIUWUXsY', '9T9Hf74oK10', '7JsX8y1ysxY', '1iG0909rllw', 'Oz06ZWiO20M', 'BioHAh1qJAQ', '9c67fiY0wGQ', 'Iu2PFX3z_1s', 'Nzq88NnDkEk', 'Clx4VXItLTE', '9J25DZhivz8', 'Af8D0E4ZXaw', 'TvyZBvOMOTc', 'W8NXH0Djyww', '8OtFthrtaJM', '0h-zjBukYpk', 'Vj1wYRQjB-o', 'GWuJjcEuzt8', 'BI97DNYfe5I', 'PZ-lDQFboO8', '1DmNV9C1hbY', 'OQvJTdtJ2H4', 'I5y0__X72p0', '9qR7uwkblbs', 'G6GlGvlkxAQ', '6_0THN4chvY', 'Njd1F0vZSm4', 'BvYR0L6f2Ig', '03bSnISJMiM', 'Dg_0XKD0Mf4', '5W7Z1C_fDaE', 'VbQk4H8hgr0', 'G-xst2euQUc', 'MLal-t_vJPM', 'BXuRRbG0Ugk', 'LSi-o-IrDMs', 'Jkswaaud0hk', '2WGyTLYerpo', '6Egk_28TtTM', 'Sqr0AcuoNnk', 'POKffnXeBds', '73jzhE8R1TQ', 'OtBXNcAL_lE', 'HEsqda8_d0Q', 'VCslbP0mgZI', 'IumbAb8q2dM']
valid_fold = ['WKA5OygbEKI', 'c5xsKMxpXnc', 'atnd_PF-Lbs', 'bvLlb-M3UXU', 'bOL9jKpeJRs', '_dI--eQ6qVU', 'ZAIRrfG22O0', 'X3j2zQgwYgE', 'aiEXnCPZubE', 'ZUXBRvtny7o']
test_fold = ['tmZoasNr4rU', 'zhpQhgha_KU', 'lXPQBPVc5Cw', 'iiK8YX8oH1E', 'tStelxIAHjw', 'nzpVDcQ0ywM', 'etzxEpPuc6I', 'cW1FSBF59ik', 'd6hH302o4v8', 'k5Y_838nuGo', 'pLTX3ipuDJI', 'jUzDDGyPkXU', 'f_pcplsH_V0', 'yvsjCA6Y5Fc', 'nbWiPyCm4g0', 'rnaNMUZpvvg', 'wMbj6ajWbic', 'cM3Yna7AavY', 'yDtzw_Y-7RU', 'vyB00TXsimI', 'dq3Nf_lMPnE', 'phBUpBr1hSo', 'd3_k5Xpfmik', 'v0zCBqDeKcE', 'tIrG4oNLFzE', 'fvVhgmXxadc', 'ob23OKe5a9Q', 'cXypl4FnoZo', 'vvZ4IcEtiZc', 'f9O3YtZ2VfI', 'c7UH_rxdZv4']

folds = [train_fold, valid_fold, test_fold]
print('folds:')
print(len(train_fold))
print(len(valid_fold))
print(len(test_fold))

affect_data = h5py.File('../data/mosi/mosi.hdf5', 'r')
print(affect_data.keys())

AUDIO = 'COVAREP'
VIDEO = 'FACET_4.2'
WORD = 'words'
labels = ['Opinion Segment Labels']

csds = [AUDIO, VIDEO, labels[0]]

seq_len = 50

keys = list(affect_data[WORD].keys())
print(len(keys))


def get_rawtext(path, data_kind, vids):
    """Get raw text modality.

    Args:
        path (str): Path to h5 file
        data_kind (str): String for data format. Should be 'hdf5'.
        vids (list): List of video ids.

    Returns:
        tuple(list,list): Tuple of text_data and video_data in lists.
    """
    if data_kind == 'hdf5':
        f = h5py.File(path, 'r')
        text_data = []
        new_vids = []
        count = 0
        for vid in vids:
            text = []
            # (id, seg) = re.match(r'([-\w]*)_(\w+)', vid).groups()
            # vid_id = '{}[{}]'.format(id, seg)
            vid_id = vid
            # TODO: fix 31 missing entries
            try:
                for word in f['words'][vid_id]['features']:
                    if word[0] != b'sp':
                        text.append(word[0].decode('utf-8'))
                text_data.append(' '.join(text))
                new_vids.append(vid_id)
            except:
                print("missing", vid, vid_id)
        return text_data, new_vids
    else:
        print('Wrong data kind!')


def get_audio_visual_text(csds, seq_len, text_data, vids):
    """Get audio visual from text."""
    data = [{} for _ in range(3)]
    output = [{} for _ in range(3)]

    for i in range(len(folds)):
        for csd in csds:
            data[i][csd] = []
        data[i]['words'] = []
        data[i]['id'] = []

    for i, key in enumerate(vids):
        which_fold = detect_entry_fold(key, folds)

        if which_fold == None:
            print("Key %s doesn't belong to any fold ... " %
                  str(key), error=False)
            continue
        for csd in csds:
            this_array = affect_data[csd][key]["features"]
            if csd in labels:
                data[which_fold][csd].append(this_array)
            else:
                data[which_fold][csd].append(lpad(this_array, seq_len=seq_len))
        data[which_fold]['words'].append(text_data[i])
        data[which_fold]['id'].append(key)

    for i in range(len(folds)):
        for csd in csds:
            output[i][csd] = np.array(data[i][csd])
        output[i]['words'] = np.stack(data[i]['words'])
        output[i]['id'] = data[i]['id']

    fold_names = ["train", "valid", "test"]
    for i in range(3):
        for csd in csds:
            print("Shape of the %s computational sequence for %s fold is %s" %
                  (csd, fold_names[i], output[i][csd].shape))
        print("Shape of the %s computational sequence for %s fold is %s" %
              ('words', fold_names[i], output[i]['words'].shape))
    return output


if __name__ == "__main__":

    raw_text, vids = get_rawtext(
        '../data/mosi/mosi.hdf5', 'hdf5', keys)
    print(raw_text[0])
    print(vids[0])
    # text_glove = glove_embeddings(raw_text, vids)
    # print(text_glove.shape)

    audio_video_text = get_audio_visual_text(
        csds, seq_len=seq_len, text_data=raw_text, vids=vids)
    print(len(audio_video_text))
    print(audio_video_text[0].keys())

    all_data = {}
    fold_names = ["train", "valid", "test"]
    key_sets = ['audio', 'vision', 'text', 'labels', 'id']

    for i, fold in enumerate(fold_names):
        all_data[fold] = {}
        all_data[fold]['vision'] = audio_video_text[i][VIDEO]
        all_data[fold]['audio'] = audio_video_text[i][AUDIO]
        all_data[fold]['text'] = audio_video_text[i]['words']
        all_data[fold]['labels'] = audio_video_text[i][labels[0]]
        all_data[fold]['id'] = audio_video_text[i]['id']

    with open('mosi_raw.pkl', 'wb') as f:
        pickle.dump(all_data, f)
