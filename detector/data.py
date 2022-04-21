#!/usr/bin/env python
"""XXX

Data is assumed to conform to the following format:

    data/
    ├── 0000
    │   ├── data_frame.AAAA.pkl
    │   ├── data_frame.BBBB.pkl
    │   ├── ...
    │   └── data_frame.CCCC.pkl
    ├── 0001
    │   ├── data_frame.DDDD.pkl
    │   ├── data_frame.EEEE.pkl
    │   ├── ...
    │   └── data_frame.FFFF.pkl
    ├── ...
    └── NNNN
        ├── data_frame.GGGG.pkl
        ├── data_frame.HHHH.pkl
        ├── ...
        └── data_frame.IIII.pkl

"""
import argparse
import glob
import logging
import os
import sys

import pandas as pd
import torch
from torch.utils.data import Dataset

import matplotlib.pyplot as plt
import numpy as np

import ontology


class TileData(Dataset):
    """XXX
    """
    def __init__(self, *, dirs, fraction_other, transform=None, target_transform=None):
        """XXX
        """
        self._transform = transform
        self._target_transform = target_transform

        # Load all data.
        df = pd.concat([pd.read_pickle(p)
                        for d in dirs for p in glob.glob(os.path.join(d, '*.pkl'))],
                       ignore_index=True)

        # Split into other/not other subsets
        df_tiles = df.loc[df['label'] != ontology.LABELS_TO_CLASSES['other']]
        df_other = df.loc[df['label'] == ontology.LABELS_TO_CLASSES['other']]

        # Drop some of the 'other' examples since they're way over represented.
        # Determine fraction of other labels that we should keep.
        #       fraction_other = n_other / (n_other + n_tiles)
        #       (1 - fraction_other) n_other = fraction_other n_tiles
        #       n_other = fraction_other n_tiles / (1 - fraction_other)
        n_other = int(fraction_other * len(df_other) / (1.0 - fraction_other))
        logging.info(f'should keep {n_other} of {len(df_other)} other samples')
        df_other_sampled = df_other.sample(n=n_other, random_state=0)

        # Combine
        self._df = pd.concat([df_tiles, df_other_sampled], ignore_index=True)

    def __len__(self):
        return len(self._df.index)

    def __getitem__(self, i):
        im = self._df.iloc[i]['tile'].reshape(64, 60, 3).transpose((2, 0, 1))
        x = torch.tensor(im / 255.0, dtype=torch.float)
        y = torch.tensor(self._df.iloc[i]['label'])

        if self._transform is not None:
            x = self._transform(x)
        if self._target_transform is not None:
            y = self._target_transform(y)
        return x, y


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--labelled', required=True, action='append',
                        help='path to labelled data dir')
    args = parser.parse_args()

    data = TileData(dirs=args.labelled, fraction_other=0.2)
    logging.info(f'loaded {len(data)} labelled samples')

    def _get_class(c):
        return data._df.loc[data._df['label'] == ontology.LABELS_TO_CLASSES[c], 'tile']

    MAX_SAMPLES = 30
    samples = {c: np.hstack([d.reshape(64, 60, 3) for d in _get_class(c)][:MAX_SAMPLES])
               for c in ontology.CLASSES if len(_get_class(c)) > 0}

    fig = plt.figure()

    for i, (k, v) in enumerate(samples.items()):
        ax = fig.add_subplot(len(samples), 1, i + 1)
        ax.imshow(v)
        ax.axis('off')

    plt.show()


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(message)s',
                        datefmt='%Y%m%d %H:%M:%S',
                        level=logging.INFO,
                        stream=sys.stdout)
    main()
