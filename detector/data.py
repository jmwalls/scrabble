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
import glob
import logging
import os

import pandas as pd
import torch
from torch.utils.data import Dataset

import ontology


class TileData(Dataset):
    """XXX
    """
    def __init__(self, *, path, transform=None, target_transform=None):
        """XXX
        """
        self._transform = transform
        self._target_transform = target_transform

        # Load all data.
        df = pd.concat([pd.read_pickle(p)
                        for p in glob.glob(os.path.join(path, '*/*.pkl'))],
                       ignore_index=True)

        # Split into other/not other subsets
        df_tiles = df.loc[df['label'] != ontology.LABELS_TO_CLASSES['other']]
        df_other = df.loc[df['label'] == ontology.LABELS_TO_CLASSES['other']]

        # Drop some of the 'other' examples since they're way over represented.
        # XXX should check that they're actually way over represented...
        df_other_sampled = df_other.sample(frac=0.1,
                                           random_state=0)

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
