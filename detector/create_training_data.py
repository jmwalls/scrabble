#!/usr/bin/env python
"""
XXX

Generate labelled tiles...
"""
import argparse
import glob
import logging
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tqdm

from canonical import CanonicalBoard
import ontology


def read_board_labels(*, path):
    """XXX
    """
    df = pd.read_csv(path, header=None)
    assert df.shape == (15, 15)

    # Replace board '.' with 'other' class key.
    df[df == '.'] = 'other'

    return [i for i in map(lambda k: ontology.LABELS_TO_CLASSES[k], df.to_numpy().flatten())]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--canonical', required=True,
                        help='path to serialized canonical board')
    parser.add_argument('--dir', required=True, help='path to image directory')
    parser.add_argument('--labels', required=True, help='path to board label csv')
    args = parser.parse_args()

    labels = read_board_labels(path=args.labels)
    board = CanonicalBoard.create_from_disk(path=args.canonical)

    for p in glob.glob(os.path.join(args.dir, '*.png')):
        logging.info(f'partitioning {p}')
        tiles = board.partition(image_path=p)
        assert len(tiles) == len(labels)

        logging.info('creating data frame')
        records = [{'label': labels[i],
                    'tile': t.flatten()} for i, t in enumerate(tiles)]
        df = pd.DataFrame.from_records(records)
        # XXX out file path...
        df.to_pickle('foo.pkl')


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(message)s',
                        datefmt='%Y%m%d %H:%M:%S',
                        level=logging.INFO,
                        stream=sys.stdout)
    sys.exit(main())
