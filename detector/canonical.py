#!/usr/bin/env python
"""
XXX

Represent a 'canonical' board...
"""
import argparse
import json
import logging
import pickle
import sys

import cv2
import numpy as np


TILE_LEN = 15  # Number of tiles per grid edge on Scrabble board.


def _load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _read_corners(path):
    data = _load_json(path)
    return np.array([[p['x'], p['y']] for p in data], dtype=np.int32)


def _transform_homography(H, p):
    res = (H @ np.hstack([p, np.ones(p.shape[0])[:, None]]).T).T
    return (res[:, :2] / res[:, 2:]).astype(np.int32)


class DetectAndDescribe:
    """XXX
    """
    def __init__(self):
        self._sift = cv2.SIFT_create()

    def compute(self, *, im):
        """XXX
        """
        logging.info('detecting features...')
        return self._sift.detectAndCompute(im, None)


class Register:
    """XXX
    """
    def __init__(self, *, keypoints, descriptors):
        self._keypoints = keypoints
        self._descriptors = descriptors

        self._features = DetectAndDescribe()

        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)   # or pass empty dictionary

        self._flann = cv2.FlannBasedMatcher(
                {'algorithm': FLANN_INDEX_KDTREE, 'tress': 5},
                {'checks': 50})

    def compute(self, *, im):
        """XXX
        """
        logging.info('registering query image...')
        keypoints, descriptors = self._features.compute(im=im)

        matches = self._flann.knnMatch(self._descriptors, descriptors, k=2)
        good = [m for (m, n) in matches if m.distance < 0.7 * n.distance]

        p_Ccanon_C2d = np.float32([self._keypoints[m.queryIdx].pt for m in good]).reshape(-1,1,2)
        p_Cquery_C2d = np.float32([keypoints[m.trainIdx].pt for m in good]).reshape(-1,1,2)

        H_CI, _ = cv2.findHomography(p_Cquery_C2d, p_Ccanon_C2d, cv2.RANSAC, 5.0)
        return H_CI


class CanonicalBoard:
    """
    Represents a 'canonical' board to which query images can then be matched and
    transformed. The corresponding tile data can then be
    extracted.

    Params
    -------
    keypoints : keypoints
    """
    TILE_XLEN = 64  # XXX px maybe shouldn't be static param
    TILE_YLEN = 60  # XXX px maybe shouldn't be static param

    def __init__(self, *, edge_len_px, xbounds, ybounds, keypoints, descriptors):
        self._edge_len_px = edge_len_px

        self._tile_xlen_px = (xbounds[1] - xbounds[0]) / TILE_LEN
        self._tile_ylen_px = (ybounds[1] - ybounds[0]) / TILE_LEN
        self._xindices = [int(xbounds[0] + i * self._tile_xlen_px)
                          for i in range(TILE_LEN)]
        self._yindices = [int(ybounds[0] + i * self._tile_ylen_px)
                          for i in range(TILE_LEN)]
        self._xbounds = xbounds
        self._ybounds = ybounds

        self._register = Register(keypoints=keypoints, descriptors=descriptors)

    @classmethod
    def create_from_data(cls, *, image_path,
                         edge_corners_path, grid_corners_path,
                         edge_len_px):
        """XXX
        """
        logging.info('creating canonical board from data')
        # Let the I frame indicate the input image frame.
        # Note that the order of corners is important here...
        p_Izedge_I2d = _read_corners(edge_corners_path)
        p_Izgrid_I2d = _read_corners(grid_corners_path)

        # Let the C frame indicate the canonical image frame.
        p_Cedge_C2d = np.array([[0,           0],
                                [edge_len_px, 0],
                                [edge_len_px, edge_len_px],
                                [0,           edge_len_px]], dtype=np.int32)

        im_I = cv2.imread(image_path)
        im_I = cv2.cvtColor(im_I, cv2.COLOR_BGR2RGB)

        H_CI, _ = cv2.findHomography(p_Izedge_I2d, p_Cedge_C2d)
        im_C =  cv2.warpPerspective(im_I, H_CI, dsize=(edge_len_px, edge_len_px))

        p_Czgrid_C2d = _transform_homography(H_CI, p_Izgrid_I2d)

        # Note: again, order is important...
        xmin = 0.5 * (p_Czgrid_C2d[0, 0] + p_Czgrid_C2d[3, 0])
        xmax = 0.5 * (p_Czgrid_C2d[1, 0] + p_Czgrid_C2d[2, 0])
        ymin = 0.5 * (p_Czgrid_C2d[0, 1] + p_Czgrid_C2d[1, 1])
        ymax = 0.5 * (p_Czgrid_C2d[2, 1] + p_Czgrid_C2d[3, 1])

        features = DetectAndDescribe()
        keypoints, descriptors = features.compute(im=im_C)

        ret = cls(edge_len_px=edge_len_px,
                  xbounds=(xmin, xmax),
                  ybounds=(ymin, ymax),
                  keypoints=keypoints,
                  descriptors=descriptors)
        return ret

    @classmethod
    def create_from_disk(cls, *, path):
        """XXX
        """
        with open(path, 'rb') as f:
            d = pickle.load(f)

        ret = cls(edge_len_px=d['edge_len_px'],
                  xbounds=d['xbounds'],
                  ybounds=d['ybounds'],
                  keypoints=[cv2.KeyPoint(**kwargs)
                             for kwargs in d['keypoints']],
                  descriptors=d['descriptors'])
        return ret


    def write_to_disk(self, *, path):
        """XXX
        """
        with open(path, 'wb') as f:
            pickle.dump({'edge_len_px': self._edge_len_px,
                         'xbounds': self._xbounds,
                         'ybounds': self._ybounds,
                         'keypoints': [{'x': k.pt[0],
                                        'y': k.pt[1],
                                        'angle': k.angle,
                                        'class_id': k.class_id,
                                        'octave': k.octave,
                                        'response': k.response,
                                        'size': k.size}
                                        for k in self._register._keypoints],
                         'descriptors': self._register._descriptors,
                         }, f)

    def partition(self, *, image_path):
        """XXX
        """
        logging.info('reading query image...')
        im_I = cv2.imread(image_path)
        im_I = cv2.cvtColor(im_I, cv2.COLOR_BGR2RGB)

        H_CI = self._register.compute(im=im_I)
        im_C = cv2.warpPerspective(im_I, H_CI, dsize=(self._edge_len_px, self._edge_len_px))

        logging.info('partitioning tiles...')
        def _tile(xi, yi):
            t = im_C[yi:yi + int(self._tile_ylen_px),
                     xi:xi + int(self._tile_xlen_px),
                     :]
            return cv2.resize(t,dsize=(CanonicalBoard.TILE_YLEN, CanonicalBoard.TILE_XLEN))

        return [_tile(xi, yi) for yi in self._yindices for xi in self._xindices]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', required=True, help='path to image')
    parser.add_argument('--edge', required=True, help='edge corner annotation json')
    parser.add_argument('--grid', required=True, help='grid corner annotation json')
    parser.add_argument('--output', default='canonical.pkl', help='output path')
    args = parser.parse_args()

    # Create canonical board and write to disk.
    board = CanonicalBoard.create_from_data(image_path=args.image,
                                            edge_corners_path=args.edge,
                                            grid_corners_path=args.grid,
                                            edge_len_px=1875)
    logging.info('writing to disk')
    board.write_to_disk(path=args.output)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(message)s',
                        datefmt='%Y%m%d %H:%M:%S',
                        level=logging.INFO,
                        stream=sys.stdout)
    main()
