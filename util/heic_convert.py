#!/usr/bin/env python
"""
Convert all .HEIC images in a directory to .png.
"""
import argparse
import glob
import os
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dir', required=True, help='path to image directory')
    args = parser.parse_args()

    for pin in glob.glob(os.path.join(args.dir, '*.HEIC')):
        pout = os.path.join(args.dir,
                            f'{os.path.splitext(os.path.basename(pin))[0]}.png')
        if os.path.exists(pout):
            print(f'already converted {pin}')
            continue

        ret = subprocess.run(['heif-convert', pin, pout],
                             capture_output=True)
        print(ret.stdout)
        if ret.returncode == 1:
            print(ret.stderr)
            break


if __name__ == '__main__':
    sys.exit(main())
