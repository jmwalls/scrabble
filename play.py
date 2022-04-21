#!/usr/bin/env python
"""XXX
"""
import argparse
import json

from game import engine


def main():
    # XXX add mutually exclusive groups:
    #   (1) set num players for new game
    #   (2) load game from disk
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()

    history = []
    while True
        try:
            move = {'player': 0,
                    'letters': 'foo',
                    'pos': 'e7',
                    orientation = engine.Orientation.HORIZONTAL}
            board.play(**move)
            history.append(move)
            # write history...
        except ValueError:
            pass



if __name__ == '__main__':
    main()
