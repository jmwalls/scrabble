"""XXX
"""
from collections import defaultdict
from enum import Enum
from functools import reduce
from operator import mul
import string

import numpy as np


LETTER_SCORES = {'a': 1,
                 'b': 3,
                 'c': 3,
                 'd': 2,
                 'e': 1,
                 'f': 4,
                 'g': 2,
                 'h': 4,
                 'i': 1,
                 'j': 8,
                 'k': 5,
                 'l': 1,
                 'm': 3,
                 'n': 1,
                 'o': 1,
                 'p': 3,
                 'q': 10,
                 'r': 1,
                 's': 1,
                 't': 1,
                 'u': 1,
                 'v': 4,
                 'w': 4,
                 'x': 8,
                 'y': 4,
                 'z': 10}

LETTER_PREMIUM = [1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1,
                  1, 1, 1, 1, 1, 3, 1, 1, 1, 3, 1, 1, 1, 1, 1,
                  1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1,
                  2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2,
                  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                  1, 3, 1, 1, 1, 3, 1, 1, 1, 3, 1, 1, 1, 3, 1,
                  1, 1, 2, 1, 1, 1, 2, 1, 2, 1, 1, 1, 2, 1, 1,
                  1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1,
                  1, 1, 2, 1, 1, 1, 2, 1, 2, 1, 1, 1, 2, 1, 1,
                  1, 3, 1, 1, 1, 3, 1, 1, 1, 3, 1, 1, 1, 3, 1,
                  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                  2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2,
                  1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1,
                  1, 1, 1, 1, 1, 3, 1, 1, 1, 3, 1, 1, 1, 1, 1,
                  1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1]

WORD_PREMIUM = [3, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 3,
                1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1,
                1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1,
                1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1,
                1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                3, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 3,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1,
                1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1,
                1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1,
                1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1,
                3, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 3]

COLS = list(string.ascii_uppercase[:15])
ROWS = [str(i + 1) for i in range(15)]


class Orientation(Enum):
    HORIZONTAL = 1
    VERTICAL = 2


def pos_to_rc(*, pos):
    """XXX
    """
    # XXX bounds checking
    return dict(row=ROWS.index(pos[1:]),
                col=COLS.index(pos[0].upper()))


def rc_to_index(*, row, col):
    """XXX
    """
    return 15 * row + col


def index_to_rc(*, index):
    """XXX
    """
    return dict(row=int(index / 15),
                col=index % 15)

def index_to_pos(*, index):
    row, col = index_to_rc(index=index).values()
    return COLS[col] + ROWS[row]


def _index(*, row, col, num_letters, orientation):
    """XXX
    """
    # XXX check for valid bounds
    if orientation == Orientation.HORIZONTAL:
        index = [rc_to_index(row=row, col=col + c) for c in range(num_letters)]
    elif orientation == Orientation.VERTICAL:
        index = [rc_to_index(row=row + r, col=col) for r in range(num_letters)]
    return index


class Word:
    """XXX
    """
    def __init__(self, *, letters, pos, orientation):
        assert isinstance(orientation, Orientation)
        self.pos = pos
        self.letters = letters
        self.orientation = orientation

    @property
    def index(self):
        """XXX
        """
        return _index(**pos_to_rc(pos=self.pos),
                      num_letters=len(self.letters),
                      orientation=self.orientation)

    def neighbors(self):
        row, col = pos_to_rc(pos=self.pos).values()

        def _inbounds(i):
            return i < 15 and i >= 0

        num_letters = len(self.letters)
        if self.orientation == Orientation.HORIZONTAL:
            # Add all the up/down neighbors.
            rows = [row + i for i in [1, -1] if _inbounds(row + i)]
            neighbors = [[rc_to_index(row=r, col=col + c) for r in rows] for c in range(num_letters)]

            # Add the left/right neighbors.
            neighbors.extend([[rc_to_index(row=row, col=col + i)
                               for i in [-1, num_letters] if _inbounds(col + i)]])
        elif self.orientation == Orientation.VERTICAL:
            # Add all the left/right neighbors.
            cols = [col + i for i in [1, -1] if _inbounds(col + i)]
            neighbors = [[rc_to_index(row=row + r, col=c) for c in cols] for r in range(num_letters)]

            # Add the up/down neighbors.
            neighbors.extend([[rc_to_index(row=row + i, col=col)
                               for i in [-1, num_letters] if _inbounds(row + i)]])
        return neighbors

    def __repr__(self):
        return self.letters


class Board:
    """XXX

    Note: users are expected to save history if they so desire, either for
    analysis or to be able to restart from a partial game.

    @param _mask: premium square mask
    """
    def __init__(self, *, num_players):
        if num_players < 1:
            raise ValueError('must have at least 1 player!')

        self._score = {i: 0 for i in range(num_players)}
        self._turns = {i: 0 for i in range(num_players)}

        self._num_players = num_players
        self._player = 0

        self._mask = [False for c in range(15) for r in range(15)]
        self._words = []

    def play(self, *, player, letters, pos, orientation):
        if player != self._player:
            raise Exception('not current player...')

        # XXX what if we stored words and their indices?
        #   * easier to look for intersections?
        #   * easier to validate a legal move?
        #   * easier to find additional words?
        # a new set of letters can:
        #   * add new words
        #   * modify existing words
        candidate = Word(letters=letters.lower(), pos=pos, orientation=orientation)

        # H8 must be played on first move.
        if sum(self._turns.values()) == 0 and \
                rc_to_index(**pos_to_rc(pos='h8')) not in candidate.index:
            raise Exception('h8 must be played on first turn!')

        # Make sure we're not changing a letter.
        board_letters = {i: l for w in self._words
                         for i, l in zip(w.index, w.letters)}
        for i, l in zip(candidate.index, candidate.letters):
            if i in board_letters and l != board_letters[i]:
                r, c = index_to_rc(index=i).values()
                raise Exception(f'Letter already set for {COLS[c]}{ROWS[r]}')

        # Determine new words from candidate.
        board_words = defaultdict(list)
        for w in self._words:
            for i in w.index:
                board_words[i].append(w)
        print(board_words)
        print(board_letters)

        # Find neighbors/merge candidate
        print(candidate.index)
        neighbors = candidate.neighbors()
        for n in neighbors:
            # print([index_to_pos(index=ni) for ni in n])
            for ni in n:
                if ni in board_words:
                    print(f'neighbor at {index_to_pos(index=ni)}')

        self._words.append(candidate)

        # XXX loop over all words...
        score = self._score_word(word=candidate)

        # Update premium square mask.
        for i in candidate.index:
            self._mask[i] = True

        self._score[player] += score
        self._turns[player] += 1

        self._player += 1
        self._player %= self._num_players

    def _score_word(self, *, word):
        word_multiplier = [max(1, (1 - self._mask[i]) * WORD_PREMIUM[i])
                           for i in word.index]
        score = sum([max(1, (1 - self._mask[i]) * LETTER_PREMIUM[i]) * LETTER_SCORES[l]
                     for i, l in zip(word.index, word.letters)])
        return reduce(mul, word_multiplier, 1) * score

    def __str__(self):
        # XXX can create board on the fly from the list of words...
        board = ['.' for c in range(15) for r in range(15)]
        for w in self._words:
            for i, l in zip(w.index, w.letters):
                board[i] = l

        scores = '\n'.join([f'Player {i}: {self._score[i]}'
                           for i in range(self._num_players)])
        header = '   {}'.format(' '.join(COLS))
        def _line(j):
            return '{:2s} {}'.format(
                    ROWS[j],
                    ' '.join([f'{l}' for l in board[15 * j:15 * (j + 1)]]))
        return '\n'.join([scores, header] + [_line(i) for i in range(15)])
