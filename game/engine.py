"""XXX
"""
from collections import defaultdict
from enum import Enum
from functools import reduce
from operator import mul
import string


# Each letter point value.
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


# Letter premium multipliers per square.
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


# Word premium multipliers per square.
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


def _indices(*, row, col, num_letters, orientation):
    """Return board indices for a set of letters.

    @param row: start row index of word.
    @param col: start col index of word.
    @param num_letters: length of letter set.
    @orientation: letter set orientation.
    """
    # XXX check for valid bounds
    if orientation == Orientation.HORIZONTAL:
        index = [rc_to_index(row=row, col=col + c) for c in range(num_letters)]
    elif orientation == Orientation.VERTICAL:
        index = [rc_to_index(row=row + r, col=col) for r in range(num_letters)]
    return index


def _group(indices):
    # XXX
    splits = [i for i, (i1, i2) in enumerate(zip(indices, indices[1:])) if i2 != i1 + 1]
    splits = [0] + splits + [len(indices)]
    return [indices[s1:s2] for s1, s2 in zip(splits, splits[1:])]

class Board:
    """XXX

    @param letters: dict mapping board index to letter.
    """
    def __init__(self):
        self._letters = {}

    def add(self, *, letters, pos, orientation):
        """XXX

        @param letters: letter set (can contain letters already on the board...)
        @param pos: start position of the letter set.
        @param orientation: orientation of the letter set.

        @return XXX
        # XXX return indices and words
        """
        # XXX check that indices are in bounds or should check upstream?
        row, col = pos_to_rc(pos=pos).values()
        indices = _indices(row=row,
                           col=col,
                           num_letters=len(letters),
                           orientation=orientation)

        # Check that we're not resetting a tile.
        for k, v in zip(indices, letters):
            if k in self._letters and v != self._letters[k]:
                r, c = index_to_rc(index=k).values()
                raise Exception(f'Letter already set for {COLS[c]}{ROWS[r]}')

        added_indices = [k for k in indices if k not in self._letters]
        first_word = len(self._letters) == 0

        for k, v in zip(indices, letters):
            self._letters[k] = v

        if first_word:
            return [(letters, indices)]

        def _has_letter(i):
            return i in self._letters

        # Check along the axis that the word was added...
        # XXX would be nice to have a cleaner way to do this instead of copying
        # functionality for vertical/horizontal...
        words = []
        if orientation == Orientation.VERTICAL:
            line_indices = [i for i in range(15) if _has_letter(rc_to_index(row=i, col=col))]
            assert len(line_indices) > 0

            for group in _group(line_indices):
                if row in group:
                    word = ''.join([self._letters[rc_to_index(row=i, col=col)] for i in group])
                    words.append((word, group))
        elif orientation == Orientation.HORIZONTAL:
            line_indices = [i for i in range(15) if _has_letter(rc_to_index(row=row, col=i))]
            assert len(line_indices) > 0

            for group in _group(line_indices):
                if col in group:
                    word = ''.join([self._letters[rc_to_index(row=row, col=i)] for i in group])
                    words.append((word, group))

        # Check along the off-axiss for each letter that was added...
        if orientation == Orientation.VERTICAL:
            rows = [row + i for i in range(len(letters))]
            for r in rows:
                line_indices = [i for i in range(15) if _has_letter(rc_to_index(row=r, col=i))]
                assert len(line_indices) > 0

                for group in _group(line_indices):
                    if len(group) > 1 and col in group:
                        word_indices = [rc_to_index(row=r, col=i) for i in group]
                        if any([i in added_indices for i in word_indices]):
                            words.append((''.join([self._letters[k] for k in word_indices]),
                                          word_indices))
        elif orientation == Orientation.HORIZONTAL:
            cols = [col + i for i in range(len(letters))]
            for c in cols:
                line_indices = [i for i in range(15) if _has_letter(rc_to_index(row=i, col=c))]
                assert len(line_indices) > 0

                for group in _group(line_indices):
                    if len(group) > 1 and row in group:
                        word_indices = [rc_to_index(row=i, col=c) for i in group]
                        if any([i in added_indices for i in word_indices]):
                            words.append((''.join([self._letters[k] for k in word_indices]),
                                          word_indices))
        return words

    def __str__(self):
        board = ['.' for c in range(15) for r in range(15)]
        for k, v in self._letters.items():
            board[k] = v

        header = '   {}'.format(' '.join(COLS))
        def _line(j):
            return '{:2s} {}'.format(
                    ROWS[j],
                    ' '.join([f'{l}' for l in board[15 * j:15 * (j + 1)]]))
        return '\n'.join([header] + [_line(i) for i in range(15)])


class Game:
    """XXX

    Note: users are expected to save history if they so desire, either for
    analysis or to be able to restart from a partial game.

    @param _num_players: total number of players (immutable)
    @param _player: current player index

    @param _score: dict mapping player index to current score
    @param _turns: dict mapping player index to number of turns played

    @param _mask: track if a premium square has been used
    """
    def __init__(self, *, num_players):
        if num_players < 1:
            raise ValueError('must have at least 1 player!')

        self._num_players = num_players
        self._player = 0

        self._score = {i: 0 for i in range(num_players)}
        self._turns = {i: 0 for i in range(num_players)}

        self._mask = [False for c in range(15) for r in range(15)]
        self._board = Board()

    def play(self, *, player, letters, pos, orientation):
        if player != self._player:
            raise Exception('not current player...')

        # H8 must be played on first move.
        indices = _indices(**pos_to_rc(pos=pos),
                           num_letters=len(letters),
                           orientation=orientation)
        if sum(self._turns.values()) == 0 and \
                rc_to_index(**pos_to_rc(pos='h8')) not in indices:
            raise Exception('h8 must be played on first turn!')

        # XXX determine words and score
        words = self._board.add(letters=letters.lower(), pos=pos, orientation=orientation)
        # score = self._score_word(word=candidate)
        # # Update premium square mask.
        # for i in candidate.index:
        #     self._mask[i] = True

        # self._score[player] += score
        self._turns[player] += 1

        self._player += 1
        self._player %= self._num_players

    def _score_word(self, *, word):
        """XXX
        """
        return
        word_multiplier = [max(1, (1 - self._mask[i]) * WORD_PREMIUM[i])
                           for i in word.index]
        score = sum([max(1, (1 - self._mask[i]) * LETTER_PREMIUM[i]) * LETTER_SCORES[l]
                     for i, l in zip(word.index, word.letters)])
        return reduce(mul, word_multiplier, 1) * score

    def __str__(self):
        """XXX
        """
        scores = '\n'.join([f'Player {i}: {self._score[i]}'
                            for i in range(self._num_players)])
        return '\n'.join([scores, str(self._board)])
