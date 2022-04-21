#!/usr/bin/env python
import unittest

import numpy as np  # XXX

import engine


class TestBoardBasics(unittest.TestCase):
    def test_num_players(self):
        """At least one player must play."""
        try:
            dut = engine.Board(num_players = 0)
            self.assertTrue(False)
        except ValueError:
            self.assertTrue(True)

    def test_wrong_player(self):
        """Play should proceed in order of players."""
        dut = engine.Board(num_players = 2)
        try:
            dut.play(player=1, letters='foo', pos=(3, 2))
            self.assertTrue(False)
        except Exception:
            self.assertTrue(True)

    def test_first_word(self):
        """First word should include center tile."""
        dut = engine.Board(num_players = 2)
        try:
            dut.play(player=0,
                     letters='start',
                     pos='b2',
                     orientation=engine.Orientation.HORIZONTAL)
            self.assertTrue(False)
        except Exception:
            self.assertTrue(True)

    def test_already_set(self):
        """Check that we won't update a letter that's already been set."""
        self.assertTrue(False)


class TestBoardIndexing(unittest.TestCase):
    def test_basic_horizontal(self):
        ret = engine._index(row=1,
                            col=1,
                            num_letters=3,
                            orientation=engine.Orientation.HORIZONTAL)
        self.assertEqual(len(ret), 3)
        self.assertEqual(ret, [16, 17, 18])

    def test_basic_vertical(self):
        ret = engine._index(row=1,
                            col=1,
                            num_letters=3,
                            orientation=engine.Orientation.VERTICAL)
        self.assertEqual(len(ret), 3)
        self.assertEqual(ret, [16, 31, 46])


class TestBoardScoring(unittest.TestCase):
    def test_used_letter_multiplier(self):
        """Check that we won't re-score a letter multiplier."""
        self.assertTrue(False)

    def test_used_word_multiplier(self):
        """Check that we won't re-score a word multiplier."""
        self.assertTrue(False)

    def test_basic(self):
        """XXX"""
        print('playing...')
        dut = engine.Board(num_players=2)
        print(dut)

        dut.play(player=0,
                 letters='start',
                 pos='g8',
                 orientation=engine.Orientation.HORIZONTAL)
        print(dut)

        dut.play(player=1,
                 letters='test',
                 pos='k8',
                 orientation=engine.Orientation.VERTICAL)
        print(dut)

        dut.play(player=0,
                 letters='arch',
                 pos='k7',
                 orientation=engine.Orientation.HORIZONTAL)
        print(dut)


class TestWords(unittest.TestCase):
    def test_basic(self):
        """XXX"""
        w1 = engine.Word(letters='abc',
                         pos='h8',
                         orientation=engine.Orientation.HORIZONTAL)
        w2 = engine.Word(letters='cde',
                         pos='j8',
                         orientation=engine.Orientation.VERTICAL)
        self.assertTrue(False)

    def test_merge(self):
        """XXX"""
        w1 = engine.Word(letters='abc',
                         pos='h8',
                         orientation=engine.Orientation.HORIZONTAL)
        w2 = engine.Word(letters='cde',
                         pos='j8',
                         orientation=engine.Orientation.VERTICAL)
        w3 = engine.Word(letters='de',
                         pos='j9',
                         orientation=engine.Orientation.VERTICAL)
        # w1/w2 should have same result as w1/w3.
        self.assertTrue(False)


if __name__ == '__main__':
    unittest.main()
