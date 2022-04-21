#!/usr/bin/env python
import unittest

import engine

# XXX make sure we're catching the *right* exceptions

class TestBoard(unittest.TestCase):
    def test_first_word(self):
        """First word should return itself."""
        print('foo')
        dut = engine.Board()
        ret = dut.add(letters='start',
                      pos='d8',
                      orientation=engine.Orientation.HORIZONTAL)
        self.assertEqual(len(ret), 1)
        self.assertTrue('start' in ret)

    def test_already_set(self):
        """Check that we won't update a letter that's already been set."""
        dut = engine.Board()
        dut.add(letters='start',
                pos='d8',
                orientation=engine.Orientation.HORIZONTAL)
        try:
            dut.add(letters='start',
                    pos='d8',
                    orientation=engine.Orientation.VERTICAL)
            self.assertTrue(False)
        except Exception:
            self.assertTrue(True)

    def test_vertical(self):
        """Detect new word added vertically."""
        dut = engine.Board()
        dut.add(letters='start',
                pos='d8',
                orientation=engine.Orientation.HORIZONTAL)
        ret = dut.add(letters='tale',
                      pos='h8',
                      orientation=engine.Orientation.VERTICAL)
        self.assertEqual(len(ret), 1)
        self.assertTrue('tale' in [r[0] for r in ret])

    def test_vertical_alt(self):
        """Detect new word added vertically expressed as added letters only."""
        dut = engine.Board()
        dut.add(letters='start',
                pos='d8',
                orientation=engine.Orientation.HORIZONTAL)
        ret = dut.add(letters='ale',
                      pos='h9',
                      orientation=engine.Orientation.VERTICAL)
        self.assertEqual(len(ret), 1)
        self.assertTrue('tale' in ret)

    def test_vertical_append(self):
        """Detect new words added vertically."""
        dut = engine.Board()
        dut.add(letters='start',
                pos='d8',
                orientation=engine.Orientation.HORIZONTAL)
        ret = dut.add(letters='safe',
                      pos='i8',
                      orientation=engine.Orientation.VERTICAL)
        self.assertEqual(len(ret), 2)
        self.assertTrue('starts' in ret)
        self.assertTrue('safe' in ret)

    def test_vertical_cross_append(self):
        """Detect new words added vertically."""
        dut = engine.Board()
        dut.add(letters='start',
                pos='d8',
                orientation=engine.Orientation.HORIZONTAL)
        ret = dut.add(letters='base',
                      pos='i6',
                      orientation=engine.Orientation.VERTICAL)
        self.assertEqual(len(ret), 2)
        self.assertTrue('starts' in ret)
        self.assertTrue('base' in ret)

    def test_vertical_cross(self):
        """Detect new words added vertically intersecting existing."""
        dut = engine.Board()
        dut.add(letters='start',
                pos='d8',
                orientation=engine.Orientation.HORIZONTAL)
        ret = dut.add(letters='score',
                      pos='g5',
                      orientation=engine.Orientation.VERTICAL)
        self.assertEqual(len(ret), 1)
        self.assertTrue('score' in ret)

    def test_horizontal(self):
        """Detect new word added vertically."""
        dut = engine.Board()
        dut.add(letters='start',
                pos='h6',
                orientation=engine.Orientation.VERTICAL)
        ret = dut.add(letters='tale',
                      pos='h10',
                      orientation=engine.Orientation.HORIZONTAL)
        self.assertEqual(len(ret), 1)
        self.assertTrue('tale' in ret)

    def test_horizontal_alt(self):
        """Detect new word added horizontally expressed as added letters only."""
        dut = engine.Board()
        dut.add(letters='start',
                pos='h6',
                orientation=engine.Orientation.VERTICAL)
        ret = dut.add(letters='ale',
                      pos='i10',
                      orientation=engine.Orientation.HORIZONTAL)
        self.assertEqual(len(ret), 1)
        self.assertTrue('tale' in ret)

    def test_horizontal_append(self):
        """Detect new words added horizontally."""
        dut = engine.Board()
        dut.add(letters='start',
                pos='h6',
                orientation=engine.Orientation.VERTICAL)
        ret = dut.add(letters='safe',
                      pos='h11',
                      orientation=engine.Orientation.HORIZONTAL)
        self.assertEqual(len(ret), 2)
        self.assertTrue('starts' in ret)
        self.assertTrue('safe' in ret)

    def test_horizontal_cross_append(self):
        """Detect new words added horizontally."""
        dut = engine.Board()
        dut.add(letters='start',
                pos='h6',
                orientation=engine.Orientation.VERTICAL)
        ret = dut.add(letters='base',
                      pos='f11',
                      orientation=engine.Orientation.HORIZONTAL)
        self.assertEqual(len(ret), 2)
        self.assertTrue('starts' in ret)
        self.assertTrue('base' in ret)

    def test_horizontal_cross(self):
        """Detect new words added horizontally intersecting existing."""
        dut = engine.Board()
        dut.add(letters='start',
                pos='h6',
                orientation=engine.Orientation.VERTICAL)
        ret = dut.add(letters='score',
                      pos='e9',
                      orientation=engine.Orientation.HORIZONTAL)
        self.assertEqual(len(ret), 1)
        self.assertTrue('score' in ret)


class TestGameBasics(unittest.TestCase):
    def test_num_players(self):
        """At least one player must play."""
        try:
            dut = engine.Game(num_players = 0)
            self.assertTrue(False)
        except ValueError:
            self.assertTrue(True)

    def test_wrong_player(self):
        """Play should proceed in order of players."""
        dut = engine.Game(num_players = 2)
        try:
            dut.play(player=1,
                     letters='start',
                     pos='d8',
                     orientation=engine.Orientation.HORIZONTAL)
            self.assertTrue(False)
        except Exception:
            self.assertTrue(True)

    def test_first_word(self):
        """First word should include center tile."""
        dut = engine.Game(num_players = 2)
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
        dut = engine.Game(num_players = 2)
        dut.play(player=0,
                 letters='start',
                 pos='d8',
                 orientation=engine.Orientation.HORIZONTAL)
        try:
            dut.play(player=1,
                     letters='start',
                     pos='d8',
                     orientation=engine.Orientation.VERTICAL)
            self.assertTrue(False)
        except Exception:
            self.assertTrue(True)


class TestBoardIndexing(unittest.TestCase):
    def test_basic_horizontal(self):
        ret = engine._indices(row=1,
                              col=1,
                              num_letters=3,
                              orientation=engine.Orientation.HORIZONTAL)
        self.assertEqual(len(ret), 3)
        self.assertEqual(ret, [16, 17, 18])

    def test_basic_vertical(self):
        ret = engine._indices(row=1,
                              col=1,
                              num_letters=3,
                              orientation=engine.Orientation.VERTICAL)
        self.assertEqual(len(ret), 3)
        self.assertEqual(ret, [16, 31, 46])


class TestGame(unittest.TestCase):
    def test_used_letter_multiplier(self):
        """Check that we won't re-score a letter multiplier."""
        self.assertTrue(False)

    def test_used_word_multiplier(self):
        """Check that we won't re-score a word multiplier."""
        self.assertTrue(False)

    def test_basic(self):
        """XXX"""
        print('playing...')
        dut = engine.Game(num_players=2)
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


if __name__ == '__main__':
    unittest.main()
