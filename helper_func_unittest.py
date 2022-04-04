#!/usr/bin/env python3
"""
Provides the helper functions unittests

It contains the unittests for checking the correct work of the helper functions
"""

import unittest
from helper_func import *

__author__ = "Jose Ignacio de Alvear Cardenas"
__copyright__ = "Copyright 2022, Jose Ignacio de Alvear Cardenas"
__credits__ = ["Jose Ignacio de Alvear Cardenas"]
__license__ = "MIT"
__version__ = "1.0.1 (04/04/2022)"
__maintainer__ = "Jose Ignacio de Alvear Cardenas"
__email__ = "j.i.dealvearcardenas@student.tudelft.nl"
__status__ = "Development"


class HelperFunc(unittest.TestCase):
    bc = 5
    tc = 3
    h = 4
    h0 = 3
    pos = 5
    chords = [6, 1, 2]
    hs = [2, 2]
    n_segments = 4

    def test_trapezoid_params(self):
        """
        Testing the function that computes the trapezoid parameters
        :return:
        """
        area, y_bar = trapezoid_params(self.bc, self.tc, self.h)
        area_result = 16
        y_bar_result = 11/6
        self.assertEqual(area, area_result)
        self.assertEqual(y_bar, y_bar_result)

    def test_compute_trapezoid_area(self):
        """
        Test that the trapezoid area is computed correctly
        :return:
        """
        self.assertEqual(compute_trapezoid_area(self.bc, self.tc, self.h), 16, "The correct answer is 16.")

    def test_compute_trapezoid_cg(self):
        """
        Test that the cg location of a trapezoid along its length is computed correctly.
        :return:
        """
        self.assertEqual(compute_trapezoid_cg(self.bc, self.tc, self.h), 11/6)

    def test_compute_chord_blade(self):
        """
        Test method that computes the chord length at a certain position within the blade given all the trapezoids that
        shape the blade
        :return:
        """
        pos = 3
        self.assertEqual(compute_chord_blade(self.chords, self.hs, pos), 1.5)

    def test_compute_chord_trapezoid(self):
        """
        Test that the chord within a trapezoid is computed correctly.
        :return:
        """
        self.assertEqual(compute_chord_trapezoid(self.bc, self.tc, self.h, self.h0, self.pos), 4)

    def test_compute_average_chord(self):
        """
        Test the computation of the average chord given a starting and final position within the blade
        :return:
        """
        pos_start = 2
        pos_end = 3
        self.assertEqual(compute_average_chord(self.chords, self.hs, pos_start, pos_end), 1.25)

    def test_compute_average_chords(self):
        """
        Test that the average chords of a trapezoidal blade are computed correctly
        :return:
        """
        average_chords, segment_chords = compute_average_chords(self.chords, self.hs, self.n_segments)
        average_chords_result = [4.75, 2.25, 1.25, 1.75]
        segment_chords_result = [6, 3.5, 1, 1.5, 2]
        self.assertEqual(average_chords, average_chords_result)
        self.assertEqual(segment_chords, segment_chords_result)


if __name__ == '__main__':
    unittest.main()
