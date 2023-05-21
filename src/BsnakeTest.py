import unittest

from Bsnake import *

class BsnakeTest(unittest.TestCase):
    def test_Bsnake(self):
        bsnake = Bsnake.init_default()

        self.assertEqual(bsnake.min_threshold_canny, 50)
        self.assertEqual(bsnake.max_threshold_canny, 100)
        self.assertEqual(bsnake.process_slices, 15)
        self.assertEqual(bsnake.hough_threshold, 15)
        self.assertEqual(bsnake.min_line_length, 10)
        self.assertEqual(bsnake.max_line_gap, 20)
        self.assertEqual(bsnake.median_blur_ksize, 9)


if __name__ == '__main__':
    unittest.main()