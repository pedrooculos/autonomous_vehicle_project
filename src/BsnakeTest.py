import unittest

from Bsnake import *

class BsnakeTest(unittest.TestCase):
    def test_Bsnake_default_init(self):
        bsnake = Bsnake.init_default()

        self.assertEqual(bsnake.min_threshold_canny, 50)
        self.assertEqual(bsnake.max_threshold_canny, 100)
        self.assertEqual(bsnake.process_slices, 15)
        self.assertEqual(bsnake.hough_threshold, 15)
        self.assertEqual(bsnake.min_line_length, 10)
        self.assertEqual(bsnake.max_line_gap, 20)
        self.assertEqual(bsnake.median_blur_ksize, 9)

    def test_line_intersection(self):
        line_intersect_1 = (0, 0, 1, 1)
        line_intersect_2 = (0, 1, 1, 0)

        line_donot_intersect_1 = (0, 0, 1, 1)
        line_donot_intersect_2 = (0, 2, 1, 3)

        line_intersect_negative_1 = (-1, -1, 1, 1)
        line_intersect_negative_2 = (-1, 1, 1, -1)

        self.assertEqual(line_intersection(line_intersect_1, line_intersect_2), (0.5, 0.5))
        self.assertEqual(line_intersection(line_intersect_negative_1, line_intersect_negative_2), (0, 0))
        with self.assertRaises(Exception):
            line_intersection(line_donot_intersect_1, line_donot_intersect_2)



if __name__ == '__main__':
    unittest.main()