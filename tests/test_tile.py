import unittest

from greenwich import tile


class TileTestCase(unittest.TestCase):
    def test_to_lonlat(self):
        self.assertAlmostEqual(tile.to_lonlat(553, 346, 10),
                               (14.4140625, 50.28933925329178))

    def test_to_tile(self):
        self.assertAlmostEqual(tile.to_tile(-115.4, 46.6, 6), (11, 22))

    def test_invalid_tile(self):
        self.assertRaises(ValueError, tile.to_lonlat, 1000, 1000, 3)
