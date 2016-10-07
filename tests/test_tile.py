import unittest

from greenwich import tile


class TileTestCase(unittest.TestCase):
    def test_from_bbox(self):
        bbox = [-95.54, -19.35, -46.19, 20.47]
        tiles = list(tile.from_bbox(bbox, (3, 4)))
        counts = {}
        for t in tiles:
            counts.setdefault(t[-1], 0)
            counts[t[-1]] += 1
        # Should have 4 tiles at zoom level 3 and 6 at z4.
        self.assertEqual(counts[3], 4)
        self.assertEqual(counts[4], 6)

    def test_to_lonlat(self):
        self.assertAlmostEqual(tile.to_lonlat(553, 346, 10),
                               (14.4140625, 50.28933925329178))

    def test_to_tile(self):
        self.assertAlmostEqual(tile.to_tile(-115.4, 46.6, 6), (11, 22))

    def test_invalid_tile(self):
        self.assertRaises(ValueError, tile.to_lonlat, 1000, 1000, 3)
