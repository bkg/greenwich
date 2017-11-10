import unittest

from greenwich.layer import MemoryLayer

from .test_raster import make_point


class MemoryLayerTestCase(unittest.TestCase):
    def setUp(self):
        p = make_point((5, 10))
        self.layer = MemoryLayer.from_records([(1, p)])

    def test_len(self):
        self.assertEqual(len(self.layer), 1)

    def test_index(self):
        self.assertTrue(self.layer[1])

    def tearDown(self):
        self.layer.close()
