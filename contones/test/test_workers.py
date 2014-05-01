import unittest

from contones.workers import run_encoderpool, ImageIOPool
from .test_raster import RasterTestBase


class WorkersTestCase(RasterTestBase):
    def test_encoderpool(self):
        files = [self.f.name, self.f.name]
        pool = ImageIOPool('HFA', files)
        encoded = pool.get_results()
        self.assertIsInstance(encoded, list)
        self.assertEqual(len(encoded), len(files))
        self.assertIsInstance(encoded[0], str)
