import unittest

from contones.io import HFAEncoder
from contones.workers import run_encoderpool, ImageIOPool
from .test_raster import RasterTestBase


#class WorkersTestCase(unittest.TestCase):
class WorkersTestCase(RasterTestBase):
    def test_encoderpool(self):
        #encoded = run_encoderpool(HFAEncoder, [self.f.name, self.f.name])
        pool = ImageIOPool(HFAEncoder, [self.f.name, self.f.name])
        #workers.run()
        #encoded = list(iter(workers.outq.get, 'STOP'))
        encoded = pool.get_results()
        self.assertIsInstance(encoded, list)
        self.assertEqual(len(encoded), 2)
        self.assertIsInstance(encoded[0], str)
