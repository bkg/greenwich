import unittest

from contones.geometry import Envelope


class EnvelopeTestCase(unittest.TestCase):

    def test_init(self):
        extent = (-120, 38, -110, 45)
        e1 = Envelope(*extent)
        extent_inv = (-110, 45, -120, 38)
        e2 = Envelope(*extent_inv)
        self.assertEqual(e1.tuple, e2.tuple)

    def test_invalid(self):
        with self.assertRaises(ValueError):
            env = Envelope(80, 2, 1, 2)
            env = Envelope(2, 1, 1, 2)
