import unittest

from greenwich.geometry import Envelope


class EnvelopeTestCase(unittest.TestCase):

    def setUp(self):
        extent = (-120, 30, -110, 40)
        self.en = Envelope(*extent)
        self.esub = Envelope(-118, 32, -115, 38)

    def test_contains(self):
        self.assertIn(self.esub, self.en)
        self.assertFalse(self.en.contains((0, 0, 0, 0)))
        self.assertRaises(TypeError, self.en.contains, ())
        self.assertRaises(TypeError, self.en.contains, 'something')
        # FIXME: this should probably throw a TypeError
        self.assertFalse(self.en.contains('four'))

    def test_eq(self):
        self.assertEqual(self.en, Envelope(*self.en.tuple))

    def test_init(self):
        # Test flipped lower-left and upper-right coordinates.
        self.assertEqual(Envelope(-120, 38, -110, 45),
                         Envelope(-110, 45, -120, 38))
        # Zero area envelopes are valid.
        self.assertIsInstance(Envelope(1, 1, 1, 1), Envelope)

    def test_intersects(self):
        # Move lower-left coord further out.
        overlapping = Envelope(self.en.min_x - 10, self.en.min_y -10,
                               *self.en.ur)
        self.assertTrue(self.en.intersects(overlapping))
        outside = self.en + 15
        self.assertFalse(self.en.intersects(outside))
        self.assertRaises(TypeError, self.en.intersects, ())

    def test_invalid(self):
        with self.assertRaises(ValueError):
            Envelope(80, 2, 1, 2)
            Envelope(2, 1, 1, 2)
