import unittest

from osgeo import gdal

from .test_raster import RasterTestBase
from greenwich.io import ImageFileIO


class ImageFileIOTestCase(unittest.TestCase):
    def setUp(self):
        self.imgio = ImageFileIO()
        vsif = gdal.VSIFOpenL(self.imgio.name, 'wb')
        self.data = '0123'
        gdal.VSIFWriteL(self.data, len(self.data), 1, vsif)

    def test_close(self):
        self.imgio.close()
        # VSIMem file should return None after closing.
        self.assertIsNone(gdal.VSIStatL(self.imgio.name))

    def test_read(self):
        imgio = ImageFileIO()
        self.assertTrue(imgio.readable())
        # Empty file should return an empty string.
        self.assertEqual(imgio.read(), '')
        self.assertEqual(gdal.VSIStatL(self.imgio.name).size,
                         len(self.data))
        self.assertEqual(self.imgio.read(), self.data)
        self.imgio.seek(0)
        self.assertEqual(self.imgio.read(), self.data)

    def test_seek(self):
        self.assertTrue(self.imgio.seekable)
        self.assertEqual(self.imgio.tell(), 0)
        self.imgio.seek(2)
        self.assertEqual(self.imgio.tell(), 2)
        self.imgio.seek(0, 2)
        self.assertEqual(self.imgio.tell(), len(self.data))

    def test_readinto(self):
        size = len(self.data) / 2
        b = bytearray(size)
        self.imgio.readinto(b)
        self.assertEqual(str(b), self.data[:size])

    def test_truncate(self):
        self.imgio.truncate(2)
        self.imgio.seek(0, 2)
        self.assertEqual(self.imgio.tell(), 2)
