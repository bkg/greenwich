import os
import tempfile
import unittest

from osgeo import gdal

from .test_raster import RasterTestBase
from greenwich.io import MemFileIO, VSIFile, vsiprefix


class MemFileIOTestCase(unittest.TestCase):
    def setUp(self):
        self.imgio = MemFileIO()
        vsif = gdal.VSIFOpenL(self.imgio.name, 'wb')
        self.data = '0123'
        gdal.VSIFWriteL(self.data, len(self.data), 1, vsif)

    def test_close(self):
        self.imgio.close()
        # VSIMem file should return None after closing as mem is freed.
        self.assertIsNone(gdal.VSIStatL(self.imgio.name))
        self.assertTrue(self.imgio.closed)
        # Reading a closed file should throw an error.
        self.assertRaises(ValueError, self.imgio.read)

    def test_read(self):
        imgio = MemFileIO()
        self.assertFalse(os.path.exists(imgio.name))
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
        self.assertEqual(bytes(b), self.data[:size])

    def test_truncate(self):
        self.imgio.truncate(2)
        self.imgio.seek(0, 2)
        self.assertEqual(self.imgio.tell(), 2)
        self.imgio.seek(0)
        self.imgio.truncate()
        self.assertEqual(self.imgio.tell(), 0)

    def test_write(self):
        memio = MemFileIO()
        data = 'stuff'
        memio.write(data)
        memio.seek(0)
        self.assertEqual(memio.read(), data)
        data = bytearray(range(10))
        memio.seek(0)
        memio.write(data)
        memio.seek(0)
        self.assertEqual(memio.read(), bytes(data))


class VSIFileTestCase(unittest.TestCase):
    def test_close(self):
        fd, name = tempfile.mkstemp()
        try:
            vsif = VSIFile(name)
            vsif.close()
            self.assertTrue(vsif.closed)
            # VSIFile never unlinks on close, unlike MemFileIO which needs to
            # free up allocated memory.
            self.assertTrue(os.path.isfile(name))
        finally:
            os.unlink(name)

    def test_read_missing(self):
        self.assertRaises(IOError, VSIFile, 'missing.zyx', 'rb')

    def test_vsiprefix(self):
        self.assertEqual(vsiprefix('test.jpg'), 'test.jpg')
        self.assertEqual(vsiprefix('/home/user/test.zip'),
                         '/vsizip//home/user/test.zip')
        self.assertEqual(vsiprefix('http://osgeo.org/data/spif83.ecw'),
                         '/vsicurl/http://osgeo.org/data/spif83.ecw')
        self.assertEqual(vsiprefix('http://osgeo.org/data/a.zip/a.tif'),
                         '/vsizip/vsicurl/http://osgeo.org/data/a.zip/a.tif')
