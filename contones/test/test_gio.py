import tempfile
import unittest

from osgeo import gdal

from .test_raster import RasterTestBase
from contones.raster import Raster
from contones.gio import driver_for_path, ImageDriver, ImageFileIO


class ImageDriverTestCase(RasterTestBase):

    def setUp(self):
        super(ImageDriverTestCase, self).setUp()
        self.imgdriver = ImageDriver('HFA')

    def test_copy(self):
        imgio = ImageFileIO()
        ds_copy = ImageDriver('PNG').copy(self.ds, imgio.name)
        self.assertIsInstance(ds_copy, Raster)
        self.assertEqual(ds_copy.driver.ext, 'png')
        # Make sure we get the same number of raster bands back.
        self.assertEqual(*map(len, (self.ds, ds_copy)))
        ds_copy.close()
        imgio.close()

    def test_raster(self):
        f = tempfile.NamedTemporaryFile(suffix='.img')
        self.assertRaises(ValueError, self.imgdriver.raster, f, (-10, -10))
        size = (10, 10)
        rast = self.imgdriver.raster(f, size)
        self.assertEqual(rast.shape, size)
        self.assertEqual(rast.driver.ext, 'img')
        # Cannot create from a non-empty file.
        self.assertRaises(IOError, self.imgdriver.raster, f.name, size)
        f.close()

    # TODO: store driver opts as instance attrs?
    def test_create_options(self):
        opts = {'TILED': 'YES', 'COMPRESS': 'DEFLATE'}
        driver = ImageDriver('GTiff')
        imgio = ImageFileIO()
        rast = driver.raster(imgio.name, (10, 10), options=opts)
        # We cannot verify metadata from an open GDALDataset, it must be
        # reopened first.
        rast.close()
        with Raster(imgio.name) as rast:
            # The compression name is changed slightly within the GDAL Dataset.
            expected_opt = 'COMPRESSION=DEFLATE'
            self.assertIn(expected_opt,
                          rast.ds.GetMetadata_List('IMAGE_STRUCTURE'))
        imgio.close()

    def test_driver_for_path(self):
        self.assertEqual(driver_for_path('test.jpg').ShortName, 'JPEG')
        self.assertEqual(driver_for_path('test.zzyyxx'), None)

    def test_init(self):
        self.assertEqual(ImageDriver().ShortName, 'GTiff')
        self.assertEqual(self.imgdriver.ShortName, 'HFA')
        self.assertEqual(self.imgdriver.ext, 'img')
        hdriver = gdal.GetDriverByName('HFA')
        self.assertEqual(ImageDriver(hdriver)._driver, hdriver)


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
