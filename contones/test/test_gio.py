import tempfile

from osgeo import gdal

from .test_raster import RasterTestBase
from contones.raster import Raster
from contones.gio import driver_for_path, ImageDriver, ImageIO


class ImageDriverTestCase(RasterTestBase):

    def test_copy_from(self):
        ds_copy = ImageDriver('PNG').copy(self.ds)
        self.assertIsInstance(ds_copy, Raster)
        self.assertEqual(ds_copy.io.ext, 'png')
        # Make sure we get the same number of raster bands back.
        self.assertEqual(*map(len, (self.ds, ds_copy)))

    def test_driver_for_path(self):
        self.assertEqual(driver_for_path('test.jpg').ShortName, 'JPEG')
        self.assertEqual(driver_for_path('test.zzyyxx'), None)

    def test_init(self):
        self.assertEqual(ImageDriver().ShortName, 'GTiff')
        img = ImageDriver('HFA')
        self.assertEqual(img.ShortName, 'HFA')
        self.assertEqual(img.ext, 'img')
        hdriver = gdal.GetDriverByName('HFA')
        self.assertEqual(ImageDriver(hdriver)._driver, hdriver)


class ImageIOTestCase(RasterTestBase):

    def test_create(self):
        f = tempfile.NamedTemporaryFile(suffix='.img')
        imgio = ImageIO(f)
        self.assertRaises(ValueError, imgio.create, ((-10, -10),))
        size = (10, 10)
        rast = imgio.create(size)
        self.assertEqual(rast.shape, size)
        self.assertEqual(rast.io.ext, 'img')
        f.close()

    def test_create_options(self):
        opts = ['TILED=YES', 'COMPRESS=DEFLATE']
        imgio = ImageIO(driver='GTiff')
        rast = imgio.create((10, 10), options=opts)
        # We cannot verify metadata from an open GDALDataset, it must be
        # reopened first.
        rast.close()
        rast = Raster(imgio.name)
        # The compression name is changed slightly within the GDAL Dataset.
        expected_opt = 'COMPRESSION=DEFLATE'
        self.assertIn(expected_opt, rast.ds.GetMetadata_List('IMAGE_STRUCTURE'))

    def test_getvalue(self):
        """Test ImageIO reading."""
        io_obj = ImageIO(self.f)
        data = io_obj.getvalue()
        self.assertIsInstance(data, str)
        self.assertGreater(data, 0)
        io_obj = ImageIO()
        self.assertRaises(IOError, io_obj.getvalue)

    def test_init(self):
        self.assertTrue(ImageIO())
        jpeg = 'test.jpg'
        rf = ImageIO(jpeg)
        self.assertEqual(rf.name, jpeg)
        rf2 = ImageIO(driver='JPEG')
        self.assertEqual(rf.driver.info, rf2.driver.info)
