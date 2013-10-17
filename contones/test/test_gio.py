import tempfile

from .test_raster import RasterTestBase
from contones.raster import Raster
from contones.gio import ImageIO, driver_for_path


class ImageIOTestCase(RasterTestBase):
    def test_getvalue(self):
        """Test ImageIO reading."""
        for dname in 'GTiff', 'HFA':
            io_obj = ImageIO(driver=dname)
            with Raster(self.f.name) as r:
                r.save(io_obj)
            data = io_obj.getvalue()
            self.assertIsNotNone(data)
            self.assertGreater(data, 0)

    def test_copy_from(self):
        ds_copy = ImageIO(driver='PNG').copy_from(self.ds)
        self.assertIsInstance(ds_copy, Raster)
        self.assertEqual(ds_copy.io.ext, 'png')
        # Make sure we get the same number of raster bands back.
        self.assertEqual(*map(len, (self.ds, ds_copy)))

    def test_create(self):
        f = tempfile.NamedTemporaryFile(suffix='.img')
        imgio = ImageIO(f.name)
        with self.assertRaises(ValueError):
            r = imgio.create(-10, -10)
        size = (10, 10)
        rast = imgio.create(*size)
        self.assertEqual(rast.shape, size)
        self.assertEqual(rast.io.ext, 'img')
        f.close()

    def test_driver_for_path(self):
        self.assertEqual(driver_for_path('test.jpg').ShortName, 'JPEG')
        self.assertEqual(driver_for_path('test.zzyyxx'), None)
