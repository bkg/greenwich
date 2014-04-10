import os
import glob
import hashlib
import tempfile
import unittest

import numpy as np
from osgeo import gdal, ogr, osr

from contones.raster import Raster, geom_to_array, open as openras
from contones.gio import ImageIO
from contones.geometry import Envelope
from contones.srs import SpatialReference

def create_gdal_datasource(fname):
    """Returns a GDAL Datasource for testing."""
    xsize, ysize = 1000, 1000
    #np.random.randint(2, size=(xsize, ysize))
    arr = np.ones((xsize, ysize), dtype=np.byte)
    driver = gdal.GetDriverByName('GTiff')
    datasource = driver.Create(fname, xsize, ysize)
    band = datasource.GetRasterBand(1)
    band.WriteArray(arr)
    band.SetNoDataValue(0)
    sref = osr.SpatialReference()
    sref.ImportFromEPSG(3857)
    datasource.SetProjection(sref.ExportToWkt())
    # Seattle and surrounding area.
    geotransform = (-13692297, 335, 0.0, 6306854, 0.0, -335)
    datasource.SetGeoTransform(geotransform)
    datasource.FlushCache()
    return datasource


class RasterTestBase(unittest.TestCase):
    def setUp(self):
        self.f = tempfile.NamedTemporaryFile(suffix='.tif')
        self.ds = Raster(create_gdal_datasource(self.f.name))

    def tearDown(self):
        self.f.close()


class RasterTestCase(RasterTestBase):
    """Test Raster class."""

    def setUp(self):
        super(RasterTestCase, self).setUp()
        # Shrink envelope
        envelope = self.ds.envelope.scale(0.8)
        self.bbox = envelope.to_geom()
        sref = SpatialReference(3857)
        self.bbox.AssignSpatialReference(sref)
        # Bounding box in WGS84 lat/lng
        self.bbox_4326 = self.bbox.Clone()
        sref_4326 = SpatialReference(osr.SRS_WKT_WGS84)
        self.bbox_4326.TransformTo(sref_4326)
        self.geom = ogr.CreateGeometryFromJson(
            '{"type":"Polygon","coordinates":'
            '[[[-123,47],[-123,48],[-122,49],[-121,48],[-121,47],[-123,47]]]}')
        self.geom.AssignSpatialReference(sref_4326)

    def test_nodata(self):
        self.assertEqual(self.ds.nodata, 0)

    def test_envelope(self):
        # Need to verify upper right and lower left pairs
        self.assertIsInstance(self.ds.envelope, Envelope)

    def hexdigest(self, s):
        return hashlib.md5(s).hexdigest()

    def test_array(self):
        # Reading outside of raster extent is an error.
        self.assertRaises(ValueError, self.ds.array, (-100, 36, -96, 39))
        arr = self.ds.array(self.bbox.GetEnvelope())
        self.assertLess(arr.shape, self.ds.shape)

    def test_crop(self):
        """Test image cropping with OGR Geometries."""
        cropped = self.ds.crop(self.bbox)
        self.assertIsInstance(cropped, Raster)
        self.assertEqual(len(cropped), len(self.ds))
        self.assertLess(cropped.RasterXSize, self.ds.RasterXSize)
        self.assertLess(cropped.RasterYSize, self.ds.RasterYSize)
        # Should return the same pixel buffer regardless of the geometry
        # coordinate system.
        px_a = self.hexdigest(self.ds.crop(self.bbox_4326).ReadRaster())
        px_b = self.hexdigest(cropped.ReadRaster())
        self.assertEqual(px_a, px_b)

    def test_mask(self):
        """Test masking a raster with a geometry."""
        rast = self.ds.mask(self.geom)
        arr = rast.array()
        rast.close()
        # First element should be masked.
        self.assertEqual(arr[0,0], self.ds.nodata)
        # Center element should be unmasked.
        center = arr.shape[0] / 2, arr.shape[1] / 2
        self.assertEqual(arr[center], 1)
        with self.ds.mask(self.bbox) as r:
            m = r.masked_array()
        self.assertLess(m.shape, self.ds.shape)

    def test_save(self):
        ext = '.img'
        f = tempfile.NamedTemporaryFile(suffix=ext)
        self.ds.save(f)
        b = f.read()
        self.assertGreater(f.file.tell(), 0)
        # Read the image header.
        img_header = b[:15]
        self.assertEqual(img_header, 'EHFA_HEADER_TAG')
        f.close()
        # Clean up associated files like .aux.xml, etc.
        paths = glob.glob(f.name.replace(ext, '*'))
        try:
            removed = map(os.unlink, paths)
        except OSError:
            pass
        # Test save with ImageIO
        imgio = ImageIO(driver='HFA')
        self.ds.save(imgio)
        # Test init from a vsimem path.
        r = Raster(imgio.path)
        self.assertEqual(r.shape, self.ds.shape)
        self.assertEqual(r.envelope, self.ds.envelope)
        self.assertNotEqual(r, self.ds)
        r.close()

    def test_geom_to_array(self):
        g = self.geom.Clone()
        g.TransformTo(self.ds.sref)
        arr = geom_to_array(g, self.ds.shape, self.ds.affine)
        self.assertEqual(arr.shape, self.ds.shape)
        self.assertEqual(arr.min(), 0)
        self.assertEqual(arr.max(), 1)

    def test_warp(self):
        epsg_id = 4326
        d = self.ds.warp(epsg_id)
        self.assertEqual(d.sref.srid, epsg_id)
        self.assertNotEqual(d.shape, self.ds.shape)
        self.assertEqual(d.array().shape, d.shape)

    def test_resample(self):
        # Half the original resolution
        dims = tuple([i / 2 for i in self.ds.shape])
        output = self.ds.resample(dims)
        self.assertEqual(output.shape, dims)

    def test_new(self):
        dcopy = self.ds.new()
        self.assertEqual(dcopy.nodata, self.ds.nodata)
        self.assertEqual(dcopy.shape, self.ds.shape)
        self.assertNotEqual(dcopy, self.ds)
        pixdat = ''.join(map(str, range(10)))
        d2 = self.ds.new(pixdat, (2, 5))
        self.assertEqual(d2.ReadRaster(), pixdat)

    def test_init(self):
        #vsipath = '/vsicurl/ftp://ftp.cpc.ncep.noaa.gov/GIS/GRADS_GIS/GeoTIFF/TEMP/us_tmax/us.tmax_nohads_ll_20110705_float.tif'
        vsipath = 'us.tmax_nohads_ll_20110705_float.tif'
        tmax = Raster(vsipath)
        self.assertTrue(tmax)
        self.assertTrue(tmax.masked_array().max())
        with openras(vsipath) as r:
            self.assertIsInstance(r, Raster)
