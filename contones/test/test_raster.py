import os
import glob
import hashlib
import tempfile
import unittest

import numpy as np
from osgeo import gdal, ogr, osr

from contones.raster import Raster, geom_to_array
from contones.io import ImageIO, driver_for_path
from contones.geometry import Envelope
from contones.srs import SpatialReference

def create_gdal_datasource(fname):
    """Returns a GDAL Datasource for testing."""
    #xsize, ysize = 500, 500
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


class RasterTestCase(RasterTestBase):
    """Test Raster class."""

    def setUp(self):
        super(RasterTestCase, self).setUp()
        # FIXME: Which one?!
        #envelope = (-13832951, 5943581, -13638043, 6099967)
        # OGR format xmin, xmax, ymin, ymax
        #envelope = (-13832951, -13638043, 5943581, 6099967)
        # Shrink envelope
        envelope = Envelope(*self.ds.extent)
        envelope.scale(0.8)
        print 'SCALED', envelope
        self.bbox = envelope.to_geom()
        sref = SpatialReference(3857)
        self.bbox.AssignSpatialReference(sref)
        # Bounding box in WGS84 lat/lng
        self.bbox_4326 = self.bbox.Clone()
        #sref_4326 = osr.SpatialReference(osr.SRS_WKT_WGS84)
        sref_4326 = SpatialReference(osr.SRS_WKT_WGS84)
        self.bbox_4326.TransformTo(sref_4326)
        self.geom = ogr.CreateGeometryFromJson(
            '{"type":"Polygon","coordinates":'
            '[[[-123,47],[-123,48],[-122,49],[-121,48],[-121,47],[-123,47]]]}')
        self.geom.AssignSpatialReference(sref_4326)

    def test_nodata(self):
        self.assertEqual(self.ds.nodata, 0)

    def test_extent(self):
        self.assertEqual(len(self.ds.extent), 4)

    def hexdigest(self, s):
        return hashlib.md5(s).hexdigest()

    def test_raster_crop(self):
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

    def test_raster_mask(self):
        """Test masking a raster with a geometry."""
        rast = self.ds.mask(self.geom)
        arr = rast.ReadAsArray()
        # First element should be masked.
        self.assertEqual(arr[0,0], self.ds.nodata)
        # Center element should be unmasked.
        center = arr.shape[0] / 2, arr.shape[1] / 2
        self.assertEqual(arr[center], 1)
        dims = (self.ds.RasterYSize, self.ds.RasterXSize)
        m_shape = self.ds.mask_asarray(self.bbox).shape
        self.assertLess(m_shape, dims)

    def test_raster_save(self):
        ext = '.img'
        f = tempfile.NamedTemporaryFile(suffix=ext)
        print f.name, f.file.tell()
        self.ds.save(f)
        b = f.read()
        self.assertGreater(f.file.tell(), 0)
        # Read the image header.
        img_header = b[:15]
        print img_header
        self.assertEqual(img_header, 'EHFA_HEADER_TAG')
        print f.name, f.file.tell()
        f.close()
        # Clean up associated files like .aux.xml, etc.
        paths = glob.glob(f.name.replace(ext, '*'))
        try:
            removed = map(os.unlink, paths)
        except OSError:
            pass

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

    def test_resample(self):
        # Half the original resolution
        dims = tuple([i / 2 for i in self.ds.shape])
        output = self.ds.resample(dims)
        self.assertEqual(output.shape, dims)

    def test_new(self):
        dcopy = self.ds.new()
        self.assertEqual(dcopy.nodata, self.ds.nodata)
        self.assertEqual(dcopy.shape, self.ds.shape)


class ImageIOTestCase(RasterTestBase):
    def test_read(self):
        """Test ImageIO reading."""
        for dname in 'GTiff', 'HFA':
            io_obj = ImageIO(driver=dname)
            with Raster(self.f.name) as r:
                r.save(io_obj)
            data = io_obj.getvalue()
            self.assertIsNotNone(data)
            self.assertGreater(data, 0)

    def test_copy_from(self):
        """Test copying a raster."""
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


class SpatialReferenceTestCase(unittest.TestCase):

    def test_wkt(self):
        sref = SpatialReference(osr.SRS_WKT_WGS84)
        self.assertEqual(sref.wkt, osr.SRS_WKT_WGS84)

    def test_epsg(self):
        epsg_id = 3310
        from_epsg = SpatialReference(epsg_id)
        self.assertEqual(from_epsg.srid, epsg_id)

    def test_proj4(self):
        p4 = SpatialReference(2805).ExportToProj4()
        from_proj4 = SpatialReference(p4)
        self.assertEqual(from_proj4.proj4, p4)

