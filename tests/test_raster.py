from __future__ import division

import os
import glob
import hashlib
from io import BytesIO
import tempfile
import unittest

import numpy as np
from osgeo import gdal, ogr, osr

from greenwich.raster import (ImageDriver, Raster, AffineTransform,
    count_unique, driver_for_path, geom_to_array, frombytes, open as ropen)
from greenwich.io import MemFileIO, VSIFile
from greenwich.geometry import Envelope
from greenwich.srs import SpatialReference

def create_gdal_datasource(fname=None, dtype=np.ubyte, bands=1):
    """Returns a GDAL Datasource for testing."""
    xsize, ysize = 800, 1000
    arr = np.ones((ysize, xsize), dtype=dtype)
    if fname:
        driver = gdal.GetDriverByName('GTiff')
    else:
        driver = gdal.GetDriverByName('MEM')
        fname = ''
    datasource = driver.Create(fname, xsize, ysize, bands)
    for bandnum in range(1, bands + 1):
        band = datasource.GetRasterBand(bandnum)
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

def make_point(coord):
    point = ogr.Geometry(ogr.wkbPoint)
    point.AddPoint(*coord)
    return point


class AffineTransformTestCase(unittest.TestCase):
    def setUp(self):
        self.affine = AffineTransform(-120, 2, 0.0, 38, 0.0, -2)

    def test_transform(self):
        x, y = self.affine.origin
        inputs = [(x, y), (x - .1, y)]
        expected = [(0, 0), (-1, 0)]
        self.assertEqual(self.affine.transform(inputs), expected)

    def test_project(self):
        coord = tuple(self.affine.project(((0, 0),)))
        expected = ((-119.0, 37.0),)
        self.assertEqual(coord, expected)


class RasterTestBase(unittest.TestCase):
    def setUp(self):
        self.ds = Raster(create_gdal_datasource(self.fp.name))

    @classmethod
    def setUpClass(cls):
        cls.fp = tempfile.NamedTemporaryFile(suffix='.tif')

    def tearDown(self):
        self.ds.close()

    @classmethod
    def tearDownClass(cls):
        cls.fp.close()


class RasterTestCase(RasterTestBase):
    """Test Raster class."""
    img_header = b'EHFA_HEADER_TAG'

    def setUp(self):
        super(RasterTestCase, self).setUp()
        # Shrink envelope
        envelope = self.ds.envelope.scale(0.2)
        self.bbox = envelope.polygon
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
        self.point = self.bbox.Centroid()
        self.point.AssignSpatialReference(self.ds.sref)

    def test_nodata(self):
        self.assertEqual(self.ds.nodata, 0)

    def test_envelope(self):
        # Need to verify upper right and lower left pairs
        self.assertIsInstance(self.ds.envelope, Envelope)

    def hexdigest(self, s):
        return hashlib.md5(s).hexdigest()

    def test_affine(self):
        atrans = AffineTransform(41.91, 0.5, 0, 12.54, 0, -0.5)
        self.ds.affine = atrans
        self.assertEqual(self.ds.affine, atrans)
        self.assertEqual(self.ds.GetGeoTransform(), atrans.tuple)

    def test_array(self):
        # Reading outside of raster extent is an error.
        self.assertRaises(ValueError, self.ds.array, (-100, 36, -96, 39))
        env = Envelope.from_geom(self.bbox).tuple
        arr = self.ds.array(env)
        self.assertLess(arr.shape, self.ds.shape)

    def test_crop(self):
        """Test image cropping with OGR Geometries."""
        cropped = self.ds.crop(self.bbox)
        self.assertIsInstance(cropped, Raster)
        self.assertEqual(len(cropped), len(self.ds))
        self.assertLess(cropped.size, self.ds.size)
        # Should return the same pixel buffer regardless of the geometry
        # coordinate system.
        px_a = self.hexdigest(self.ds.crop(self.bbox_4326).ReadRaster())
        px_b = self.hexdigest(cropped.ReadRaster())
        self.assertEqual(px_a, px_b)
        px_c = self.hexdigest(self.ds.ReadRaster(*(0, 0) + cropped.size))
        # Reading with cropped dims should be equivalent.
        self.assertEqual(px_a, px_c)
        r = self.ds.crop(tuple(Envelope.from_geom(self.bbox)))
        csum = r[0].Checksum()
        r.close()
        self.assertEqual(csum, cropped[0].Checksum())

    def test_clip(self):
        """Test clipping a raster with a geometry."""
        bands = 3
        multiband = Raster(create_gdal_datasource(bands=bands))
        rast = multiband.clip(self.geom)
        self.assertEqual(len(rast), bands)
        arr = rast.array()[0]
        # First element should be masked.
        self.assertEqual(arr[0,0], multiband.nodata)
        # Center element should be unmasked.
        center = (arr.shape[0] // 2, arr.shape[1] // 2)
        self.assertEqual(arr[center], 1)
        with self.ds.clip(self.bbox) as r:
            m = r.masked_array()
        self.assertLess(m.shape, self.ds.shape)

    def test_clip_point(self):
        r = self.ds.clip(self.point)
        self.assertEqual(r.array(), np.array([[1]]))
        r.close()

    def test_clip_subpixel(self):
        """Test clipping a raster with a polygon smaller than a pixel."""
        # Create point at center of pixel (3, 3).
        coord = map(lambda a, b: a + b * 4.0 - b / 2.0,
                    self.ds.affine.origin, self.ds.affine.scale)
        point = make_point(coord)
        # Scale value by 49% since 50% would include pixel neighbors.
        poly = point.Buffer(self.ds.affine.scale[0] * .49)
        poly.AssignSpatialReference(self.ds.sref)
        r = self.ds.clip(poly)
        self.assertEqual(r.array(), np.array([[1]]))
        r.close()

    def test_clip_overlapping(self):
        """Test clipping a raster with an edge ovelapping polygon."""
        outer_ul = list(self.ds.affine.origin)
        for idx, val in enumerate(self.ds.affine.scale):
            outer_ul[idx] -= val
        point = make_point(outer_ul)
        poly = point.Buffer(self.ds.affine.scale[0] * 2)
        poly.AssignSpatialReference(self.ds.sref)
        r = self.ds.clip(poly)
        # Should return 2x2 pixel window with upper-left corner unmasked.
        self.assertEqual(r.array().tolist(), [[1, 0], [0, 0]])
        r.close()

    def test_close(self):
        with self.assertRaises(AttributeError):
            self.ds.abc123
        self.ds.close()
        self.assertTrue(self.ds.closed)
        with self.assertRaises(ValueError):
            self.ds.GetRasterBand(1)

    def test_frombytes(self):
        shape = (2, 3, 5)
        pixdat = bytes(np.random.random(shape).astype(np.float32).data)
        r = frombytes(pixdat, (3, 2, 5), gdal.GDT_Float32)
        self.assertEqual(r.shape, shape)
        self.assertEqual(r.ReadRaster(), pixdat)
        # Create from bytes.
        pixdat = bytes(bytearray(range(10)))
        r2 = self.ds.new((2, 5))
        r2.frombytes(pixdat)
        self.assertEqual(r2.ReadRaster(), pixdat)
        self.assertEqual(r2.size, (2, 5))
        r2.close()
        # Create with floating point values.
        rfloat = ImageDriver('MEM').raster('memds', (10, 10, 3),
                                           gdal.GDT_Float64)
        b = bytes(np.random.random((10, 10, 3)).data)
        rfloat.WriteRaster(0, 0, rfloat.RasterXSize, rfloat.RasterYSize, b)
        self.assertEqual(rfloat.ReadRaster(), b)
        b2 = bytes(np.random.random((5, 5, 3)).data)
        rf2 = rfloat.new((5, 5, 3))
        rf2.frombytes(b2)
        self.assertEqual(*map(self.hexdigest, (rf2.ReadRaster(), b2)))
        rfloat.close()
        rf2.close()

    def test_masked_array(self):
        m = self.ds.masked_array(self.point)
        self.assertEqual(m, np.ma.masked_array([[1]]))
        extent = tuple(self.ds.envelope.scale(.25))
        m = self.ds.masked_array(extent)
        self.assertLess(m.shape, self.ds.shape)
        self.assertEqual(m[-1,-1], 1)

    def test_save(self):
        ext = '.img'
        fp = tempfile.NamedTemporaryFile(suffix=ext)
        self.ds.save(fp)
        b = fp.read()
        self.assertGreater(fp.file.tell(), 0)
        # Read the image header.
        self.assertEqual(b[:15], self.img_header)
        fp.file.seek(0)
        # Test with filename as str.
        self.ds.save(fp.name)
        self.assertEqual(fp.read()[:15], self.img_header)
        fp.close()
        # Clean up associated files like .aux.xml, etc.
        paths = glob.glob(fp.name.replace(ext, '*'))
        try:
            removed = map(os.unlink, paths)
        except OSError:
            pass

    def test_save_memio(self):
        imgio = MemFileIO(suffix='.img')
        # Test save with MemFileIO object
        self.ds.save(imgio)
        # Test init from a vsimem path.
        r = Raster(imgio.name)
        self.assertEqual(r.driver.ext, 'img')
        self.assertEqual(r.shape, self.ds.shape)
        self.assertEqual(r.envelope, self.ds.envelope)
        self.assertNotEqual(r, self.ds)
        # Bad file extensions should fail.
        self.assertRaises(ValueError, r.save, 'fail.xxx')
        with MemFileIO() as memio:
            r.save(memio, ImageDriver('HFA'))
            imgdata = memio.read(15)
        self.assertEqual(imgdata, self.img_header)
        r.close()

    def test_slice(self):
        rast = ImageDriver('MEM').raster('', (3, 3, 5))
        band = rast.GetRasterBand(5)
        band.Fill(1)
        self.assertEqual(rast[-1].Checksum(), band.Checksum())
        rast.close()

    def test_geom_to_array(self):
        geom = self.geom.Clone()
        geom.TransformTo(self.ds.sref)
        point = geom.Centroid()
        point.AssignSpatialReference(self.ds.sref)
        poly = point.Buffer(self.ds.affine.scale[0])
        # Create "island" polygon to test interior and exterior rings.
        gdiff = geom.Difference(poly)
        mpoly = ogr.Geometry(ogr.wkbMultiPolygon)
        mpoly.AssignSpatialReference(self.ds.sref)
        mpoly.AddGeometry(gdiff)
        for g in geom, gdiff, mpoly:
            arr = geom_to_array(g, self.ds.size, self.ds.affine)
            self.assertEqual(arr.shape, self.ds.shape)
            self.assertEqual((arr.min(), arr.max()), (0, 1))
        x, y = self.ds.affine.transform(point.GetPoints())[0]
        # Pixel within island should use the background color.
        self.assertEqual(arr[y,x], 1)
        p2 = make_point(self.ds.affine.origin)
        poly2 = ogr.Geometry(ogr.wkbMultiPolygon25D)
        poly2.AssignSpatialReference(self.ds.sref)
        for p in point, p2:
            pb = p.Buffer(self.ds.affine.scale[0])
            poly2.AddGeometry(pb)
        poly2.SetCoordinateDimension(3)
        self.assertEqual(poly2.GetCoordinateDimension(), 3)
        arr2 = geom_to_array(poly2, self.ds.size, self.ds.affine)
        idx = np.where(arr2 == 0)
        with self.ds.clip(poly2) as clip:
            m = clip.masked_array()
        for a, b in zip(np.where(~m.mask), idx):
            self.assertEqual(a.tolist(), b.tolist())

    def test_geom3d_to_array(self):
        poly = self.point.Buffer(self.ds.affine.scale[0])
        poly.AssignSpatialReference(self.ds.sref)
        poly.SetCoordinateDimension(3)
        self.assertEqual(poly.GetCoordinateDimension(), 3)
        self.assertEqual(poly.GetGeometryType(), ogr.wkbPolygon25D)
        arr = geom_to_array(poly, self.ds.size, self.ds.affine)
        self.assertEqual((arr.min(), arr.max()), (0, 1))

    def test_count_unique(self):
        a = np.array([(0, 1), (0, 2)])
        self.assertEqual(list(count_unique(a)), [(0, 2), (1, 1), (2, 1)])

    def test_warp(self):
        epsg_id = 4326
        r = self.ds.warp(epsg_id)
        self.assertEqual(r.sref.srid, epsg_id)
        self.assertNotEqual(r.shape, self.ds.shape)
        self.assertEqual(r.array().shape, r.shape)
        epsg_id = 3857
        fp = tempfile.NamedTemporaryFile()
        r = self.ds.warp(epsg_id, dest=fp.name)
        self.assertEqual(r.sref.srid, epsg_id)
        fp.close()

    @unittest.skipIf('TRAVIS' in os.environ,
                     'known issue with warp in Travis build enviroment')
    def test_warp_failure(self):
        # Not possible to transform between these coordinate systems.
        self.assertRaises(ValueError, self.ds.warp, 29635)

    def test_resample(self):
        # Half the original resolution
        factor = 2
        size = tuple([i // factor for i in self.ds.size])
        output = self.ds.resample(size)
        self.assertEqual(output.size, size)
        self.assertEqual(output.affine.scale,
                         tuple(x * factor for x in self.ds.affine.scale))

    def test_new(self):
        dcopy = self.ds.new()
        self.assertEqual(dcopy.nodata, self.ds.nodata)
        self.assertEqual(dcopy.shape, self.ds.shape)
        self.assertNotEqual(dcopy, self.ds)
        # Reduced size withouth pixel data.
        size = tuple([x // 10 for x in self.ds.size])
        dsmall = self.ds.new(size=size)
        self.assertEqual(dsmall.size, size)

    def test_init(self):
        r = Raster(self.fp)
        self.assertIsInstance(r, Raster)
        r.close()
        self.assertFalse(self.ds.closed)
        self.assertRaises(IOError, Raster, 'zzz')
        self.assertRaises(IOError, Raster, None)
        self.assertRaises(IndexError, self.ds.__getitem__, 3)
        self.assertTrue(self.ds == self.ds)
        self.ds.close()

    def test_open(self):
        self.assertIsInstance(ropen(self.fp), Raster)
        self.assertRaises(TypeError, ropen, 123)
        self.assertRaises(TypeError, ropen, ('zyx',))
        self.assertRaises(IOError, ropen, '')

    def test_open_bytesio(self):
        bio = BytesIO(self.fp.read())
        with ropen(bio) as r:
            self.assertIsInstance(r, Raster)
            with VSIFile(r.name) as vsif:
                imgdata = vsif.read()
        self.assertEqual(imgdata, bio.getvalue())
        self.assertRaises(IOError, ropen, BytesIO(b'0123'))


class ImageDriverTestCase(RasterTestBase):

    def setUp(self):
        super(ImageDriverTestCase, self).setUp()
        self.imgdriver = ImageDriver('HFA')
        self.memdriver = ImageDriver('MEM')
        # Test driver specific creation settings.
        opts = {'tiled': 'yes', 'compress': 'deflate'}
        self.tiff = ImageDriver('GTiff', **opts)

    def test_copy(self):
        imgio = MemFileIO()
        ds_copy = ImageDriver('PNG').copy(self.ds, imgio.name)
        self.assertIsInstance(ds_copy, Raster)
        self.assertEqual(ds_copy.driver.ext, 'png')
        # Make sure we get the same number of raster bands back.
        self.assertEqual(*map(len, (self.ds, ds_copy)))
        ds_copy.close()
        # This driver should not support creation or copying.
        self.assertRaises(IOError, ImageDriver('SDTS').copy, self.ds, imgio)
        imgio.close()

    def test_copy_path(self):
        self.ds.close()
        imgio = MemFileIO()
        ds_copy = ImageDriver('JPEG').copy(self.ds.name, imgio.name)
        self.assertEqual(ds_copy.driver.ext, 'jpg')
        ds_copy.close()

    def test_filter_writable(self):
        self.assertIsInstance(ImageDriver.filter_writable()['GTiff'], dict)

    def test_raster(self):
        size = (8, 10, 3)
        with self.memdriver.raster('memds', size, gdal.GDT_Float64) as r:
            rsize = r.size
            bandcount = len(r)
        self.assertEqual(rsize, size[:2])
        self.assertEqual(bandcount, size[-1])
        # This driver should not support creation.
        self.assertRaises(IOError, ImageDriver('PNG').raster,
                          MemFileIO(), (128, 112))

    def test_raster_compression(self):
        imgio = MemFileIO()
        rast = self.tiff.raster(imgio, (10, 10))
        # We cannot verify metadata from a gdal.Dataset in update mode, it must
        # be reopened as read-only first.
        rast.close()
        with Raster(imgio.name) as rast:
            imgmeta = rast.GetMetadata_List('IMAGE_STRUCTURE')
        # The compression name is changed slightly within the GDAL Dataset.
        expected_opt = 'COMPRESSION=DEFLATE'
        self.assertIn(expected_opt, imgmeta)
        self.tiff.settings.update(compress='packbits')
        rast = self.tiff.raster(imgio, (10, 10))
        rast.close()
        with Raster(imgio.name) as rast:
            imgmeta = rast.GetMetadata_List('IMAGE_STRUCTURE')
        self.assertIn('COMPRESSION=PACKBITS', imgmeta)

    def test_raster_fromfile(self):
        fp = tempfile.NamedTemporaryFile(suffix='.img')
        size = (7, 11)
        rast = self.imgdriver.raster(fp, size)
        self.assertEqual(rast.size, size)
        self.assertEqual(rast.driver.ext, 'img')
        rast.close()
        # Cannot create from a non-empty file.
        self.assertRaises(IOError, self.imgdriver.raster, fp.name, size)
        fp.close()

    def test_raster_netcdf(self):
        # Test compressed netCDF creation
        opts = {'format': 'nc4c', 'compress': 'deflate', 'zlevel': 6}
        driver = ImageDriver('netCDF', **opts)
        # No support for netCDF and VSI, use a tempfile.
        fp = tempfile.NamedTemporaryFile(suffix='.nc')
        r = driver.raster(fp, (10, 8, 3))
        r.close()
        r = Raster(fp)
        self.assertTrue(r.size, (10, 8))
        # GDAL is not reading the compression info back, however "ncdump -hs"
        # indicates it is indeed present.
        #self.assertEqual(opts, r.GetMetadata_Dict('IMAGE_STRUCTURE'))
        r.close()

    def test_raster_size_args(self):
        self.assertRaises(TypeError, self.memdriver.raster, '', 10)
        self.assertRaises(ValueError, self.memdriver.raster, '', (-10, -10))
        self.assertRaises(ValueError, self.memdriver.raster, '', (2, 2, 3, 1))

    def test_options(self):
        self.assertGreater(len(self.tiff.options), 0)
        # Inspect available compression options for geotiff.
        self.assertIn('LZW', self.tiff.options['COMPRESS']['choices'])
        # No creation opts should be available for virtual rasters.
        self.assertEqual(ImageDriver('VRT').options, {})

    def test_driver_for_path(self):
        self.assertEqual(driver_for_path('test.jpg').ShortName, 'JPEG')
        self.assertEqual(driver_for_path('test.zzyyxx'), None)

    def test_init(self):
        self.assertRaises(TypeError, ImageDriver, 'zzz')
        self.assertEqual(self.tiff.format, 'GTiff')
        self.assertEqual(self.tiff.mimetype, 'image/tiff')
        self.assertEqual(self.imgdriver.format, 'HFA')
        self.assertEqual(self.imgdriver.ext, 'img')
        hdriver = gdal.GetDriverByName('HFA')
        self.assertEqual(ImageDriver(hdriver)._driver, hdriver)
        self.assertIsInstance(ImageDriver(u'PNG'), ImageDriver)
