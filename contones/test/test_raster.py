import os
import binascii
import glob
import hashlib
import tempfile
import unittest

import numpy as np
from osgeo import gdal, ogr, osr

#from contones.gdalwrap import (Raster, GeoTIFFEncoder, HFAEncoder,
#from ..gdalwrap import (Raster, GeoTIFFEncoder, HFAEncoder,
    #geom_to_array, envelope_asgeom, run_encoderpool)
from contones.raster import Raster, envelope_asgeom, Envelope, geom_to_array
from contones.io import GeoTIFFEncoder, HFAEncoder, run_encoderpool

#TESTDIR = os.path.dirname(__file__)
# create a GEOS geometry from a geojson string.
#POLYGON = ogr.CreateGeometryFromJson(
    #'{"type":"Polygon","coordinates":'
    #'[[[-120,38],[-120,40],[-119,41],[-118,40],[-118,38],[-120,38]]]}')
#POLYGON.transform(3857)
#osr.SpatialReference(osr.SRS_WKT_WGS84)
#POLYGON.TransformTo(3857)
#sref = osr.SpatialReference()

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


class RasterTestCase(unittest.TestCase):
    """Test Raster class."""

    def setUp(self):
        self.f = tempfile.NamedTemporaryFile(suffix='.tif')
        self.ds = Raster(create_gdal_datasource(self.f.name))
        # FIXME: Which one?!
        #envelope = (-13832951, 5943581, -13638043, 6099967)
        # OGR format xmin, xmax, ymin, ymax
        #envelope = (-13832951, -13638043, 5943581, 6099967)
        # Get the raster extent and shrink it on all sides by 100 pixels to
        # test cropping the image.
        #gt = self.ds.GetGeoTransform()
        #pixeldist = [i * 100 for i in gt[1], gt[-1]]
        #envelope = (self.ds.extent[0] + pixeldist[0], self.ds.extent[2] + pixeldist[0],
                    #self.ds.extent[1] + pixeldist[1], self.ds.extent[3] + pixeldist[1])
        # Shrink envelope
        envelope = Envelope(*self.ds.extent)
        envelope.scale(0.8)
        print 'SCALED', envelope
        self.bbox = envelope.to_geom()
        #print 'ASGEOM-ENV', envelope.to_geom().GetEnvelope()

        sref = osr.SpatialReference()
        sref.ImportFromEPSG(3857)
        self.bbox.AssignSpatialReference(sref)
        # Bounding box in WGS84 lat/lng
        self.bbox_4326 = self.bbox.Clone()
        sref_4326 = osr.SpatialReference(osr.SRS_WKT_WGS84)
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

    def test_encoders(self):
        """Test raster encoders."""
        for Encoder in [GeoTIFFEncoder, HFAEncoder]:
            encoder_obj = Encoder()
            with Raster(self.f.name) as r:
                r.save(encoder_obj)
                #print binascii.b2a_base64(ds_buffer)
            #r = Raster(self.f.name)
            #r.save(encoder_obj)
            #r.close()
            data = encoder_obj.read()
            self.assertIsNotNone(data)
            self.assertGreater(data, 0)

    def test_encoder_copy(self):
        """Test copying a raster."""
        ds_copy = GeoTIFFEncoder().copy_from(self.ds)
        self.assertIsInstance(ds_copy, Raster)
        # Make sure we get the same number of raster bands back.
        self.assertEqual(*map(len, (self.ds, ds_copy)))

    def test_encoderpool(self):
        encoded = run_encoderpool(HFAEncoder, [self.f.name, self.f.name])
        self.assertIsInstance(encoded, list)
        self.assertEqual(len(encoded), 2)
        self.assertIsInstance(encoded[0], str)

    def test_rasterize_geom(self):
        return True
        g = self.geom.Clone()
        g.TransformTo(self.ds.sref)
        # Test the pixel window based on the geometry extent.
        #pixwin = self.ds._pixelwin_from_extent(g.extent)
        pixwin = self.ds._pixelwin_from_extent(g.GetEnvelope())
        self.assertGreater(pixwin['lr_px'], pixwin['ul_px'])
        # just testing the first dimension
        self.assertEqual(
            pixwin['lr_px'][0] - pixwin['ul_px'][0], pixwin['dims'][0])
        # TODO: Could use gdal.RasterizeLayer(), but need a good way to
        # OGRLayerShadow like gdal.RasterizeLayer(memds, [1],
        # ogr.Geometry(wkb=str(obj.geometry.wkb)), burn_values=[1])
        # The above may be faster than PIL.
        # Now do the rasterization
        arr = geom_to_array(
            self.geom, pixwin['dims'], pixwin['geotrans'])
        self.assertEqual(tuple(reversed(arr.shape)), pixwin['dims'])
        # We should have 1 or True in the binary array.
        self.assertTrue(arr.any())
