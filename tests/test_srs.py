import unittest
from osgeo import osr

from greenwich.srs import SpatialReference, transform_lonlat, transform_tile


class SpatialReferenceTestCase(unittest.TestCase):
    def test_wkt(self):
        sref = SpatialReference(osr.SRS_WKT_WGS84)
        self.assertEqual(sref.wkt, osr.SRS_WKT_WGS84)
        self.assertEqual(str(sref), osr.SRS_WKT_WGS84)
        wkt = unicode(osr.SRS_WKT_WGS84)
        self.assertEqual(str(SpatialReference(wkt)), wkt)

    def test_epsg(self):
        epsg_id = 3310
        from_epsg = SpatialReference(epsg_id)
        self.assertEqual(from_epsg.srid, epsg_id)
        self.assertEqual(SpatialReference().srid, None)

    def test_epsg_strings(self):
        self.assertEqual(SpatialReference('EPSG:4269').srid, 4269)
        self.assertEqual(SpatialReference('4269').srid, 4269)

    def test_named_crs(self):
        self.assertEqual(
            SpatialReference('urn:ogc:def:crs:OGC:1.3:CRS84').srid, 4326)

    def test_proj4(self):
        p4 = SpatialReference(2805).ExportToProj4()
        from_proj4 = SpatialReference(p4)
        self.assertEqual(from_proj4.proj4, p4)
        p4 = ('+proj=aea +lat_1=20 +lat_2=60 +lat_0=40 +lon_0=-96 +x_0=0 '
              '+y_0=0 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs ')
        sref = SpatialReference(p4)
        self.assertEqual(sref.proj4, p4)
        # Spatial ref with no EPSG code defined is None.
        self.assertIs(sref.srid, None)

    def test_equality(self):
        self.assertEqual(SpatialReference(3857), SpatialReference(3857))
        self.assertNotEqual(SpatialReference(4326), SpatialReference(3857))


class TransformTileTestCase(unittest.TestCase):
    def test_transform(self):
        self.assertAlmostEqual(transform_tile(553, 346, 10),
                               (14.4140625, 50.28933925329178))

    def test_transform_lonlat(self):
        self.assertAlmostEqual(transform_lonlat(-115.4, 46.6, 6), (11, 22))

    def test_invalid_tile(self):
        self.assertRaises(ValueError, transform_tile, 1000, 1000, 3)
