from collections import namedtuple
import json
import unittest

from osgeo import ogr

from greenwich.geometry import Envelope, Geometry
from greenwich.srs import SpatialReference

# Test geometry objects with 'wkb' attr from GeoDjango, Shapely.
WKBGeom = namedtuple('WKBGeom', ['wkb'])


class EnvelopeTestCase(unittest.TestCase):

    def setUp(self):
        extent = (-120, 30, -110, 40)
        self.en = Envelope(*extent)
        self.esub = Envelope(-118, 32, -115, 38)

    def test_contains(self):
        self.assertIn(self.esub, self.en)
        self.assertFalse(self.en.contains((0, 0, 0, 0)))
        self.assertRaises(TypeError, self.en.contains, ())
        self.assertRaises(TypeError, self.en.contains, 'something')
        # FIXME: this should probably throw a TypeError
        self.assertFalse(self.en.contains('four'))

    def test_eq(self):
        self.assertEqual(self.en, Envelope(*self.en.tuple))

    def test_init(self):
        # Zero area envelopes are valid.
        self.assertIsInstance(Envelope(1, 1, 1, 1), Envelope)

    def test_intersects(self):
        # Move lower-left coord further out.
        overlapping = Envelope(self.en.min_x - 10, self.en.min_y -10,
                               *self.en.ur)
        self.assertTrue(self.en.intersects(overlapping))
        outside = self.en + 15
        self.assertFalse(self.en.intersects(outside))
        self.assertRaises(TypeError, self.en.intersects, ())

    def test_invalid(self):
        with self.assertRaises(ValueError):
            Envelope(80, 2, 1, 2)
            Envelope(2, 1, 1, 2)


class GeometryTestCase(unittest.TestCase):

    def setUp(self):
        self.gdict = {'type': 'Polygon',
                      'coordinates': [[[-123,47],[-123,48],[-122,49],
                                       [-121,48],[-121,47],[-123,47]]]}

    def test_init(self):
        geom = Geometry(self.gdict)
        jsondata = geom.ExportToJson()
        self.assertEqual(json.loads(jsondata), self.gdict)

        jsondata = json.loads(Geometry(geojson=self.gdict).ExportToJson())
        self.assertEqual(jsondata, self.gdict)
        jsondata = Geometry(geojson=json.dumps(self.gdict)).ExportToJson()
        self.assertEqual(json.loads(jsondata), self.gdict)

        wkt = 'POLYGON ((0 0,5 0,5 5,0 5,0 0))'
        self.assertEqual(Geometry(wkt=wkt).ExportToWkt(), wkt)
        self.assertEqual(Geometry(ogr.wkbPolygon).GetGeometryType(),
                         ogr.wkbPolygon)
        g = ogr.CreateGeometryFromWkt('Point (2 1)')
        wkb = g.ExportToWkb()
        wg = WKBGeom(wkb)
        self.assertEqual(Geometry(wg).ExportToWkb(), wkb)

    def test_init_spatialref(self):
        epsg_id = 4326
        sref = SpatialReference(epsg_id)
        geom = Geometry(self.gdict, srs=epsg_id)
        self.assertEqual(geom.GetSpatialReference(), sref)
        geom = Geometry(self.gdict, srs=sref)
        self.assertEqual(geom.GetSpatialReference(), sref)
