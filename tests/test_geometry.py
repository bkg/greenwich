from collections import namedtuple
import json
import unittest

from osgeo import ogr

from greenwich.geometry import transform, Envelope, Geometry
from greenwich.srs import SpatialReference

# Test geometry objects with 'wkb' attr from GeoDjango, Shapely.
WKBGeom = namedtuple('WKBGeom', ['wkb'])


class EnvelopeTestCase(unittest.TestCase):

    def setUp(self):
        self.extent = (-120, 30, -110, 40)
        self.en = Envelope(*self.extent)
        self.esub = Envelope(-118, 32, -115, 38)

    def test_contains(self):
        self.assertIn(self.esub, self.en)
        self.assertFalse(self.en.contains((0, 0, 0, 0)))
        self.assertFalse(self.en.contains((-118, 45, -115, 50)))
        self.assertRaises(ValueError, self.en.contains, ())

    def test_subtract(self):
        shrink = Envelope(0, 0, 100, 100) - Envelope(-50, -50, 50, 50)
        self.assertEqual(tuple(shrink), (0, 0, 50, 50))

    def test_eq(self):
        self.assertEqual(self.en, Envelope(*self.extent))
        self.assertFalse(self.en == 3)
        self.assertEqual(self.en[-1], self.extent[-1])

    def test_add(self):
        e = self.en + Envelope(-124, 28, -112, 41)
        self.assertEqual(tuple(e), (-124, 28, -110, 41))

    def test_expand(self):
        self.en.expand((0, 0, 10, 10))
        self.assertEqual(tuple(self.en), (-120, 0, 10, 40))
        e = Envelope(-120, 38, -118, 40)
        e.expand((0, 0))
        self.assertEqual(tuple(e), (-120, 0, 0, 40))

    def test_init(self):
        # Test flipped lower-left and upper-right coordinates.
        self.assertEqual(Envelope(-120, 38, -110, 45),
                         Envelope(-110, 45, -120, 38))
        # Zero area envelopes are valid.
        self.assertIsInstance(Envelope(1, 1, 1, 1), Envelope)
        self.assertEqual(len(self.en), 4)
        self.assertRaises(ValueError, Envelope, tuple('abcd'))

    def test_intersect(self):
        self.assertEqual(self.en.intersect(self.esub), self.esub)
        e = Envelope(-112, 32, 0, 36)
        self.assertEqual(self.en.intersect(e), Envelope(-112, 32, -110, 36))
        e2 = Envelope(0, 0, 10, 10)
        self.assertEqual(self.en.intersect(e2), Envelope(0, 0, 0, 0))

    def test_intersects(self):
        # Move lower-left coord further out.
        overlapping = Envelope(self.en.min_x - 10, self.en.min_y - 10,
                               *self.en.ur)
        self.assertTrue(self.en.intersects(overlapping))
        outside = Envelope(0, 0, 5, 5)
        self.assertFalse(self.en.intersects(outside))
        outside_below = Envelope(-118, 20, -116, 25)
        self.assertFalse(self.en.intersects(outside_below))
        self.assertRaises(ValueError, self.en.intersects, ())

    def test_invalid(self):
        self.assertRaises(ValueError, Envelope, 'something')
        self.assertRaises(TypeError, Envelope, None)

    def test_scale(self):
        factor = .25
        en2 = self.en * factor
        self.assertEqual(en2.centroid, self.en.centroid)
        self.assertEqual(en2.width, self.en.width * factor)
        self.assertEqual(en2.height, self.en.height * factor)


class GeometryTestCase(unittest.TestCase):

    def setUp(self):
        self.gdict = {'type': 'Polygon',
                      'coordinates': [[[-123, 47],[-123, 48],[-122, 49],
                                       [-121, 48],[-121, 47],[-123, 47]]]}
        self.geom = Geometry(self.gdict, srs=4326)

    def test_geo_interface(self):
        g = self.gdict['coordinates'][0]
        self.assertEqual(self.geom.__geo_interface__, self.gdict)
        point = {'type': 'Point', 'coordinates': [2.0, 4.0]}
        g = Geometry(point)
        self.assertEqual(g.__geo_interface__, point)

    def test_init(self):
        jsondata = self.geom.ExportToJson()
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

    def test_init_error(self):
        self.assertRaises(ValueError, Geometry, '')

    def test_init_spatialref(self):
        epsg_id = 4326
        sref = SpatialReference(epsg_id)
        geom = Geometry(self.gdict, srs=epsg_id)
        self.assertEqual(geom.GetSpatialReference(), sref)
        geom = Geometry(self.gdict, srs=sref)
        self.assertEqual(geom.GetSpatialReference(), sref)

    def test_transform_mask(self):
        sref = SpatialReference(3857)
        tr = transform(self.geom.ExportToWkb(), sref)
        self.assertEqual(tr.GetSpatialReference(), sref)
