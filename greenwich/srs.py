"""Spatial reference systems"""
import math
from osgeo import osr

def transform_tile(xtile, ytile, zoom):
    """Returns a tuple of (longitude, latitude) from a map tile xyz coordinate.

    See http://wiki.openstreetmap.org/wiki/Slippy_map_tilenames#Lon..2Flat._to_tile_numbers_2

    Arguments:
    xtile - x tile location as int or float
    ytile - y tile location as int or float
    zoom - zoom level as int or float
    """
    n = 2.0 ** zoom
    lon = xtile / n * 360.0 - 180.0
    # Caculate latitude in radians and convert to degrees constrained from -90
    # to 90. Values too big for tile coordinate pairs are invalid and could
    # overflow.
    try:
        lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    except OverflowError:
        raise ValueError('Invalid tile coordinate for zoom level %d' % zoom)
    lat = math.degrees(lat_rad)
    return lon, lat

def transform_lonlat(lon, lat, zoom):
    """Returns a tuple of (xtile, ytile) from a (longitude, latitude) coordinate.

    See http://wiki.openstreetmap.org/wiki/Slippy_map_tilenames

    Arguments:
    lon - longitude as int or float
    lat - latitude as int or float
    zoom - zoom level as int or float
    """
    lat_rad = math.radians(lat)
    n = 2.0 ** zoom
    xtile = int((lon + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.log(math.tan(lat_rad) +
                (1 / math.cos(lat_rad))) / math.pi) / 2.0 * n)
    return xtile, ytile


class SpatialReference(osr.SpatialReference):
    """Base class for extending osr.SpatialReference."""

    def __init__(self, sref=None):
        super(SpatialReference, self).__init__()
        try:
            sref = sref.strip()
            part = sref.split(':')[-1]
        except AttributeError:
            part = sref
        try:
            epsg = int(part)
        except ValueError:
            if sref.startswith('+proj='):
                self.ImportFromProj4(sref)
            elif sref.startswith('urn:ogc:def:crs'):
                self.SetWellKnownGeogCS(part)
            else:
                self.ImportFromWkt(sref)
            # Add EPSG authority if applicable
            self.AutoIdentifyEPSG()
        except TypeError:
            pass
        else:
            self.ImportFromEPSG(epsg)

    def __eq__(self, another):
        return bool(self.IsSame(another))

    def __ne__(self, another):
        return not self.__eq__(another)

    def __repr__(self):
        return '<%s %r>' % (self.__class__.__name__, self.proj4)

    def __str__(self):
        return self.wkt

    @property
    def srid(self):
        """Returns the EPSG ID as int if it exists."""
        epsg_id = (self.GetAuthorityCode('PROJCS') or
                   self.GetAuthorityCode('GEOGCS'))
        try:
            return int(epsg_id)
        except TypeError:
            return

    @property
    def wkt(self):
        """Returns this projection in WKT format."""
        return self.ExportToWkt()

    @property
    def proj4(self):
        """Returns this projection as a proj4 string."""
        return self.ExportToProj4()
