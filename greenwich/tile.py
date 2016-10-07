import math
import itertools

from greenwich.geometry import Envelope

def from_bbox(bbox, zlevs):
    """Yields tile (x, y, z) tuples for a bounding box and zoom levels.

    Arguments:
    bbox - bounding box as a 4-length sequence
    zlevs - sequence of tile zoom levels
    """
    env = Envelope(bbox)
    for z in zlevs:
        corners = [to_tile(*coord + (z,)) for coord in env.ul, env.lr]
        xs, ys = [range(p1, p2 + 1) for p1, p2 in zip(*corners)]
        for coord in itertools.product(xs, ys, (z,)):
            yield coord

def to_lonlat(xtile, ytile, zoom):
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

def to_tile(lon, lat, zoom):
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
