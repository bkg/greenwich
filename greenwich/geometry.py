from osgeo import ogr
try:
    import simplejson as json
except ImportError:
    import json

from greenwich.base import Comparable
from greenwich.srs import SpatialReference

def transform(geom, to_sref):
    """Returns a transformed Geometry.

    Arguments:
    geom -- any coercible Geometry value or Envelope
    to_sref -- SpatialReference
    """
    # If we have an envelope, assume it's in the target sref.
    try:
        geom = getattr(geom, 'polygon', Envelope(geom).polygon)
    except (TypeError, ValueError):
        pass
    else:
        geom.AssignSpatialReference(to_sref)
    try:
        geom_sref = geom.GetSpatialReference()
    except AttributeError:
        return transform(Geometry(geom), to_sref)
    if geom_sref is None:
        raise Exception('Cannot transform from unknown spatial reference')
    # Reproject geom if necessary
    if not geom_sref.IsSame(to_sref):
        geom = geom.Clone()
        geom.TransformTo(to_sref)
    return geom


class Envelope(Comparable):
    """Rectangular bounding extent.

    This class closely resembles OGREnvelope which is not included in the SWIG
    bindings.
    """

    def __init__(self, *args):
        """Creates an envelope from lower-left and upper-right coordinates.

        Arguments:
        args -- min_x, min_y, max_x, max_y or a four-tuple
        """
        if len(args) == 1:
            args = args[0]
        try:
            extent = map(float, args)
        except (TypeError, ValueError) as exc:
            exc.args = ('Cannot create Envelope from "%s"' % repr(args),)
            raise
        try:
            self.min_x, self.max_x = sorted(extent[::2])
            self.min_y, self.max_y = sorted(extent[1::2])
        except ValueError as exc:
            exc.args = ('Sequence length should be "4", not "%d"' % len(args),)
            raise

    def __add__(self, other):
        combined = Envelope(tuple(self))
        combined.expand(other)
        return combined

    def __contains__(self, other):
        return self.contains(other)

    def __getitem__(self, index):
        return self.tuple[index]

    def __iter__(self):
        for val in self.tuple:
            yield val

    def __len__(self):
        return len(self.__dict__)

    def __mul__(self, factor):
        return self.scale(factor)

    def __repr__(self):
        return '<%s: %r>' % (self.__class__.__name__, self.tuple)

    def __sub__(self, other):
        return self.intersect(other)

    @property
    def centroid(self):
        """Returns the envelope centroid as a (x, y) tuple."""
        return self.min_x + self.width * 0.5, self.min_y + self.height * 0.5

    def contains(self, other):
        """Returns true if this envelope contains another.

        Arguments:
        other -- Envelope or tuple of (minX, minY, maxX, maxY)
        """
        try:
            return (self.min_x <= other.min_x and
                    self.min_y <= other.min_y and
                    self.max_x >= other.max_x and
                    self.max_y >= other.max_y)
        except AttributeError:
            # Perhaps we have a tuple, try again with an Envelope.
            return self.contains(Envelope(other))

    def expand(self, other):
        """Expands this envelope by the given Envelope or tuple.

        Arguments:
        other -- Envelope, two-tuple, or four-tuple
        """
        if len(other) == 2:
            other += other
        mid = len(other) / 2
        self.ll = map(min, self.ll, other[:mid])
        self.ur = map(max, self.ur, other[mid:])

    @staticmethod
    def from_geom(geom):
        """Returns an Envelope from an OGR Geometry."""
        extent = geom.GetEnvelope()
        return Envelope(map(extent.__getitem__, (0, 2, 1, 3)))

    @property
    def height(self):
        return self.max_y - self.min_y

    def intersect(self, other):
        """Returns the intersection of this and another Envelope."""
        inter = Envelope(tuple(self))
        if inter.intersects(other):
            mid = len(other) / 2
            inter.ll = map(max, inter.ll, other[:mid])
            inter.ur = map(min, inter.ur, other[mid:])
        else:
            inter.ll = (0, 0)
            inter.ur = (0, 0)
        return inter

    def intersects(self, other):
        """Returns true if this envelope intersects another.

        Arguments:
        other -- Envelope or tuple of (minX, minY, maxX, maxY)
        """
        try:
            return (self.min_x <= other.max_x and
                    self.max_x >= other.min_x and
                    self.min_y <= other.max_y and
                    self.max_y >= other.min_y)
        except AttributeError:
            return self.intersects(Envelope(other))

    @property
    def ll(self):
        """Returns the lower left coordinate."""
        return self.min_x, self.min_y

    @ll.setter
    def ll(self, coord):
        """Set lower-left from (x, y) tuple."""
        self.min_x, self.min_y = coord

    @property
    def lr(self):
        """Returns the lower right coordinate."""
        return self.max_x, self.min_y

    @lr.setter
    def lr(self, coord):
        self.max_x, self.min_y = coord

    def scale(self, xfactor, yfactor=None):
        """Returns a new envelope rescaled from center by the given factor(s).

        Arguments:
        xfactor -- int or float X scaling factor
        yfactor -- int or float Y scaling factor
        """
        yfactor = xfactor if yfactor is None else yfactor
        x, y = self.centroid
        xshift = self.width * xfactor * 0.5
        yshift = self.height * yfactor * 0.5
        return Envelope(x - xshift, y - yshift, x + xshift, y + yshift)

    @property
    def polygon(self):
        """Returns an OGR Geometry for this envelope."""
        ring = ogr.Geometry(ogr.wkbLinearRing)
        for coord in self.ll, self.lr, self.ur, self.ul, self.ll:
            ring.AddPoint_2D(*coord)
        polyg = ogr.Geometry(ogr.wkbPolygon)
        polyg.AddGeometryDirectly(ring)
        return polyg

    @property
    def tuple(self):
        """Returns the maximum extent as a tuple."""
        return self.ll + self.ur

    @property
    def ul(self):
        """Returns the upper left coordinate."""
        return self.min_x, self.max_y

    @ul.setter
    def ul(self, coord):
        self.min_x, self.max_y = coord

    @property
    def ur(self):
        """Returns the upper right coordinate."""
        return self.max_x, self.max_y

    @ur.setter
    def ur(self, coord):
        """Returns the upper right coordinate."""
        self.max_x, self.max_y = coord

    @property
    def width(self):
        return self.max_x - self.min_x


@property
def __geo_interface__(self):
    return json.loads(self.ExportToJson())

# Monkey-patch ogr.Geometry to provide geo-interface support.
ogr.Geometry.__geo_interface__ = __geo_interface__

def Geometry(*args, **kwargs):
    """Returns an ogr.Geometry instance optionally created from a geojson str
    or dict. The spatial reference may also be provided.
    """
    # Look for geojson as a positional or keyword arg.
    arg = kwargs.pop('geojson', None) or len(args) and args[0]
    try:
        srs = kwargs.pop('srs', None) or arg.srs.wkt
    except AttributeError:
        srs = SpatialReference(4326)
    if hasattr(arg, 'keys'):
        geom = ogr.CreateGeometryFromJson(json.dumps(arg))
    elif hasattr(arg, 'startswith'):
        if arg.startswith('{'):
            geom = ogr.CreateGeometryFromJson(arg)
        # WKB as hexadecimal string.
        elif ord(arg[0]) in [0, 1]:
            geom = ogr.CreateGeometryFromWkb(arg)
        elif arg.startswith('<gml'):
            geom = ogr.CreateGeometryFromGML(arg)
    elif hasattr(arg, 'wkb'):
        geom = ogr.CreateGeometryFromWkb(bytes(arg.wkb))
    else:
        geom = ogr.Geometry(*args, **kwargs)
    if geom:
        if not isinstance(srs, SpatialReference):
            srs = SpatialReference(srs)
        geom.AssignSpatialReference(srs)
    return geom
