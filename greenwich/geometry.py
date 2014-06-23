from osgeo import ogr
try:
    import simplejson as json
except ImportError:
    import json

from greenwich.srs import SpatialReference


class Envelope(object):
    """Rectangular bounding extent.

    This class closely resembles OGREnvelope which is not included in the SWIG
    bindings.
    """

    def __init__(self, min_x, min_y, max_x, max_y):
        """
        Creates an envelope, from lower-left and upper-right coordinates.
        Coordinate pairs for upper-left and lower-right may also be given, they
        will be swapped.
        """
        # Swap values if they are inverted.
        if min_x > max_x and min_y > max_y:
            min_x, max_x = max_x, min_x
            min_y, max_y = max_y, min_y
        self.min_x = min_x
        self.min_y = min_y
        self.max_x = max_x
        self.max_y = max_y
        if self.min_x > self.max_x or self.min_y > self.max_y:
            raise ValueError('Invalid coordinate extent')

    def __add__(self, i):
        return Envelope(*[x + i for x in self])

    def __contains__(self, envp):
        return self.contains(envp)

    def __eq__(self, another):
        return self.tuple == getattr(another, 'tuple', False)

    def __iter__(self):
        for i in self.tuple:
            yield i

    def __repr__(self):
        return '<%s: %r>' % (self.__class__.__name__, self.tuple)

    def __sub__(self, i):
        return self.__add__(-i)

    def contains(self, envp):
        """Returns true if this envelope contains another.

        Arguments:
        envp -- Envelope or tuple of (minX, minY, maxX, maxY)
        """
        try:
            return self.ll <= envp.ll and self.ur >= envp.ur
        except AttributeError:
            # Perhaps we have a tuple, try again with an Envelope.
            return self.contains(Envelope(*envp))

    @staticmethod
    def from_geom(geom):
        """Returns an Envelope from an OGR Geometry."""
        extent = geom.GetEnvelope()
        return Envelope(extent[0], extent[2], extent[1], extent[3])

    @property
    def height(self):
        return self.max_y - self.min_y

    def intersects(self, envp):
        """Returns true if this envelope intersects another.

        Arguments:
        envp -- Envelope or tuple of (minX, minY, maxX, maxY)
        """
        try:
            return self.ll <= envp.ur and self.ur >= envp.ll
        except AttributeError:
            return self.intersects(Envelope(*envp))

    @property
    def ll(self):
        """Returns the lower left coordinate."""
        return self.min_x, self.min_y

    @property
    def lr(self):
        """Returns the lower right coordinate."""
        return self.max_x, self.min_y

    def scale(self, factor_x, factor_y=None):
        """Returns a new envelope rescaled by the given factor(s)."""
        factor_y = factor_x if factor_y is None else factor_y
        w = self.width * factor_x / 2.0
        h = self.height * factor_y / 2.0
        return Envelope(self.min_x + w, self.min_y + h,
                        self.max_x - w, self.max_y - h)

    def to_geom(self):
        """Returns an OGR Geometry for this envelope."""
        ring = ogr.Geometry(ogr.wkbLinearRing)
        for coord in self.ll, self.lr, self.ur, self.ul, self.ll:
            ring.AddPoint(*coord)
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

    @property
    def ur(self):
        """Returns the upper right coordinate."""
        return self.max_x, self.max_y

    @property
    def width(self):
        return self.max_x - self.min_x


def Geometry(*args, **kwargs):
    """Returns an ogr.Geometry instance optionally created from a geojson str
    or dict. The spatial reference may also be provided.
    """
    srs = kwargs.pop('srs', None)
    # Look for geojson as a positional or keyword arg.
    arg = kwargs.pop('geojson', None) or len(args) and args[0]
    if hasattr(arg, 'keys'):
        geom = ogr.CreateGeometryFromJson(json.dumps(arg))
    elif hasattr(arg, 'startswith') and arg.startswith('{'):
        geom = ogr.CreateGeometryFromJson(arg)
    elif hasattr(arg, 'wkb'):
        geom = ogr.CreateGeometryFromWkb(bytes(arg.wkb))
    else:
        geom = ogr.Geometry(*args, **kwargs)
    if srs and geom:
        if not isinstance(srs, SpatialReference):
            srs = SpatialReference(srs)
        geom.AssignSpatialReference(srs)
    return geom
