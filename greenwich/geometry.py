from osgeo import ogr
try:
    import simplejson as json
except ImportError:
    import json

from greenwich.base import Comparable
from greenwich.srs import SpatialReference


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
            exc.args = ('Cannot create Envelope from "%s"' % args,)
            raise
        try:
            self.min_x, self.max_x = sorted(extent[::2])
            self.min_y, self.max_y = sorted(extent[1::2])
        except ValueError as exc:
            exc.args = ('Sequence length should be "4", not "%d"' % len(args),)
            raise

    def __add__(self, envp):
        combined = Envelope(tuple(self))
        combined.expand(envp)
        return combined

    def __contains__(self, envp):
        return self.contains(envp)

    def __getitem__(self, index):
        return self.tuple[index]

    def __iter__(self):
        for val in self.tuple:
            yield val

    def __len__(self):
        return len(self.__dict__)

    def __repr__(self):
        return '<%s: %r>' % (self.__class__.__name__, self.tuple)

    def __sub__(self, envp):
        return self.intersect(envp)

    def contains(self, envp):
        """Returns true if this envelope contains another.

        Arguments:
        envp -- Envelope or tuple of (minX, minY, maxX, maxY)
        """
        try:
            return self.ll <= envp.ll and self.ur >= envp.ur
        except AttributeError:
            # Perhaps we have a tuple, try again with an Envelope.
            return self.contains(Envelope(envp))

    def clip(self, envp):
        mid = len(self) / 2
        self.ll = map(max, self.ll, envp[:mid])
        self.ur = map(min, self.ur, envp[mid:])

    def expand(self, envp):
        """Expands this envelope by the given Envelope or tuple.

        Arguments:
        envp -- Envelope, two-tuple, or four-tuple
        """
        if len(envp) == 2:
            envp += envp
        mid = len(self) / 2
        self.ll = map(min, self.ll, envp[:mid])
        self.ur = map(max, self.ur, envp[mid:])

    @staticmethod
    def from_geom(geom):
        """Returns an Envelope from an OGR Geometry."""
        extent = geom.GetEnvelope()
        return Envelope(map(extent.__getitem__, (0, 2, 1, 3)))

    @property
    def height(self):
        return self.max_y - self.min_y

    def intersect(self, envp):
        """Returns the intersection of this and another Envelope."""
        intersection = Envelope(tuple(self))
        intersection.clip(envp)
        return intersection

    def intersects(self, envp):
        """Returns true if this envelope intersects another.

        Arguments:
        envp -- Envelope or tuple of (minX, minY, maxX, maxY)
        """
        try:
            return self.ll <= envp.ur and self.ur >= envp.ll
        except AttributeError:
            return self.intersects(Envelope(envp))

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
