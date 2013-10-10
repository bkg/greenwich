from osgeo import ogr


class Envelope(object):
    """Rectangular bounding extent.

    This class closely resembles OGREnvelope which is not included in the SWIG
    bindings.
    """

    def __init__(self, min_x, min_y, max_x, max_y):
        self.min_x = min_x
        self.min_y = min_y
        self.max_x = max_x
        self.max_y = max_y
        if not self.min_x < self.max_x and self.min_y < self.max_y:
            raise Exception('Invalid coordinate extent')

    def __repr__(self):
        return str(self.tuple)

    @property
    def ur(self):
        """Returns the upper right coordinate."""
        return self.max_x, self.max_y

    #def lower_right(self):

    @property
    def lr(self):
        """Returns the lower right coordinate."""
        return self.max_x, self.min_y

    @property
    def ll(self):
        """Returns the lower left coordinate."""
        return self.min_x, self.min_y

    @property
    def ul(self):
        """Returns the upper left coordinate."""
        return self.min_x, self.max_y

    @property
    def tuple(self):
        """Returns the maximum extent as a tuple."""
        return self.ll + self.ur

    @property
    def height(self):
        return self.max_y - self.min_y

    @property
    def width(self):
        return self.max_x - self.min_x

    def scale(self, factor_x, factor_y=None):
        """Rescale the envelope by the given factor(s)."""
        factor_y = factor_x if factor_y is None else factor_y
        w = self.width * factor_x / 2.0
        h = self.height * factor_y / 2.0
        self.min_x += w
        self.max_x -= w
        self.min_y += h
        self.max_y -= h

    def to_geom(self):
        """Returns an OGR Geometry for this envelope."""
        ring = ogr.Geometry(ogr.wkbLinearRing)
        for coord in self.ll, self.lr, self.ur, self.ul, self.ll:
            ring.AddPoint(*coord)
        polyg = ogr.Geometry(ogr.wkbPolygon)
        polyg.AddGeometryDirectly(ring)
        return polyg

    @staticmethod
    def from_geom(geom):
        """Returns an Envelope from an OGR Geometry."""
        extent = geom.GetEnvelope()
        return Envelope(extent[0], extent[2], extent[1], extent[3])
