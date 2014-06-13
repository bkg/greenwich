"""Spatial reference systems"""
from osgeo import osr


class SpatialReference(osr.SpatialReference):
    """Base class for extending osr.SpatialReference."""

    def __init__(self, sref=None):
        super(SpatialReference, self).__init__()
        if isinstance(sref, int):
            self.ImportFromEPSG(sref)
        elif isinstance(sref, basestring):
            if sref.strip().startswith('+proj='):
                self.ImportFromProj4(sref)
            else:
                self.ImportFromWkt(sref)
            # Add EPSG authority if applicable
            self.AutoIdentifyEPSG()

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
