"""Spatial reference systems"""
from osgeo import osr


class SpatialReference(object):
    """Wraps osr.SpatialReference with flexible instance creation."""

    def __init__(self, sref):
        if isinstance(sref, int):
            sr = osr.SpatialReference()
            sr.ImportFromEPSG(sref)
        elif isinstance(sref, str):
            if '+proj=' in sref:
                sr = osr.SpatialReference()
                sr.ImportFromProj4(sref)
            else:
                sr = osr.SpatialReference(sref)
            # Add EPSG authority if applicable
            sr.AutoIdentifyEPSG()
        else:
            raise TypeError('Cannot create SpatialReference '
                            'from {}'.format(str(sref)))
        self._sref = sr

    def __getattr__(self, attr):
        return getattr(self._sref, attr)

    def __repr__(self):
        return self.wkt

    @property
    def srid(self):
        """Returns the EPSG ID as int if it exists."""
        epsg_id = (self._sref.GetAuthorityCode('PROJCS') or
                   self._sref.GetAuthorityCode('GEOGCS'))
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
