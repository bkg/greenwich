"""Spatial reference systems"""
from osgeo import osr


class BaseSpatialReference(osr.SpatialReference):
    """Base class for extending osr.SpatialReference."""

    def __repr__(self):
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


class SpatialReference(object):
    """A spatial reference."""

    def __new__(cls, sref):
        """Returns a new BaseSpatialReference instance

        This allows for customized construction of osr.SpatialReference which
        has no init method which precludes the use of super().
        """
        sr = BaseSpatialReference()
        if isinstance(sref, int):
            sr.ImportFromEPSG(sref)
        elif isinstance(sref, str):
            if sref.strip().startswith('+proj='):
                sr.ImportFromProj4(sref)
            else:
                sr.ImportFromWkt(sref)
            # Add EPSG authority if applicable
            sr.AutoIdentifyEPSG()
        else:
            raise TypeError('Cannot create SpatialReference '
                            'from {}'.format(str(sref)))
        return sr
