"""Spatial reference systems"""
from osgeo import osr


# Monkey patch SpatialReference since inheriting from SWIG classes is a hack
def srid(self):
    """Returns the EPSG ID as int if it exists."""
    epsg_id = (self.GetAuthorityCode('PROJCS') or
                self.GetAuthorityCode('GEOGCS'))
    try:
        return int(epsg_id)
    except TypeError:
        return
osr.SpatialReference.srid = property(srid)

def wkt(self):
    """Returns this projection in WKT format."""
    return self.ExportToWkt()
osr.SpatialReference.wkt = property(wkt)

def proj4(self):
    """Returns this projection as a proj4 string."""
    return self.ExportToProj4()
osr.SpatialReference.proj4 = property(proj4)

def __repr__(self): return self.wkt
osr.SpatialReference.__repr__ = __repr__


class SpatialReference(object):

    def __new__(cls, sref):
        sr = osr.SpatialReference()
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
