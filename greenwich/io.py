"""GDAL IO handling"""
from __future__ import absolute_import
import os
import uuid

from osgeo import gdal


class ImageFileIO(object):
    _vsimem = '/vsimem'

    def __init__(self, basename=None, suffix=None, mode='rb', delete=True):
        basename = (basename or str(uuid.uuid4())) + (suffix or '')
        self.name = os.path.join(self._vsimem, basename)
        vsif = gdal.VSIFOpenL(self.name, 'wb')
        gdal.VSIFCloseL(vsif)
        self.vsif = gdal.VSIFOpenL(self.name, mode)
        self.closed = not self.readable()

    def __del__(self):
        self.close()

    #def __iter__(self):

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return True

    def close(self):
        if not self.closed:
            gdal.VSIFCloseL(self.vsif)
            self.unlink()
            self.closed = True

    def is_temp(self):
        """Returns true if this resides only in memory."""
        return self.name.startswith(self._vsimem)

    def read(self, n=-1):
        #_complain_ifclosed(self.closed)
        if n is None or n < 0:
            fstat = gdal.VSIStatL(self.name)
            n = fstat.size
        return gdal.VSIFReadL(1, n, self.vsif) or ''

    #def readall(self):

    def readable(self):
        if self.vsif is None:
            raise IOError('Could not read from {}'.format(self.name))
        return True

    #def readinto(self, b):
        #Read up to len(b) bytes into bytearray b and return the number of bytes read

    def seek(self, offset, whence=0):
        gdal.VSIFSeekL(self.vsif, offset, whence)
        #TODO: Return the new absolute position as in IOBase.seek

    def seekable(self):
        return True

    def tell(self):
        return gdal.VSIFTellL(self.vsif)

    #def truncate(self, size=None):
        #gdal.VSIFTruncateL

    def unlink(self):
        """Delete the file or vsimem path."""
        gdal.Unlink(self.name)

    def write(self):
        raise io.UnsupportedOperation(
            '%s.write() is not supported' % self.__class__.__name__)

    def writable(self):
        return False
