"""GDAL IO handling"""
from __future__ import absolute_import
import os
import uuid

from osgeo import gdal


class MemFileIO(object):
    """Implement IO interface for GDAL VSI file in memory."""
    _vsimem = '/vsimem'

    def __init__(self, basename=None, suffix=None, mode='w+b'):
        basename = (basename or str(uuid.uuid4())) + (suffix or '')
        self.name = os.path.join(self._vsimem, basename)
        self.vsif = gdal.VSIFOpenL(self.name, mode)
        self.mode = mode
        self.closed = not self.readable()

    def __del__(self):
        self.close()

    def __enter__(self):
        self._check_closed()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return True

    def __repr__(self):
        return '<%s %s>' % (self.__class__.__name__, self.mode)

    def _check_closed(self):
        if self.closed:
            raise ValueError('I/O operation on closed file')

    def close(self):
        if not self.closed:
            gdal.VSIFCloseL(self.vsif)
            # Free allocated memory.
            gdal.Unlink(self.name)
            self.closed = True

    def read(self, n=-1):
        self._check_closed()
        if n is None or n < 0:
            fstat = gdal.VSIStatL(self.name)
            n = fstat.size
        return gdal.VSIFReadL(1, n, self.vsif) or ''

    def readable(self):
        if self.vsif is None:
            raise IOError('Could not read from %s' % self.name)
        return True

    def readinto(self, b):
        # Read up to len(b) bytes into bytearray b and return the number of
        # bytes read.
        data = self.read(len(b))
        size = len(data)
        b[:size] = data
        return size

    def seek(self, offset, whence=0):
        self._check_closed()
        gdal.VSIFSeekL(self.vsif, offset, whence)

    def seekable(self):
        return True

    def tell(self):
        self._check_closed()
        return gdal.VSIFTellL(self.vsif)

    def truncate(self, pos=None):
        self._check_closed()
        if pos is None:
            pos = self.tell()
        gdal.VSIFTruncateL(self.vsif, pos)
        return pos

    def write(self, data):
        self._check_closed()
        if isinstance(data, bytearray):
            data = bytes(data)
        gdal.VSIFWriteL(data, len(data), 1, self.vsif)

    def writable(self):
        return True
