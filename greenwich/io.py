"""GDAL IO handling"""
from __future__ import absolute_import
import os
from urlparse import urlparse
import uuid

from osgeo import gdal

VSI_SCHEMES = {'http': '/vsicurl/'}
VSI_TYPES = {'.zip': '/vsizip/', '.gz': '/vsigzip/', '.tgz': '/vsitar/'}

def vsiprefix(path):
    """Returns a GDAL virtual filesystem prefixed path.

    Arguments:
    path -- file path as str
    """
    vpath = path.lower()
    scheme = VSI_SCHEMES.get(urlparse(vpath).scheme, '')
    for ext in VSI_TYPES:
        if ext in vpath:
            filesys = VSI_TYPES[ext]
            break
    else:
        filesys = ''
    if filesys and scheme:
        filesys = filesys[:-1]
    return ''.join((filesys, scheme, path))


class VSIFile(object):
    """Implement IO interface for GDAL VSI file."""

    def __init__(self, name, mode='rb'):
        self.name = name
        self.mode = mode
        self.closed = False
        self._vsif = gdal.VSIFOpenL(name, mode)
        if self._vsif is None:
            self.closed = True
            raise IOError('Could not open "%s"' % self.name)

    def __del__(self):
        # Modules may be unloaded at program exit prior to this method
        # call, so silence these errors.
        try:
            self.close()
        except AttributeError:
            pass

    def __enter__(self):
        self._check_closed()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return True

    def __iter__(self):
        self._check_closed()
        return self

    def next(self):
        line = self.readline()
        if not line:
            raise StopIteration
        return line

    def __repr__(self):
        status = 'closed' if self.closed else 'open'
        repstr = '<%s: %s %r, mode "%s">'
        return repstr % (self.__class__.__name__, status, self.name, self.mode)

    def _check_closed(self):
        if self.closed:
            raise ValueError('I/O operation on closed file')

    def close(self):
        if not self.closed:
            self.closed = True
            gdal.VSIFCloseL(self._vsif)

    def read(self, n=-1):
        self._check_closed()
        if n is None or n < 0:
            fstat = gdal.VSIStatL(self.name)
            n = fstat.size
        return gdal.VSIFReadL(1, n, self._vsif) or ''

    def readable(self):
        return set(self.mode) & {'r', '+'} and not self.closed

    def readinto(self, b):
        # Read up to len(b) bytes into bytearray b and return the number of
        # bytes read.
        data = self.read(len(b))
        size = len(data)
        b[:size] = data
        return size

    def readline(self, limit=-1):
        res = bytearray()
        while limit < 0 or len(res) < limit:
            b = self.read(1)
            if not b:
                break
            res += b
            if res.endswith(b'\n'):
                break
        return bytes(res)

    def seek(self, offset, whence=0):
        self._check_closed()
        gdal.VSIFSeekL(self._vsif, offset, whence)

    def seekable(self):
        return True

    def tell(self):
        self._check_closed()
        return gdal.VSIFTellL(self._vsif)

    def truncate(self, pos=None):
        self._check_closed()
        if pos is None:
            pos = self.tell()
        gdal.VSIFTruncateL(self._vsif, pos)
        return pos

    def write(self, data):
        self._check_closed()
        if isinstance(data, bytearray):
            data = bytes(data)
        count = len(data)
        if gdal.VSIFWriteL(data, 1, count, self._vsif) != count:
            raise IOError('Failed writing to "%s"' % self.name)

    def writable(self):
        return set(self.mode) & {'w', '+'} and not self.closed


class MemFileIO(VSIFile):
    """Implement IO interface for GDAL VSI file in memory which will be freed
    on close by default.
    """
    _vpath = '/vsimem'

    def __init__(self, basename=None, suffix=None, mode='w+b', delete=True):
        basename = (basename or str(uuid.uuid4())) + (suffix or '')
        name = os.path.join(self._vpath, basename)
        super(MemFileIO, self).__init__(name, mode)
        self.delete = delete

    def close(self):
        if not self.closed:
            self.closed = True
            gdal.VSIFCloseL(self._vsif)
            # Free allocated memory.
            if self.delete:
                gdal.Unlink(self.name)

    def getvalue(self):
        if self.tell() > 0:
            self.seek(0)
        return self.read()

    def readable(self):
        # Opened mem files are always readable regardless of mode.
        return not self.closed
