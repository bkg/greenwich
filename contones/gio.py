"""GDAL IO handling"""
import os
import uuid

from osgeo import gdal
import contones.raster

def available_drivers():
    """Returns a dictionary of enabled GDAL Driver metadata keyed by the
    'ShortName' attribute.
    """
    drivers = {}
    for i in range(gdal.GetDriverCount()):
        d = gdal.GetDriver(i)
        drivers[d.ShortName] = d.GetMetadata()
        d = None
    return drivers

def driver_for_path(path):
    """Returns the gdal.Driver for a path or None based on the file extension.

    Arguments:
    path -- file path as str with a GDAL supported file extension
    """
    extsep = os.path.extsep
    ext = (path.rsplit(extsep, 1)[-1] if extsep in path else path).lower()
    avail = ImageDriver.registry if ext else {}
    for k, v in avail.items():
        avail_ext = v.get('DMD_EXTENSION', '').lower()
        if ext == avail_ext:
            return ImageDriver(gdal.GetDriverByName(k))
    return None

def driverdict_tolist(d):
    """Returns a GDAL formatted options list from a dict."""
    return map('='.join, d.items())


class ImageDriver(object):
    """Wrap gdal.Driver"""
    # GDAL driver default creation options.
    defaults = {'img': {'COMPRESSED': 'YES'},
                'nc': {'COMPRESS': 'DEFLATE'},
                'tif': {'TILED': 'YES', 'COMPRESS': 'PACKBITS'}}
    registry = available_drivers()

    def __init__(self, driver=None):
        """
        Keyword args:
        driver -- str GDALDriver name like 'GTiff' or GDALDriver instance
        """
        # Use geotiff as the default when path and driver are not provided.
        if not driver:
            driver = 'GTiff'
        if isinstance(driver, str):
            driver = gdal.GetDriverByName(driver)
        if not isinstance(driver, gdal.Driver):
            raise TypeError('No GDAL driver for {}'.format(driver))
        self._driver = driver
        self.options = self.defaults.get(self.ext, {})

    def __getattr__(self, attr):
        return getattr(self._driver, attr)

    def __repr__(self):
        return '{}: {}'.format(self.__class__.__name__, str(self.info))

    def copy(self, source, dest, options=None):
        """Returns a copied Raster instance.

        Arguments:
        source -- the source Raster instance or filepath as str
        dest -- destination filepath as str
        Keyword args:
        options -- dict of dataset creation options
        """
        if not isinstance(source, contones.raster.Raster):
            source = contones.raster.Raster(source)
        if source.name == dest:
            raise ValueError(
                'Input and output are the same location: {}'.format(source.name))
        options = driverdict_tolist(options or self.options)
        ds = self.CreateCopy(dest, source.ds, options=options)
        return contones.raster.Raster(ds)

    def Create(self, *args, **kwargs):
        """Calls Driver.Create() with optionally provided creation options as
        dict, or falls back to driver specific defaults.
        """
        options = kwargs.pop('options', {})
        kwargs['options'] = driverdict_tolist(options or self.options)
        return self._driver.Create(*args, **kwargs)

    def _is_empty(self, path):
        """Returns True if file is empty or non-existent."""
        try:
            return os.path.getsize(path) == 0
        except OSError:
            # File does not even exist
            return True

    def raster(self, path, shape, datatype=gdal.GDT_Byte, options=None):
        """Returns a new Raster instance.

        gdal.Driver.Create() does not support all formats.

        Arguments:
        path -- file object or path as str
        shape -- two or three-tuple of (xsize, ysize, bandcount)
        datatype -- GDAL pixel data type
        options -- dict of dataset creation options
        """
        path = getattr(path, 'name', path)
        if len(shape) == 2:
            shape += (1,)
        nx, ny, bandcount = shape
        if nx < 0 or ny < 0:
            raise ValueError('Size cannot be negative')
        # Do not write to a non-empty file.
        if not self._is_empty(path):
            errmsg = '{0} already exists, open with Raster({0})'.format(path)
            raise IOError(errmsg)
        ds = self.Create(path, nx, ny, bandcount, datatype, options=options)
        if not ds:
            raise ValueError(
                'Could not create {} using {}'.format(path, str(self)))
        return contones.raster.Raster(ds)

    @property
    def info(self):
        """Returns a dict of gdal.Driver metadata."""
        return self._driver.GetMetadata()

    @property
    def ext(self):
        """Returns the file extension."""
        return self.info.get('DMD_EXTENSION', '')

    @property
    def mimetype(self):
        """Returns the MIME type."""
        return self.info.get('DMD_MIMETYPE', 'application/octet-stream')

    @property
    def format(self):
        return self._driver.ShortName


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
