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

    def copy(self, source, dest=None, options=None):
        """Returns a copied Raster instance.

        Arguments:
        source -- the source Raster instance
        Keyword args:
        dest -- filepath as str or ImageIO instance
        options -- dict of dataset creation options
        """
        dest = dest or ImageIO(dest)
        if source.name == dest.name:
            raise ValueError(
                'Input and output are the same location: {}'.format(source.name))
        options = driverdict_tolist(options or self.options)
        ds = self.CreateCopy(dest.name, source.ds, options=options)
        return contones.raster.Raster(ds)

    def Create(self, *args, **kwargs):
        """Calls Driver.Create() with optionally provided creation options as
        dict, or falls back to driver specific defaults.
        """
        options = kwargs.pop('options', {})
        kwargs['options'] = driverdict_tolist(options or self.options)
        ds = self._driver.Create(*args, **kwargs)
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


class ImageIO(object):
    """File or memory (VSIMEM) backed IO for GDAL datasets.

    GDAL does not integrate with file-like objects but provides its own
    mechanisms for handling IO.

    path -- path to a raster dataset like '/tmp/test.tif'
    driver -- name as string like 'JPEG' or a gdal.Driver instance
    """
    _vsimem = '/vsimem'

    def __init__(self, path=None, driver=None):
        path = getattr(path, 'name', path)
        if path:
            self.driver = driver_for_path(path)
            self.name = path
        else:
            self.driver = ImageDriver(driver)
            self.name = self._tempname()
        self.closed = False

    def __del__(self):
        self.close()

    def close(self):
        #gdal.VSIFCloseL(self.name)
        if self.is_temp() and not self.closed:
            self.unlink()
        self.closed = True

    #def create_raster
    def create(self, shape, datatype=gdal.GDT_Byte, options=None):
        """Returns a new Raster instance.

        gdal.Driver.Create() does not support all formats.

        Arguments:
        shape -- two or three-tuple of (xsize, ysize, bandcount)
        datatype -- GDAL pixel data type
        options -- dict of dataset creation options
        """
        if len(shape) == 2:
            shape += (1,)
        nx, ny, bandcount = shape
        if nx < 0 or ny < 0:
            raise ValueError('Size cannot be negative')
        # Do not write to a non-empty file.
        if not self._is_empty():
            errmsg = '{0} already exists, open with Raster({0})'.format(self.name)
            raise IOError(errmsg)
        ds = self.driver.Create(self.name, nx, ny, bandcount, datatype,
                                options=options)
        if not ds:
            raise ValueError(
                'Could not create {} using {}'.format(self.name, str(self)))
        return ds

    def getvalue(self):
        """Returns the raster data buffer as a byte string."""
        f = gdal.VSIFOpenL(self.name, 'rb')
        if f is None:
            raise IOError('Could not read from {}'.format(self.name))
        fstat = gdal.VSIStatL(self.name)
        data = gdal.VSIFReadL(1, fstat.size, f)
        gdal.VSIFCloseL(f)
        return data

    def _is_empty(self):
        """Returns True if file is empty or non-existent."""
        try:
            return os.path.getsize(self.name) == 0
        except OSError:
            # File does not even exist
            return True

    def is_temp(self):
        """Returns true if this resides only in memory."""
        return self.name.startswith(self._vsimem)

    def _tempname(self):
        """Returns a temporary VSI memory filename."""
        basename = '{}.{}'.format(str(uuid.uuid4()), self.driver.ext)
        return os.path.join(self._vsimem, basename)

    def unlink(self):
        """Delete the file or vsimem path."""
        gdal.Unlink(self.name)
