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



class ImageDriver(object):
    # GDAL driver default creation options.
    defaults = {'tif': ['TILED=YES', 'COMPRESS=PACKBITS'],
                'img': ['COMPRESSED=YES']}
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
        self.options = self.defaults.get(self.ext, [])

    def __getattr__(self, attr):
        return getattr(self._driver, attr)

    def __repr__(self):
        return '{}: {}'.format(self.__class__.__name__, str(self.info))

    def copy(self, source, dest=None, options=None):
        """Returns a copied Raster instance.
        Arguments:
        source -- the source Raster instance
        """
        dest = ImageIO(dest)
        if source.name == dest.name:
            raise ValueError(
                'Input and output are the same location: {}'.format(source.name))
        ds = self.CreateCopy(dest.name, source.ds,
                             options=options or self.options)
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
                                options or self.driver.options)
        if not ds:
            raise ValueError(
                'Could not create {} using {}'.format(self.name, str(self)))
        return contones.raster.Raster(ds)

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

    def _tempname(self):
        """Returns a temporary VSI memory filename."""
        basename = '{}.{}'.format(str(uuid.uuid4()), self.driver.ext)
        return os.path.join(self._vsimem, basename)

    def unlink(self):
        """Delete the file or vsimem path."""
        gdal.Unlink(self.name)
