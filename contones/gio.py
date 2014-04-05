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
    path -- file path as str with a GDAL support file extension
    """
    path = path or ''
    extsep = os.path.extsep
    ext = (path.rsplit(extsep, 1)[-1] if extsep in path else path).lower()
    avail = available_drivers() if ext else {}
    for k, v in avail.items():
        avail_ext = v.get('DMD_EXTENSION')
        if ext == avail_ext:
            return gdal.GetDriverByName(k)
    return None


class ImageIO(object):
    """File or memory (VSIMEM) backed IO for GDAL datasets.

    GDAL does not integrate with file-like objects but provides its own
    mechanisms for handling IO.
    """
    _vsimem = '/vsimem'
    # GDAL driver default creation options.
    drivers = {'tif': ['TILED=YES', 'COMPRESS=PACKBITS'],
               'img': ['COMPRESSED=YES']}

    def __init__(self, path=None, driver=None):
        """
        Keyword args:
        path -- str GDALDriver name like 'GTiff' or path to a new raster
        dataset like '/data/test.tif'
        driver -- GDALDriver instance
        """
        # Use geotiff as the default when path and driver are not provided.
        if not any((path, driver)):
            driver = 'GTiff'
        if isinstance(driver, str):
            driver = gdal.GetDriverByName(driver)
        if driver is None:
            driver = driver_for_path(path)
        if not isinstance(driver, gdal.Driver):
            raise Exception('No GDAL driver for {}'.format(path))
        self._driver = driver
        self.driver_opts = self.drivers.get(self.ext, [])
        self.path = path or self._tempname()

    def __getattr__(self, attr):
        return getattr(self._driver, attr)

    def __repr__(self):
        return '{}: {}'.format(self.__class__.__name__, str(self.info))

    def is_empty(self):
        """Returns True if file is empty."""
        try:
            return os.path.getsize(self.path) == 0
        except OSError:
            # File does not even exist
            return True

    def create(self, nx, ny, bandcount=1, datatype=gdal.GDT_Byte,
               options=None):
        """Returns a new Raster instance.

        gdal.Driver.Create() does not support all formats.
        """
        # Do not write to a non-empty file.
        if not self.is_empty():
            errmsg = '{0} already exists, open with Raster({0})'.format(self.path)
            raise IOError(errmsg)
        if nx < 0 or ny < 0:
            raise ValueError('Size cannot be negative')
        ds = self.Create(self.path, nx, ny, bandcount,
                         datatype, options or self.driver_opts)
        if not ds:
            raise ValueError(
                'Could not create {} using {}'.format(self.path, str(self)))
        return contones.raster.Raster(ds)

    def _tempname(self):
        basename = '{}.{}'.format(str(uuid.uuid4()), self.ext)
        return os.path.join(self._vsimem, basename)

    def copyfrom(self, dataset, options=None):
        """Returns a copied Raster instance."""
        if self.path == dataset.GetDescription():
            raise ValueError(
                'Input and output are the same location: {}'.format(self.path))
        ds = self.CreateCopy(self.path, dataset.ds,
                             options=options or self.driver_opts)
        return contones.raster.Raster(ds)

    def getvalue(self):
        """Returns the raster data buffer as a byte string."""
        f = gdal.VSIFOpenL(self.path, 'rb')
        if f is None:
            raise IOError('Could not read from {}'.format(self.path))
        fstat = gdal.VSIStatL(self.path)
        data = gdal.VSIFReadL(1, fstat.size, f)
        gdal.VSIFCloseL(f)
        return data

    def unlink(self):
        """Delete the file or vsimem path."""
        gdal.Unlink(self.path)

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
