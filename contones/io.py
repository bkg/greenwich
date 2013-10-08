import os
import uuid

from osgeo import gdal
import contones.raster

def available_drivers():
    drivers = {}
    for i in range(gdal.GetDriverCount()):
        d = gdal.GetDriver(i)
        drivers[d.ShortName] = d.GetMetadata()
        d = None
    return drivers

def convert(inpath, outpath=None, geom=None):
    if outpath is None:
        outpath = ImageIO()
    with contones.raster.Raster(inpath) as r:
        r.save(outpath)
    return outpath


class ImageIO(object):
    """Base encoder for GDAL Datasets derived from GDAL.Driver, used mainly
    for raster image encoding. New raster formats should subclass this.
    """
    _vsimem = '/vsimem'
    # GDAL driver default creation options.
    drivers = {'tif': ['COMPRESS=PACKBITS'],
               'img': ['COMPRESSED=YES']}

    def __init__(self, path=None, driver=None):
        """
        Keyword args:
        path -- str GDALDriver name like 'GTiff' or path to a new raster
        dataset like '/data/test.tif'
        driver -- GDALDriver instance
        """
        if isinstance(driver, str):
            driver = gdal.GetDriverByName(driver)
        if driver is None:
            driver = self.driver_for_path(path)
        if not isinstance(driver, gdal.Driver):
            raise Exception('No GDAL driver for {}'.format(path))
        self.driver = driver
        self.driver_opts = self.drivers.get(self.ext, [])
        self.path = path or self.get_tmpname()

    def __getattr__(self, attr):
        return getattr(self.driver, attr)

    def create(self, nx, ny, bandcount, datatype, options=None):
        self._check_exists()
        ds = self.Create(self.path, nx, ny, bandcount,
                         datatype, options or self.driver_opts)
        return contones.raster.Raster(ds)

    #def vsipath(self):
    def get_tmpname(self):
        basename = '{}.{}'.format(str(uuid.uuid4()), self.ext)
        return os.path.join(self._vsimem, basename)

    def _check_exists(self):
        if os.path.exists(self.path):
            raise IOError('{} already exists'.format(self.path))

    def copy_from(self, dataset, options=None):
        #self._check_exists()
        ds = self.CreateCopy(self.path, dataset.ds,
                             options=options or self.driver_opts)
        return contones.raster.Raster(ds)

    def read(self, size=0):
        """Returns the raster data buffer as str."""
        f = gdal.VSIFOpenL(self.path, 'rb')
        if f is None:
            raise IOError('Could not read from {}'.format(self.path))
        fstat = gdal.VSIStatL(self.path)
        data = gdal.VSIFReadL(1, fstat.size, f)
        gdal.VSIFCloseL(f)
        return data

    def unlink(self):
        gdal.Unlink(self.path)

    @property
    def info(self):
        return self.driver.GetMetadata()

    @property
    def ext(self):
        return self.info.get('DMD_EXTENSION', '')

    @property
    def mimetype(self):
        return self.info.get('DMD_MIMETYPE', 'application/octet-stream')

    def driver_for_path(self, path):
        path = path or ''
        extsep = os.path.extsep
        ext = (path.rsplit(extsep, 1)[-1] if extsep in path else path).lower()
        avail = available_drivers() if ext else {}
        drivername = 'GTiff'
        for k, v in avail.items():
            avail_ext = v.get('DMD_EXTENSION')
            if ext == avail_ext:
                drivername = k
                break
        return gdal.GetDriverByName(drivername)
