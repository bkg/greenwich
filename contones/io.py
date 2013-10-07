import os
import uuid

from osgeo import gdal
import contones.raster


# TODO: Generalize and replace _run_encoder()
def convert(inpath, outpath=None, geom=None):
    if outpath is None:
        outpath = get_imageio_for(outpath)()
    with contones.raster.Raster(inpath) as r:
        r.save(outpath)
    return outpath

def get_imageio_for(path):
    """Returns the io class from a file path or gdal.Driver ShortName."""
    extsep = os.path.extsep
    ext = path.rsplit(extsep, 1)[-1] if extsep in path else path
    #ext = os.path.splitext(path)[-1] if extsep in path else path
    for cls in BaseImageIO.__subclasses__():
        if ext in [cls.ext, cls.driver_name]:
            return cls
    raise Exception('No IO class for {}'.format(path))


# TODO: Work with all GDAL types
# TODO: These not strictly encoders as they have filepaths, etc. Rename to
# Transformer, Converter, Driver? Or, FileStore, ImageFile, ImageFileStore?
#class BaseEncoder(object):
#class BaseImageStore(object):
class BaseImageIO(object):
    """Base encoder for GDAL Datasets derived from GDAL.Driver, used mainly
    for raster image encoding. New raster formats should subclass this.
    """
    _vsimem = '/vsimem'
    # Specify this in subclass
    driver_name = None
    driver_opts = []
    ext = None

    def __init__(self, path=None):
        self.driver = gdal.GetDriverByName(self.driver_name)
        self.path = path or self.get_tmpname()

    def __getattr__(self, attr):
        return getattr(self.driver, attr)

    def create(self, nx, ny, bandcount, datatype):
        self._check_exists()
        ds = self.Create(self.path, nx, ny, bandcount,
                         datatype, self.driver_opts)
        return contones.raster.Raster(ds)

    #def vsipath(self):
    def get_tmpname(self):
        basename = '{}.{}'.format(str(uuid.uuid4()), self.ext)
        return os.path.join(self._vsimem, basename)

    def _check_exists(self):
        if os.path.exists(self.path):
            raise IOError('{} already exists'.format(self.path))

    def copy_from(self, dataset):
        #self._check_exists()
        ds = self.CreateCopy(self.path, dataset.ds,
                             options=self.driver_opts)
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


#class ImageIO(object):
class ImageIO(BaseImageIO):
    # GDAL driver defaults
    drivers = {'GTiff': ('tif', ['COMPRESS=PACKBITS']),
               'HFA': ('img', ['COMPRESSED=YES'])}

    def __init__(self, path=None, driver=None):
        if path and driver is None:
            driver = self.driver_for_path(path)
        #if not driver:
        if not isinstance(driver, gdal.Driver):
            raise Exception('No GDAL driver for {}'.format(path))
        self.driver = driver
        opts = self.drivers.get(self.driver.ShortName)
        self.driver_opts = opts[1] if opts else []
        self.path = path or self.get_tmpname()
        self._ext = None

    @property
    def ext(self):
        #k = self.driver.ShortName
        #self.defaults.get(k, k.lower().replace(' ', ''))
        try:
            #self.ext = opts[0]
            #self.ext = self.driver_opts[0]
            self._ext = self.drivers[self.driver.ShortName][0]
        except KeyError:
            self._ext = self.driver.ShortName.lower().replace(' ', '')
        return self._ext

    def driver_for_path(self, path):
        extsep = os.path.extsep
        ext = path.rsplit(extsep, 1)[-1] if extsep in path else path
        return gdal.GetDriverByName(self.drivers.get(ext, 'GTiff'))


class GeoTIFFEncoder(BaseImageIO):
    """GeoTIFF raster encoder."""
    driver_name = 'GTiff'
    driver_opts = ['COMPRESS=PACKBITS']
    ext = 'tif'


class HFAEncoder(BaseImageIO):
    """Erdas Imagine raster encoder."""
    driver_name = 'HFA'
    driver_opts = ['COMPRESSED=YES']
    ext = 'img'
