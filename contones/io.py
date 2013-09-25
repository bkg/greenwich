import multiprocessing
import os
import uuid

from osgeo import gdal
import contones.raster

def _run_encoder(path, encoder_cls, geom=None):
    encoder = encoder_cls()
    with contones.raster.Raster(path) as r:
        if geom:
            with r.crop(geom) as cropped:
                cropped.save(encoder)
        else:
            r.save(encoder)
    buff = encoder.read()
    # Remove the dataset from memory
    encoder.unlink()
    return buff

# TODO: Generalize and replace _run_encoder()
def convert(inpath, outpath=None):
    if outpath is None:
        outpath = get_imageio_for(outpath)()
    with contones.raster.Raster(path) as r:
        r.save(outpath)
    return outpath

def run_encoderpool(encoder_cls, pathnames, geom=None, multicore=True):
    """Run an encoder job using a pool of workers.

    Arguments:
    path -- path to a GDAL dataset
    encoder_cls -- encoder class to use, not an instance

    Keyword args:
    geom -- geometry used to crop raster as a geos.Polygon or None
    multicore -- true/false, process in parallel by default
    """
    encoder = encoder_cls()
    if not multicore:
        return [_run_encoder(path, encoder_cls, geom) for path in pathnames]
    num_workers = multiprocessing.cpu_count()
    num_workers = num_workers / 2 if num_workers > 4 else num_workers
    pool = multiprocessing.Pool(num_workers)
    results = [pool.apply(_run_encoder, (path, encoder_cls, geom,))
               for path in pathnames]
    pool.close()
    return results

def get_imageio_for(path):
    """Returns the io class from a file path or gdal.Driver ShortName."""
    extsep = os.path.extsep
    ext = path.rsplit(extsep, 1)[-1] if extsep in path else path
    #ext = os.path.splitext(path)[-1] if extsep in path else path
    for cls in BaseImageIO.__subclasses__():
        if ext in [cls.ext, cls.driver_name]:
            return cls
    raise Exception('No IO class for {}'.format(path))


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
        #self._check_exists()
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
