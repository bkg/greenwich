"""Raster data handling"""
import os
import xml.etree.cElementTree as ET

import numpy as np
from PIL import Image, ImageDraw
from osgeo import gdal, gdalconst

from greenwich.io import MemFileIO
from greenwich.geometry import Envelope
from greenwich.srs import SpatialReference

def available_drivers():
    """Returns a dictionary of enabled GDAL Driver metadata keyed by the
    'ShortName' attribute.
    """
    drivers = {}
    for i in range(gdal.GetDriverCount()):
        d = gdal.GetDriver(i)
        drivers[d.ShortName] = d.GetMetadata()
    return drivers

def driver_for_path(path, drivers=None):
    """Returns the gdal.Driver for a path or None based on the file extension.

    Arguments:
    path -- file path as str with a GDAL supported file extension
    """
    ext = (os.path.splitext(path)[1][1:] or path).lower()
    drivers = drivers or ImageDriver.registry if ext else {}
    for name, meta in drivers.items():
        if ext == meta.get('DMD_EXTENSION', '').lower():
            return ImageDriver(name)
    return None

def driverdict_tolist(d):
    """Returns a GDAL formatted options list of strings from a dict."""
    return ['%s=%s' % (k, v) for k, v in d.items()]

def geom_to_array(geom, size, affine):
    """Converts an OGR polygon to a 2D NumPy array.

    Arguments:
    geom -- OGR Polygon or MultiPolygon
    size -- array size in pixels as a tuple of (width, height)
    affine -- AffineTransform
    """
    img = Image.new('L', size, 1)
    draw = ImageDraw.Draw(img)
    # MultiPolygon or Polygon with interior rings don't need another level of
    # nesting, but non-donut Polygons do.
    if geom.GetGeometryCount() <= 1:
        geom = [geom]
    for g in geom:
        if g.GetCoordinateDimension() > 2:
            g.FlattenTo2D()
        boundary = g.Boundary()
        coords = boundary.GetPoints() if boundary else g.GetPoints()
        draw.polygon(affine.transform(coords), 0)
    return np.asarray(img)

def count_unique(arr):
    """Returns a two-tuple of pixel count and bin value for every unique pixel
    value in the array.

    Arguments:
    arr -- numpy ndarray
    """
    return [a.tolist() for a in np.histogram(arr, np.unique(arr))]


class AffineTransform(object):
    """Affine transformation between projected and pixel coordinate spaces."""

    def __init__(self, ul_x, scale_x, rx, ul_y, ry, scale_y):
        """Generally this will be initialized from a 5-element tuple in the
        format returned by GetGeoTransform().

        Arguments:
        ul_x -- top left corner x coordinate
        scale_x -- x scaling
        rx -- x rotation
        ul_y -- top left corner y coordinate
        ry -- y rotation
        scale_y -- y scaling
        """
        # Origin coordinate in projected space.
        self.origin = (ul_x, ul_y)
        self.scale_x = scale_x
        self.scale_y = scale_y
        # Rotation in X and Y directions. (0, 0) is north up.
        self.rotation = (rx, ry)

    def __repr__(self):
        return '<%s %r>' % (self.__class__.__name__, self.tuple)

    def __eq__(self, another):
        return self.tuple == getattr(another, 'tuple', None)

    def __ne__(self, another):
        return not self.__eq__(another)

    #def __getitem__(self, idx):
        #return self.tuple[idx]

    def transform_to_projected(self, coords):
        """Convert image pixel/line coordinates to georeferenced x/y, return a
        generator of two-tuples.

        Arguments:
        coords -- input coordinates as iterable containing two-tuples/lists
        such as ((0, 0), (10, 10))
        """
        geotransform = self.tuple
        for x, y in coords:
            geo_x = geotransform[0] + geotransform[1] * x + geotransform[2] * y
            geo_y = geotransform[3] + geotransform[4] * x + geotransform[5] * y
            # Move the coordinate to the center of the pixel.
            geo_x += geotransform[1] / 2.0
            geo_y += geotransform[5] / 2.0
            yield geo_x, geo_y

    def transform(self, coords):
        """Transform from projection coordinates (Xp,Yp) space to pixel/line
        (P,L) raster space, based on the provided geotransformation.

        Arguments:
        coords -- input coordinates as iterable containing two-tuples/lists
        such as ((-120, 38), (-121, 39))
        """
        # Use local vars for better performance here.
        origin = self.origin
        sx = self.scale_x
        sy = self.scale_y
        return [(int((x - origin[0]) / sx), int((y - origin[1]) / sy))
                for x, y in coords]

    @property
    def tuple(self):
        start = self.origin
        rx, ry = self.rotation
        return (start[0], self.scale_x, rx, start[1], ry, self.scale_y)


class ImageDriver(object):
    """Wrap gdal.Driver"""
    # GDAL driver default creation options.
    defaults = {'img': {'compressed': 'yes'},
                'nc': {'compress': 'deflate'},
                'tif': {'tiled': 'yes', 'compress': 'packbits'}}
    registry = available_drivers()
    _writekey = 'DCAP_CREATE'

    def __init__(self, driver='GTiff', **kwargs):
        """
        Keyword args:
        driver -- str gdal.Driver name like 'GTiff' or gdal.Driver instance
        kwargs -- GDAL raster creation options
        """
        if isinstance(driver, str):
            driver = gdal.GetDriverByName(driver) or driver
        if not isinstance(driver, gdal.Driver):
            raise TypeError('No GDAL driver for %s' % driver)
        self._driver = driver
        # The default raster creation options to use.
        self.settings = kwargs or self.defaults.get(self.ext, {})
        self._options = None
        self.writeable = self._writekey in self.info
        self.copyable = 'DCAP_CREATECOPY' in self.info

    def __getattr__(self, attr):
        return getattr(self._driver, attr)

    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, self._driver.ShortName)

    def copy(self, source, dest):
        """Returns a copied Raster instance.

        Arguments:
        source -- the source Raster instance or filepath as str
        dest -- destination filepath as str
        """
        if not self.copyable:
            raise IOError('Driver does not support raster copying')
        if not isinstance(source, Raster):
            source = Raster(source)
        if source.name == dest:
            raise ValueError(
                'Input and output are the same location: %s' % source.name)
        settings = driverdict_tolist(self.settings)
        ds = self.CreateCopy(dest, source.ds, options=settings)
        return Raster(ds)

    def Create(self, *args, **kwargs):
        """Calls Driver.Create() with optionally provided creation options as
        dict, or falls back to driver specific defaults.
        """
        if not self.writeable:
            raise IOError('Driver does not support raster creation')
        options = kwargs.pop('options', {})
        kwargs['options'] = driverdict_tolist(options or self.settings)
        return self._driver.Create(*args, **kwargs)

    @classmethod
    def filter_writeable(cls):
        """Return a dict of drivers supporting raster writes."""
        return {k: v for k, v in cls.registry.items() if cls._writekey in v}

    @property
    def options(self):
        """Returns a dict of driver specific raster creation options.

        See GDAL format docs at http://www.gdal.org/formats_list.html
        """
        if self._options is None:
            try:
                elem = ET.fromstring(
                    self.info.get('DMD_CREATIONOPTIONLIST', ''))
            except ET.ParseError:
                elem = []
            opts = {}
            for child in elem:
                choices = [val.text for val in child]
                if choices:
                    child.attrib.update(choices=choices)
                opts[child.attrib.pop('name')] = child.attrib
            self._options = opts
        return self._options

    def _is_empty(self, path):
        """Returns True if file is empty or non-existent."""
        try:
            return os.path.getsize(path) == 0
        except OSError:
            # File does not even exist
            return True

    def raster(self, path, shape, bandtype=gdal.GDT_Byte):
        """Returns a new Raster instance.

        gdal.Driver.Create() does not support all formats.

        Arguments:
        path -- file object or path as str
        shape -- two or three-tuple of (xsize, ysize, bandcount)
        bandtype -- GDAL pixel data type
        """
        path = getattr(path, 'name', path)
        if len(shape) == 2:
            shape += (1,)
        nx, ny, bandcount = shape
        if nx < 0 or ny < 0:
            raise ValueError('Size cannot be negative')
        # Do not write to a non-empty file.
        if not self._is_empty(path):
            raise IOError('%s already exists, open with Raster()' % path)
        ds = self.Create(path, nx, ny, bandcount, bandtype)
        if not ds:
            raise ValueError(
                'Could not create %s using %s' % (path, str(self)))
        return Raster(ds)

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


class Raster(object):
    """Wrap a GDAL Dataset with additional behavior."""

    def __init__(self, dataset, mode=gdalconst.GA_ReadOnly):
        """Initialize a Raster data set from a path or file

        Arguments:
        dataset -- path as str or file object
        Keyword args:
        mode -- gdal constant representing access mode
        """
        # Get the name if we have a file object.
        dataset = getattr(dataset, 'name', dataset)
        if not isinstance(dataset, gdal.Dataset):
            dataset = gdal.Open(dataset, mode)
        if dataset is None:
            raise IOError('Could not open %s' % dataset)
        self.ds = dataset
        self.name = self.ds.GetDescription()
        # Bands are not zero based, available bands are a 1-based list of ints.
        self.bandlist = range(1, len(self) + 1)
        self.affine = AffineTransform(*self.GetGeoTransform())
        self.sref = SpatialReference(dataset.GetProjection())
        self._nodata = None
        self._envelope = None
        self._driver = None
        self.closed = False
        # Closes the gdal.Dataset
        dataset = None

    def __getattr__(self, attr):
        """Delegate calls to the gdal.Dataset."""
        try:
            return getattr(self.ds, attr)
        except AttributeError:
            if self.closed:
                raise ValueError('Operation on closed raster file')
            raise AttributeError(
                '%s has no attribute "%s"' % (self.__class__.__name__, attr))

    def __getitem__(self, i):
        """Returns a single Band instance.

        This is a zero-based index which matches Python list behavior but
        differs from the GDAL one-based approach of handling multiband images.
        """
        i += 1
        band = self.ds.GetRasterBand(i)
        if not band:
            raise IndexError('No band for %s' % i)
        return band

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return True

    def __len__(self):
        return self.ds.RasterCount

    def __eq__(self, another):
        if type(another) is type(self):
            return self.__dict__ == another.__dict__
        return False

    def __ne__(self, another):
        return not self.__eq__(another)

    def __repr__(self):
        status = 'closed' if self.closed else 'open'
        return '<%s: %s %r>' % (self.__class__.__name__, status, self.name)

    def array(self, envelope=()):
        """Returns an NDArray, optionally subset by spatial envelope.

        Keyword args:
        envelope -- coordinate extent tuple or Envelope
        """
        args = ()
        if envelope:
            args = self.get_offset(envelope)
        return self.ds.ReadAsArray(*args)

    def close(self):
        """Close the GDAL dataset."""
        # De-ref the GDAL Dataset to completely close it.
        self.ds = None
        self.closed = True

    def clip(self, geom):
        """Returns a new raster instance clipped to a particular geometry.

        Arguments:
        geom -- OGR Polygon or MultiPolygon
        """
        return self._mask(geom)

    def crop(self, bbox):
        """Returns a new raster instance cropped to a bounding box.

        Arguments:
        bbox -- bounding box as an OGR Polygon
        """
        return self._mask(bbox)

    @property
    def envelope(self):
        """Returns the minimum bounding rectangle as a tuple of min X, min Y,
        max X, max Y.
        """
        if self._envelope is None:
            origin = self.affine.origin
            ur_x = origin[0] + self.ds.RasterXSize * self.affine.scale_x
            ll_y = origin[1] + self.ds.RasterYSize * self.affine.scale_y
            self._envelope = Envelope(origin[0], ll_y, ur_x, origin[1])
        return self._envelope

    def frombytes(self, bytedata):
        w, h = self.size
        self.ds.WriteRaster(0, 0, w, h, bytedata, band_list=self.bandlist)

    def get_offset(self, envelope):
        """Returns a 4-tuple pixel window (x_offset, y_offset, x_size, y_size).

        Arguments:
        envelope -- coordinate extent tuple or Envelope
        """
        if isinstance(envelope, tuple):
            envelope = Envelope(*envelope)
        if not (self.envelope.contains(envelope) or
                self.envelope.intersects(envelope)):
            raise ValueError('Envelope does not intersect with this extent')
        ul_px, lr_px = self.affine.transform((envelope.ul, envelope.lr))
        nx = min(lr_px[0] - ul_px[0], self.ds.RasterXSize - ul_px[0])
        ny = min(lr_px[1] - ul_px[1], self.ds.RasterYSize - ul_px[1])
        return ul_px + (nx, ny)

    @property
    def driver(self):
        """Returns the underlying ImageDriver instance."""
        if self._driver is None:
            self._driver = ImageDriver(self.ds.GetDriver())
        return self._driver

    def new(self, pixeldata=None, size=(), affine=None):
        """Derive new Raster instances.

        Keyword args:
        pixeldata -- bytestring containing pixel data
        size -- tuple of image size (width, height)
        affine -- affine transformation tuple
        """
        size = size or self.size + (len(self),)
        band = self.GetRasterBand(1)
        imgio = MemFileIO(suffix='.%s' % self.driver.ext)
        rcopy = self.driver.raster(imgio.name, size, bandtype=band.DataType)
        imgio.close()
        rcopy.SetProjection(self.GetProjection())
        rcopy.SetGeoTransform(affine or self.GetGeoTransform())
        colors = band.GetColorTable()
        for outband in rcopy:
            if self.nodata is not None:
                outband.SetNoDataValue(self.nodata)
            if colors:
                outband.SetColorTable(colors)
        if pixeldata:
            rcopy.frombytes(pixeldata)
        return rcopy

    def _mask(self, geom):
        geom = self._transform_maskgeom(geom)
        env = Envelope.from_geom(geom)
        readargs = self.get_offset(env)
        dims = readargs[2:4]
        affine = AffineTransform(*self.GetGeoTransform())
        # Update origin coordinate for the new affine transformation.
        affine.origin = env.ul
        # Without a simple envelope, this becomes a masking operation rather
        # than a crop.
        if not geom.Equals(env.to_geom()):
            arr = self.ds.ReadAsArray(*readargs)
            mask_arr = geom_to_array(geom, dims, affine)
            m = np.ma.masked_array(arr, mask=mask_arr)
            #m.set_fill_value(self.nodata)
            if self.nodata is not None:
                m = np.ma.masked_values(m, self.nodata)
            pixbuf = str(np.getbuffer(m.filled()))
        else:
            pixbuf = self.ds.ReadRaster(*readargs)
        clone = self.new(pixbuf, dims, affine.tuple)
        return clone

    def masked_array(self, envelope=()):
        """Returns a MaskedArray using nodata values.

        Keyword args:
        envelope -- coordinate extent tuple or Envelope
        """
        arr = self.array(envelope)
        if self.nodata is None:
            return np.ma.masked_array(arr)
        return np.ma.masked_values(arr, self.nodata)

    @property
    def nodata(self):
        """Returns read only property for band nodata value, assuming single
        band rasters for now.
        """
        if self._nodata is None:
            self._nodata = self[0].GetNoDataValue()
        return self._nodata

    def ReadRaster(self, *args, **kwargs):
        """Returns a string of raster data for partial or full extent.

        Overrides gdal.Dataset.ReadRaster() with the full raster size by
        default.
        """
        if len(args) < 4:
            args = (0, 0, self.ds.RasterXSize, self.ds.RasterYSize)
        return self.ds.ReadRaster(*args, **kwargs)

    def resample(self, size, interpolation=gdalconst.GRA_NearestNeighbour):
        """Returns a new instance resampled to provided size.

        Arguments:
        size -- tuple of x,y image dimensions
        """
        # Find the scaling factor for pixel size.
        factors = (size[0] / float(self.RasterXSize),
                   size[1] / float(self.RasterYSize))
        affine = AffineTransform(*self.GetGeoTransform())
        affine.scale_x *= factors[0]
        affine.scale_y *= factors[1]
        dest = self.new(size=size, affine=affine.tuple)
        # Uses self and dest projection when set to None
        gdal.ReprojectImage(self.ds, dest.ds, None, None, interpolation)
        return dest

    def save(self, to, driver=None):
        """Save this instance to the path and format provided.

        Arguments:
        to -- output path as str, file, or MemFileIO instance
        Keyword args:
        driver -- GDAL driver name as string or ImageDriver
        """
        path = getattr(to, 'name', to)
        if not driver and isinstance(path, str):
            driver = driver_for_path(path, self.driver.filter_writeable())
        elif isinstance(driver, str):
            driver = ImageDriver(driver)
        if driver is None:
            raise ValueError('Driver not found for %s' % path)
        r = driver.copy(self, path)
        r.close()

    def SetProjection(self, to_sref):
        if not hasattr(to_sref, 'ExportToWkt'):
            to_sref = SpatialReference(to_sref)
        self.sref = to_sref
        self.ds.SetProjection(to_sref.ExportToWkt())

    def SetGeoTransform(self, geotrans_tuple):
        """Sets the affine transformation."""
        self.affine = AffineTransform(*geotrans_tuple)
        self.ds.SetGeoTransform(geotrans_tuple)

    @property
    def shape(self):
        """Returns a tuple of row, column, (band count if multidimensional)."""
        shp = (self.ds.RasterYSize, self.ds.RasterXSize, self.ds.RasterCount)
        return shp[:2] if shp[2] <= 1 else shp

    @property
    def size(self):
        """Returns a 2-tuple of (width, height) in pixels."""
        return (self.ds.RasterXSize, self.ds.RasterYSize)

    def _transform_maskgeom(self, geom):
        if isinstance(geom, Envelope):
            geom = geom.to_geom()
        geom_sref = geom.GetSpatialReference()
        if geom_sref is None:
            raise Exception('Cannot transform from unknown spatial reference')
        # Reproject geom if necessary
        if not geom_sref.IsSame(self.sref):
            geom = geom.Clone()
            geom.TransformTo(self.sref)
        return geom

    def warp(self, to_sref, dest=None, interpolation=gdalconst.GRA_NearestNeighbour):
        """Returns a new reprojected instance.

        Arguments:
        to_sref -- spatial reference as a proj4 or wkt string, or a
        SpatialReference
        Keyword args:
        dest -- filepath as str
        interpolation -- GDAL interpolation type
        """
        if not hasattr(to_sref, 'ExportToWkt'):
            to_sref = SpatialReference(to_sref)
        dest_wkt = to_sref.ExportToWkt()
        dtype = self[0].DataType
        err_thresh = 0.125
        # Determine new values for destination raster dimensions and
        # geotransform.
        vrt = gdal.AutoCreateWarpedVRT(self.ds, None, dest_wkt,
                                       interpolation, err_thresh)
        warpsize = (vrt.RasterXSize, vrt.RasterYSize, len(self))
        warptrans = vrt.GetGeoTransform()
        vrt = None
        imgio = dest or MemFileIO()
        newrast = self.driver.raster(imgio, warpsize, dtype)
        if isinstance(imgio, MemFileIO):
            imgio.close()
        newrast.SetGeoTransform(warptrans)
        newrast.SetProjection(to_sref)
        for band in newrast:
            band.SetNoDataValue(self.nodata)
            band = None
        # Uses self and newrast projection when set to None
        gdal.ReprojectImage(self.ds, newrast.ds, None, None, interpolation)
        return newrast


# Alias the raster constructor as open().
open = Raster

def frombytes(data, size, bandtype=gdal.GDT_Byte):
    r = ImageDriver('MEM').raster('', size, bandtype)
    r.frombytes(data)
    return r
