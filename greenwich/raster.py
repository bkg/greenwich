"""Raster data handling"""
from __future__ import division

import os
import collections
import math
import xml.etree.cElementTree as ET

import numpy as np
from osgeo import gdal, gdal_array, gdalconst, ogr

from greenwich.base import Comparable
from greenwich.io import MemFileIO, vsiprefix
from greenwich.geometry import transform, Envelope
from greenwich.layer import MemoryLayer
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

def geom_to_array(geom, size, affine, options=None):
    """Converts an OGR polygon to a 2D NumPy array.

    Arguments:
    geom -- OGR Geometry
    size -- array size in pixels as a tuple of (width, height)
    affine -- AffineTransform
    """
    options = options or []
    driver = ImageDriver('MEM')
    rast = driver.raster(driver.ShortName, size)
    rast.affine = affine
    rast.sref = geom.GetSpatialReference()
    with MemoryLayer.from_records([(1, geom)]) as ml:
        status = gdal.RasterizeLayer(rast.ds, (1,), ml.layer, burn_values=(1,),
                                     options=options)
    arr = rast.array()
    rast.close()
    return arr

def rasterize(layer, rast):
    """Returns a Raster from layer features.

    Arguments:
    layer -- Layer to rasterize
    rast -- Raster with target affine, size, and sref
    """
    driver = ImageDriver('MEM')
    r2 = driver.raster(driver.ShortName, rast.size)
    r2.affine = rast.affine
    sref = rast.sref
    if not sref.srid:
        sref = SpatialReference(4326)
    r2.sref = sref
    ml = MemoryLayer(sref, layer.GetGeomType())
    ml.load(layer)
    status = gdal.RasterizeLayer(
        r2.ds, (1,), ml.layer, options=['ATTRIBUTE=%s' % ml.id])
    ml.close()
    return r2

def count_unique(arr):
    """Returns a list of two-tuples with unique value and occurrence count.

    Arguments:
    arr -- numpy ndarray
    """
    return zip(*np.unique(arr, return_counts=True))


class AffineTransform(Comparable):
    """Affine transformation between projected and pixel coordinate spaces."""

    def __init__(self, xorigin, xscale, rx, yorigin, ry, yscale):
        """Generally this will be initialized from a six-element tuple in the
        format returned by gdal.Dataset.GetGeoTransform().

        Arguments:
        xorigin -- top left corner x coordinate
        xscale -- x scaling
        rx -- x rotation
        yorigin -- top left corner y coordinate
        ry -- y rotation
        yscale -- y scaling
        """
        # Origin coordinate in projected space.
        self.origin = (xorigin, yorigin)
        self.scale = (xscale, yscale)
        # Rotation in X and Y directions. (0, 0) is north up.
        self.rotation = (rx, ry)
        # Avoid repeated calls to tuple() by iterators and slices.
        self._len = len(self.tuple)

    def __getitem__(self, index):
        return self.tuple[index]

    def __iter__(self):
        for val in self.tuple:
            yield val

    def __len__(self):
        return self._len

    def __repr__(self):
        return '<%s %r>' % (self.__class__.__name__, self.tuple)

    def project(self, coords):
        """Convert image pixel/line coordinates to georeferenced x/y, return a
        generator of two-tuples.

        Arguments:
        coords -- input coordinates as iterable containing two-tuples/lists
        such as ((0, 0), (10, 10))
        """
        easting, sx, rx, northing, sy, ry = self.tuple
        return [(easting + sx * x + rx * y,
                 northing + sy * x + ry * y)
                for x, y in coords]

    def transform(self, coords):
        """Transform from projection coordinates (Xp,Yp) space to pixel/line
        (P,L) raster space, based on the provided geotransformation.

        Arguments:
        coords -- input coordinates as iterable containing two-tuples/lists
        such as ((-120, 38), (-121, 39))
        """
        # Use local vars for better performance here.
        origin_x, origin_y = self.origin
        sx, sy = self.scale
        return [(int(math.floor((x - origin_x) / sx)),
                 int(math.floor((y - origin_y) / sy)))
                for x, y in coords]

    @property
    def tuple(self):
        start = self.origin
        rx, ry = self.rotation
        sx, sy = self.scale
        return (start[0], sx, rx, start[1], ry, sy)


class ImageDriver(object):
    """Wrap gdal.Driver"""
    _copykey = 'DCAP_CREATECOPY'
    _writekey = 'DCAP_CREATE'
    # GDAL driver default creation options.
    defaults = {'img': {'compressed': 'yes'},
                'nc': {'compress': 'deflate'},
                'tif': {'tiled': 'yes', 'compress': 'packbits'},
                'xyz': {'column_separator': ',', 'add_header_line': 'yes'}}
    registry = available_drivers()

    def __init__(self, driver='GTiff', strictmode=True, **kwargs):
        """
        Keyword args:
        driver -- str gdal.Driver name like 'GTiff' or gdal.Driver instance
        kwargs -- GDAL raster creation options
        """
        if hasattr(driver, 'format'):
            driver = gdal.GetDriverByName(str(driver)) or driver
        if not isinstance(driver, gdal.Driver):
            raise TypeError('No GDAL driver for %s' % driver)
        self._driver = driver
        # The default raster creation options to use.
        self.settings = kwargs or self.defaults.get(self.ext, {})
        self.strictmode = strictmode
        self._options = None
        self.writable = self._writekey in self.info
        self.copyable = self._copykey in self.info

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
            should_close = True
        else:
            should_close = False
        if source.name == dest:
            raise ValueError(
                'Input and output are the same location: %s' % source.name)
        settings = driverdict_tolist(self.settings)
        ds = self.CreateCopy(dest, source.ds, self.strictmode,
                             options=settings)
        if should_close:
            source.close()
        return Raster(ds)

    def Create(self, *args, **kwargs):
        """Calls Driver.Create() with optionally provided creation options as
        dict, or falls back to driver specific defaults.
        """
        if not self.writable:
            raise IOError('Driver does not support raster creation')
        options = kwargs.pop('options', {})
        kwargs['options'] = driverdict_tolist(options or self.settings)
        return self._driver.Create(*args, **kwargs)

    @classmethod
    def _filter_by(cls, key):
        return {k: v for k, v in cls.registry.items() if key in v}

    @classmethod
    def filter_writable(cls):
        """Return a dict of drivers supporting raster writes."""
        return cls._filter_by(cls._writekey)

    @classmethod
    def filter_copyable(cls):
        """Return a dict of drivers supporting raster copies."""
        return cls._filter_by(cls._copykey)

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

    def raster(self, path, size, bandtype=gdal.GDT_Byte):
        """Returns a new Raster instance.

        gdal.Driver.Create() does not support all formats.

        Arguments:
        path -- file object or path as str
        size -- two or three-tuple of (xsize, ysize, bandcount)
        bandtype -- GDAL pixel data type
        """
        path = getattr(path, 'name', path)
        try:
            is_multiband = len(size) > 2
            nx, ny, nbands = size if is_multiband else size + (1,)
        except (TypeError, ValueError) as exc:
            exc.args = ('Size must be 2 or 3-item sequence',)
            raise
        if nx < 1 or ny < 1:
            raise ValueError('Invalid raster size %s' % (size,))
        # Do not write to a non-empty file.
        if not self._is_empty(path):
            raise IOError('%s already exists, open with Raster()' % path)
        ds = self.Create(path, nx, ny, nbands, bandtype)
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


class ImageWindow(collections.namedtuple('ImageWindow', 'x y width height')):
    """A window of image coordinates for reading a raster."""
    __slots__ = ()

    @property
    def dims(self):
        return (self.width, self.height)

    @property
    def origin(self):
        return (self.x, self.y)


class Raster(Comparable):
    """Wrap a GDAL Dataset with additional behavior."""

    def __init__(self, path, mode=gdalconst.GA_ReadOnly):
        """Initialize a Raster data set from a path or file

        Arguments:
        path -- path as str, file object, or gdal.Dataset
        Keyword args:
        mode -- gdal constant representing access mode
        """
        if path and not isinstance(path, gdal.Dataset):
            # Get the name if we have a file-like object.
            dataset = gdal.Open(getattr(path, 'name', path), mode)
        else:
            dataset = path
        if not dataset:
            raise IOError('Failed to open: "%s"' % path)
        self.ds = dataset
        self.name = self.ds.GetDescription()
        # Bands are not zero based, available bands are a 1-based list of ints.
        self.bandlist = range(1, len(self) + 1)
        # Initialize attrs without calling their setters.
        self._affine = AffineTransform(*dataset.GetGeoTransform())
        self._sref = SpatialReference(dataset.GetProjection())
        #self.dtype = gdal_array.codes[self[0].DataType]
        self._nodata = None
        self._envelope = None
        self._driver = None
        self.closed = False
        # Closes the gdal.Dataset
        dataset = None
        self.mask_opts = []

    def __getattr__(self, attr):
        """Delegate calls to the gdal.Dataset."""
        try:
            return getattr(self.ds, attr)
        except AttributeError:
            if self.closed:
                raise ValueError('Operation on closed raster file')
            raise AttributeError(
                '%s has no attribute "%s"' % (self.__class__.__name__, attr))

    def __getitem__(self, index):
        """Returns a single Band instance.

        This is a zero-based index which matches Python list behavior but
        differs from the GDAL one-based approach of handling multiband images.
        """
        if index < 0:
            index += len(self)
        index += 1
        band = self.ds.GetRasterBand(index)
        if not band:
            raise IndexError('No band for "%s"' % index)
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

    def __repr__(self):
        status = 'closed' if self.closed else 'open'
        return '<%s: %s %r>' % (self.__class__.__name__, status, self.name)

    def _get_affine(self):
        return self._affine

    def SetGeoTransform(self, affine):
        """Sets the affine transformation.

        Intercepts the gdal.Dataset call to ensure use as a property setter.

        Arguments:
        affine -- AffineTransform or six-tuple of geotransformation values
        """
        if isinstance(affine, collections.Sequence):
            affine = AffineTransform(*affine)
        self._affine = affine
        self.ds.SetGeoTransform(affine)

    affine = property(_get_affine, SetGeoTransform)

    def array(self, envelope=()):
        """Returns an NDArray, optionally subset by spatial envelope.

        Keyword args:
        envelope -- coordinate extent tuple or Envelope
        """
        args = ()
        if envelope:
            args = self.get_window(envelope)
        return self.ds.ReadAsArray(*args)

    def close(self):
        """Close the GDAL dataset."""
        # De-ref the GDAL Dataset to completely close it.
        self.ds = None
        self.closed = True

    def clip(self, geom):
        """Returns a new Raster instance clipped and masked to a geometry.

        Arguments:
        geom -- OGR Polygon or MultiPolygon
        """
        return self._subset(geom)

    def crop(self, bbox):
        """Returns a new raster instance cropped to a bounding box.

        Arguments:
        bbox -- bounding box as an OGR Polygon, Envelope, or tuple
        """
        return self._subset(bbox)

    @property
    def envelope(self):
        """Returns the minimum bounding rectangle as a tuple of min X, min Y,
        max X, max Y.
        """
        if self._envelope is None:
            origin = self.affine.origin
            ur_x = origin[0] + self.ds.RasterXSize * self.affine.scale[0]
            ll_y = origin[1] + self.ds.RasterYSize * self.affine.scale[1]
            self._envelope = Envelope(origin[0], ll_y, ur_x, origin[1])
        return self._envelope

    def frombytes(self, bytedata):
        w, h = self.size
        self.ds.WriteRaster(0, 0, w, h, bytedata, band_list=self.bandlist)

    def get_window(self, envelope):
        """Returns a 4-tuple pixel window (x_offset, y_offset, x_size, y_size).

        Arguments:
        envelope -- coordinate extent tuple or Envelope
        """
        if isinstance(envelope, collections.Sequence):
            envelope = Envelope(envelope)
        if not (self.envelope.contains(envelope) or
                self.envelope.intersects(envelope)):
            raise ValueError('Envelope does not intersect with this extent')
        coords = self.affine.transform((envelope.ul, envelope.lr))
        nxy = [(min(dest, size - 1) + 1 - max(origin, 0))
               for size, origin, dest in zip(self.size, *coords)]
        return ImageWindow(*coords[0] + tuple(nxy))

    @property
    def driver(self):
        """Returns the underlying ImageDriver instance."""
        if self._driver is None:
            self._driver = ImageDriver(self.ds.GetDriver())
        return self._driver

    def new(self, size=(), affine=None):
        """Derive new Raster instances.

        Keyword args:
        size -- tuple of image size (width, height)
        affine -- AffineTransform or six-tuple of geotransformation values
        """
        size = size or self.size + (len(self),)
        band = self.ds.GetRasterBand(1)
        driver = ImageDriver('MEM')
        rcopy = driver.raster(driver.ShortName, size, band.DataType)
        rcopy.sref = self.GetProjection()
        rcopy.affine = affine or tuple(self.affine)
        colors = band.GetColorTable()
        for outband in rcopy:
            if self.nodata is not None:
                outband.SetNoDataValue(self.nodata)
            if colors:
                outband.SetColorTable(colors)
        return rcopy

    def _subset(self, geom):
        geom = transform(geom, self.sref)
        env = Envelope.from_geom(geom).intersect(self.envelope)
        win = self.get_window(env)
        affine = AffineTransform(*tuple(self.affine))
        # Update origin coordinate for the new affine transformation.
        affine.origin = self.affine.project([win.origin])[0]
        # Without an envelope or point, this becomes a masking operation.
        if not geom.Equals(env.polygon) and geom.GetGeometryType() != ogr.wkbPoint:
            arr = self._masked_array(env)
            # This will broadcast whereas np.ma.masked_array() does not.
            arr.mask = ~np.ma.make_mask(
                geom_to_array(geom, win.dims, affine, self.mask_opts))
            pixbuf = bytes(arr.filled().data)
        else:
            pixbuf = self.ds.ReadRaster(*win)
        clone = self.new(win.dims + (len(self),), affine)
        clone.frombytes(pixbuf)
        return clone

    def _masked_array(self, envelope=()):
        arr = self.array(envelope)
        if self.nodata is not None:
            if not np.isnan(self.nodata):
                return np.ma.masked_values(arr, self.nodata, copy=False)
            else:
                marr = np.ma.masked_invalid(arr, copy=False)
                marr.set_fill_value(self.nodata)
                return marr
        return np.ma.masked_array(arr, copy=False)

    def masked_array(self, geometry=None):
        """Returns a MaskedArray using nodata values.

        Keyword args:
        geometry -- any geometry, envelope, or coordinate extent tuple
        """
        if geometry is None:
            return self._masked_array()
        geom = transform(geometry, self.sref)
        env = Envelope.from_geom(geom).intersect(self.envelope)
        arr = self._masked_array(env)
        if geom.GetGeometryType() != ogr.wkbPoint:
            win = self.get_window(env)
            affine = AffineTransform(*tuple(self.affine))
            affine.origin = self.affine.project([win.origin])[0]
            mask = ~np.ma.make_mask(
                geom_to_array(geom, win.dims, affine, self.mask_opts))
            arr.mask = arr.mask | mask
        return arr

    @property
    def nodata(self):
        """Returns property for band nodata value, assuming single band rasters
        for now.
        """
        if self._nodata is None:
            self._nodata = self[0].GetNoDataValue()
        return self._nodata

    @nodata.setter
    def nodata(self, val):
        for band in self:
            band.SetNoDataValue(val)

    def ReadRaster(self, *args, **kwargs):
        """Returns raster data bytes for partial or full extent.

        Overrides gdal.Dataset.ReadRaster() with the full raster size by
        default.
        """
        args = args or (0, 0, self.ds.RasterXSize, self.ds.RasterYSize)
        return self.ds.ReadRaster(*args, **kwargs)

    def resample(self, size, interpolation=gdalconst.GRA_NearestNeighbour):
        """Returns a new instance resampled to provided size.

        Arguments:
        size -- tuple of x,y image dimensions
        """
        # Find the scaling factor for pixel size.
        factors = (size[0] / float(self.RasterXSize),
                   size[1] / float(self.RasterYSize))
        affine = AffineTransform(*tuple(self.affine))
        affine.scale = (affine.scale[0] / factors[0],
                        affine.scale[1] / factors[1])
        dest = self.new(size, affine)
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
        if not driver and hasattr(path, 'encode'):
            driver = driver_for_path(path, self.driver.filter_copyable())
        elif hasattr(driver, 'encode'):
            driver = ImageDriver(driver)
        if driver is None or not driver.copyable:
            raise ValueError('Copy supporting driver not found for %s' % path)
        driver.copy(self, path).close()

    def _get_sref(self):
        return self._sref

    def SetProjection(self, sref):
        """Sets the spatial reference.

        Intercepts the gdal.Dataset call to ensure use as a property setter.

        Arguments:
        sref -- SpatialReference or any format supported by the constructor
        """
        if not hasattr(sref, 'ExportToWkt'):
            sref = SpatialReference(sref)
        self._sref = sref
        self.ds.SetProjection(sref.ExportToWkt())

    sref = property(_get_sref, SetProjection)

    @property
    def shape(self):
        """Returns a tuple of row, column, (band count if multidimensional)."""
        shp = (self.ds.RasterYSize, self.ds.RasterXSize, self.ds.RasterCount)
        return shp[:2] if shp[2] <= 1 else shp

    @property
    def size(self):
        """Returns a 2-tuple of (width, height) in pixels."""
        return (self.ds.RasterXSize, self.ds.RasterYSize)

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
        if vrt is None:
            raise ValueError('Could not warp %s to %s' % (self, dest_wkt))
        warpsize = (vrt.RasterXSize, vrt.RasterYSize, len(self))
        warptrans = vrt.GetGeoTransform()
        vrt = None
        if dest is None:
            imgio = MemFileIO()
            rwarp = self.driver.raster(imgio, warpsize, dtype)
            imgio.close()
        else:
            rwarp = self.driver.raster(dest, warpsize, dtype)
        rwarp.SetGeoTransform(warptrans)
        rwarp.SetProjection(to_sref)
        if self.nodata is not None:
            for band in rwarp:
                band.Fill(self.nodata)
                band.SetNoDataValue(self.nodata)
                band = None
        # Uses self and rwarp projection when set to None
        gdal.ReprojectImage(self.ds, rwarp.ds, None, None, interpolation)
        return rwarp


def open(path, mode=gdalconst.GA_ReadOnly):
    """Returns a Raster instance.

    Arguments:
    path -- local or remote path as str or file-like object
    Keyword args:
    mode -- gdal constant representing access mode
    """
    path = getattr(path, 'name', path)
    try:
        return Raster(vsiprefix(path), mode)
    except AttributeError:
        try:
            imgdata = path.read()
        except AttributeError:
            raise TypeError('Not a file-like object providing read()')
        else:
            imgio = MemFileIO(delete=False)
            gdal.FileFromMemBuffer(imgio.name, imgdata)
            return Raster(imgio, mode)
    raise ValueError('Failed to open raster from "%r"' % path)

def fromarray(arr):
    """Returns an in-memory raster from a numpy array."""
    bandtype = gdal_array.flip_code(arr.dtype.type)
    r = frombytes(arr.tobytes(), tuple(reversed(arr.shape)), bandtype)
    if hasattr(arr, 'fill_value'):
        r.nodata = arr.fill_value
    return r

def frombytes(data, size, bandtype=gdal.GDT_Byte):
    """Returns an in-memory raster initialized from a pixel buffer.

    Arguments:
    data -- byte buffer of raw pixel data
    size -- two or three-tuple of (xsize, ysize, bandcount)
    bandtype -- band data type
    """
    r = ImageDriver('MEM').raster('', size, bandtype)
    r.frombytes(data)
    return r
