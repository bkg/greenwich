"""Raster data handling"""
import os

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

def driver_for_path(path):
    """Returns the gdal.Driver for a path or None based on the file extension.

    Arguments:
    path -- file path as str with a GDAL supported file extension
    """
    ext = (os.path.splitext(path)[1][1:] or path).lower()
    drivers = ImageDriver.registry if ext else {}
    for name, meta in drivers.items():
        if ext == meta.get('DMD_EXTENSION', '').lower():
            return ImageDriver(name)
    return None

def driverdict_tolist(d):
    """Returns a GDAL formatted options list from a dict."""
    return map('='.join, d.items())

def geom_to_array(geom, matrix_size, affine):
    """Converts an OGR polygon to a 2D NumPy array.

    Arguments:
    geom -- OGR Polygon or MultiPolygon
    matrix_size -- array size in pixels as a tuple of (width, height)
    affine -- AffineTransform
    """
    img = Image.new('L', matrix_size, 1)
    draw = ImageDraw.Draw(img)
    if not geom.GetGeometryName().startswith('MULTI'):
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

    def __init__(self, ul_x, scale_x, c0, ul_y, c1, scale_y):
        """Generally this will be initialized from a 5-element tuple in the
        format returned by GetGeoTransform().

        Arguments:
        ul_x -- top left corner x coordinate
        scale_x -- x scaling
        c0 -- coefficient
        ul_y -- top left corner y coordinate
        c1 -- coefficient
        scale_y -- y scaling
        """
        # Origin coordinate in projected space.
        self.origin = (ul_x, ul_y)
        self.scale_x = scale_x
        self.scale_y = scale_y
        self.coeffs = (c0, c1)

    def __repr__(self):
        return str(self.tuple)

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
        # Assumes north up images.
        start = self.origin
        c0, c1 = self.coeffs
        return (start[0], self.scale_x, c0, start[1], c1, self.scale_y)


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
        if not isinstance(source, Raster):
            source = Raster(source)
        if source.name == dest:
            raise ValueError(
                'Input and output are the same location: {}'.format(source.name))
        options = driverdict_tolist(options or self.options)
        ds = self.CreateCopy(dest, source.ds, options=options)
        return Raster(ds)

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
        self.affine = AffineTransform(*self.GetGeoTransform())
        self.sref = SpatialReference(dataset.GetProjection())
        self._nodata = None
        self._envelope = None
        self._driver = None
        # Closes the GDALDataset
        dataset = None

    def __getattr__(self, attr):
        """Delegate calls to the GDALDataset."""
        return getattr(self.ds, attr)

    def __getitem__(self, i):
        """Returns a single Band instance.

        This is a one-based index which matches the GDAL approach of handling
        multiband images.
        """
        band = self.GetRasterBand(i)
        if not band:
            raise IndexError('No band for {}'.format(i))
        return band

    #TODO: handle subdataset iteration
    def __iter__(self):
        # Bands are not zero based
        for i in range(1, self.RasterCount + 1):
            yield self[i]

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return True

    def __len__(self):
        return self.RasterCount

    def __eq__(self, another):
        if type(another) is type(self):
            return self.__dict__ == another.__dict__
        return False

    def __ne__(self, another):
        return not self.__eq__(another)

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__,
                               self.ds.GetDescription())

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
            ur_x = origin[0] + self.RasterXSize * self.affine.scale_x
            ll_y = origin[1] + self.RasterYSize * self.affine.scale_y
            self._envelope = Envelope(origin[0], ll_y, ur_x, origin[1])
        return self._envelope

    def get_offset(self, envelope):
        """Returns a 4-tuple pixel window (x_offset, y_offset, x_size, y_size).

        Arguments:
        envelope -- coordinate extent tuple or Envelope
        """
        if isinstance(envelope, tuple):
            envelope = Envelope(*envelope)
        if not (self.envelope.contains(envelope) or
                self.envelope.intersects(envelope)):
            raise ValueError('Envelope does not intersect with this extent.')
        ul_px, lr_px = self.affine.transform((envelope.ul, envelope.lr))
        nx = min(lr_px[0] - ul_px[0], self.RasterXSize - ul_px[0])
        ny = min(lr_px[1] - ul_px[1], self.RasterYSize - ul_px[1])
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
        size = size or self.shape
        band = self.GetRasterBand(1)
        imgio = MemFileIO(suffix=self.driver.ext)
        rcopy = self.driver.raster(imgio.name, size, datatype=band.DataType)
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
            bands = range(1, size[-1] + 1) if len(size) > 2 else None
            args = (0, 0) + size[:2] + (pixeldata,) + size[:2]
            rcopy.WriteRaster(*args, band_list=bands)
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
            self._nodata = self[1].GetNoDataValue()
        return self._nodata

    def ReadRaster(self, *args, **kwargs):
        """Returns a string of raster data for partial or full extent.

        Overrides GDALDataset.ReadRaster() with the full raster size by
        default.
        """
        if len(args) < 4:
            args = (0, 0, self.RasterXSize, self.RasterYSize)
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

    # TODO: allow ImageDriver instances?
    def save(self, to, driver=None):
        """Save this instance to the path and format provided.

        Arguments:
        to -- output path as str, file, or MemFileIO instance
        Keyword args:
        driver -- GDAL driver name as string
        """
        path = getattr(to, 'name', to)
        if driver:
            driver = ImageDriver(driver)
        elif isinstance(path, str):
            driver = driver_for_path(path)
        else:
            raise Exception('Driver not found for %s' % driver or path)
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

    def warp(self, to_sref, interpolation=gdalconst.GRA_NearestNeighbour):
        """Returns a new reprojected instance.

        Arguments:
        to_sref -- spatial reference as a proj4 or wkt string, or a
        SpatialReference
        """
        if not hasattr(to_sref, 'ExportToWkt'):
            to_sref = SpatialReference(to_sref)
        dest_wkt = to_sref.ExportToWkt()
        dtype = self[1].DataType
        err_thresh = 0.125
        # Call AutoCreateWarpedVRT() to fetch default values for target raster
        # dimensions and geotransform
        # src_wkt : left to default value --> will use the one from source
        vrt = gdal.AutoCreateWarpedVRT(self.ds, None, dest_wkt, interpolation,
                                       err_thresh)
        size = (vrt.RasterXSize, vrt.RasterYSize, self.RasterCount)
        dst_gt = vrt.GetGeoTransform()
        vrt = None
        imgio = MemFileIO()
        dest = self.driver.raster(imgio.name, size, dtype)
        imgio.close()
        dest.SetGeoTransform(dst_gt)
        dest.SetProjection(to_sref)
        for band in dest:
            band.SetNoDataValue(self.nodata)
            band = None
        # Uses self and dest projection when set to None
        gdal.ReprojectImage(self.ds, dest.ds, None, None, interpolation)
        return dest


open = Raster
