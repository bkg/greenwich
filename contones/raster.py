import os

import numpy as np
from PIL import Image, ImageDraw
from osgeo import gdal, gdalconst, ogr, osr
import contones.io

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


class Envelope(object):

    def __init__(self, min_x, min_y, max_x, max_y):
        self.min_x = min_x
        self.min_y = min_y
        self.max_x = max_x
        self.max_y = max_y
        if not self.min_x < self.max_x and self.min_y < self.max_y:
            raise Exception('Invalid coordinate extent')

    def __repr__(self):
        return str(self.tuple)

    @property
    def ur(self):
        return self.max_x, self.max_y

    #def lower_right(self):

    @property
    def lr(self):
        return self.max_x, self.min_y

    @property
    def ll(self):
        return self.min_x, self.min_y

    @property
    def ul(self):
        return self.min_x, self.max_y

    @property
    def tuple(self):
        return self.ll + self.ur

    @property
    def height(self):
        return self.max_y - self.min_y

    @property
    def width(self):
        return self.max_x - self.min_x

    def scale(self, factor_x, factor_y=None):
        """Rescale the envelope by the given factor(s)."""
        factor_y = factor_x if factor_y is None else factor_y
        w = self.width * factor_x / 2.0
        h = self.height * factor_y / 2.0
        self.min_x += w
        self.max_x -= w
        self.min_y += h
        self.max_y -= h

    def to_geom(self):
        """Returns an OGR Geometry for this envelope."""
        ring = ogr.Geometry(ogr.wkbLinearRing)
        #coords = (self.ll, self.lr, self.ur, self.ul, self.ll)
        for coord in self.ll, self.lr, self.ur, self.ul, self.ll:
            ring.AddPoint(*coord)
        polyg = ogr.Geometry(ogr.wkbPolygon)
        polyg.AddGeometryDirectly(ring)
        return polyg

    @staticmethod
    def from_geom(geom):
        """Returns an Envelope from an OGR Geometry."""
        extent = geom.GetEnvelope()
        # ul, lr
        #corners = (extent[0], extent[3], extent[1], extent[2])
        #corners = (extent[0], extent[2], extent[1], extent[3])
        #return Envelope(*corners)
        return Envelope(extent[0], extent[2], extent[1], extent[3])


class AffineTransform(object):

    def __init__(self, geotrans_tuple):
        """
        Arguments:
        geotrans_tuple -- geotransformation as a five element tuple like
            (-124.625, 0.125, 0.0, 44.0, 0.0, -0.125,).
        """
        # Origin coordinate in projected space.
        self.origin = geotrans_tuple[0], geotrans_tuple[3]
        self.pixel_size = geotrans_tuple[1], geotrans_tuple[5]

    def __repr__(self):
        return str(self.tuple)

    #def __getitem__(self, idx):
        #return self.tuple[idx]

    def pixel_to_xy(self, coords):
        """Convert image pixel/line coordinates to georeferenced x/y, return a
        generator of two-tuples.

        Arguments:
        coords -- input coordinates as iterable containing two-tuples/lists such as
        ((-120, 38), (-121, 39))
        geotransform -- GDAL GeoTransformation tuple
        """
        geotransform = self.tuple
        for x, y in coords:
            geo_x = geotransform[0] + geotransform[1] * x + geotransform[2] * y
            geo_y = geotransform[3] + geotransform[4] * x + geotransform[5] * y
            # Move the coordinate to the center of the pixel.
            geo_x += geotransform[1] / 2.0
            geo_y += geotransform[5] / 2.0
            yield geo_x, geo_y

    # TODO: work with single coords as well.
    #def xy_to_pixel(self, coords):
    def transform(self, coords):
        """Transform from projection coordinates (Xp,Yp) space to pixel/line
        (P,L) raster space, based on the provided geotransformation.

        Arguments:
        coords -- input coordinates as iterable containing two-tuples/lists such as
        ((-120, 38), (-121, 39))
        gt -- GDAL GeoTransformation tuple
        """
        #return [(int((x - gt[0]) / gt[1]), int((y - gt[3]) / gt[5]))
        return [(int((x - self.origin[0]) / self.pixel_size[0]),
                 int((y - self.origin[1]) / self.pixel_size[1]))
                for x, y in coords]

    @property
    def tuple(self):
        # Assumes north up images.
        return (self.origin[0], self.pixel_size[0], 0.0, self.origin[1], 0.0,
                self.pixel_size[1])


class SpatialReference(object):
    def __init__(self, sref):
        if isinstance(sref, int):
            sr = osr.SpatialReference()
            sr.ImportFromEPSG(sref)
            #self._sref = sr
        elif isinstance(sref, str):
            if '+proj=' in sref:
                sr = osr.SpatialReference()
                sr.ImportFromProj4(sref)
                # 0 or 1
            else:
                sr = osr.SpatialReference(sref)
                #self._sref = osr.SpatialReference(sref)
            # Add EPSG authority if applicable
            sr.AutoIdentifyEPSG()
        else:
            raise TypeError('Cannot create SpatialReference from {}'.format(str(to_sref)))
        self._sref = sr
        #self._srid = None

    #*** AttributeError: AttributeError("'SpatialReference' object has no attribute 'ExportToWkt'",)
    def __getattr__(self, attr):
        return getattr(self._sref, attr)

    @property
    def srid(self):
        epsg_id = (self._sref.GetAuthorityCode('GEOGCS') or
                   self._sref.GetAuthorityCode('GEOGCS'))
        try:
            return int(epsg_id)
        except TypeError:
            return


class Raster(object):
    """Wrap a GDAL Dataset with additional behavior."""

    def __init__(self, dataset, mode=gdalconst.GA_ReadOnly):
        if not isinstance(dataset, gdal.Dataset):
            dataset = gdal.Open(dataset, mode)
        if dataset is None:
            raise IOError('Could not open %s' % dataset)
        self.ds = dataset
        self.sref = osr.SpatialReference(dataset.GetProjection())
        self._nodata = None
        self._extent = None
        self._io = None
        self.affine = AffineTransform(self.GetGeoTransform())
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

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return True

    def __len__(self):
        return self.RasterCount

    def close(self):
        """Close the GDAL dataset."""
        # De-ref the GDAL Dataset to completely close it.
        self.ds = None

    def crop(self, bbox):
        """Returns a pixel buffer as str, pixel dimensions, and
        geotransformation tuple.

        Arguments:
        bbox -- bounding box as an OGR Polygon
        """
        return self._mask(bbox)

    #FIXME: Check geom envelope bounds intersects.
    def _mask(self, geom):
        if isinstance(geom, Envelope):
            geom = geom.to_geom()
        geom = self._transform_maskgeom(geom)
        env = Envelope.from_geom(geom)
        bbox = env.to_geom()
        ul_px, lr_px = self.affine.transform((env.ul, env.lr))
        nx = min(lr_px[0] - ul_px[0], self.RasterXSize - ul_px[0])
        ny = min(lr_px[1] - ul_px[1], self.RasterYSize - ul_px[1])
        dims = (nx, ny)
        affine = AffineTransform(self.GetGeoTransform())
        # Update origin coordinate for the new affine transformation.
        affine.origin = env.ul
        # Without a simple bounding box, this is really a masking operation
        # rather than a simple crop.
        if not geom.Equals(bbox):
            arr = self.ReadAsArray(*ul_px + dims)
            mask_arr = geom_to_array(geom, dims, affine)
            m = np.ma.masked_array(arr, mask=mask_arr)
            #m.set_fill_value(self.nodata)
            m = np.ma.masked_values(m, self.nodata)
            pixbuf = str(np.getbuffer(m.filled()))
        else:
            pixbuf = self.ReadRaster(*ul_px + dims)
        clone = self.new(pixbuf, dims, affine.tuple)
        return clone

    def new(self, pixeldata=None, dimensions=None, affine=None):
        """Derive new Raster instances.

        Keyword args:
        pixeldata -- bytestring containing pixel data
        dimensions -- tuple of image size
        affine -- affine transformation tuple
        """
        pixels_x, pixels_y = dimensions or (self.RasterXSize, self.RasterYSize)
        band = self.GetRasterBand(1)
        rcopy = self.io.create(
            pixels_x, pixels_y, self.RasterCount, band.DataType)
        rcopy.SetProjection(self.GetProjection())
        rcopy.SetGeoTransform(affine or self.GetGeoTransform())
        colors = band.GetColorTable()
        for outband in rcopy:
            outband.SetNoDataValue(self.nodata)
            if colors:
                band_copy.SetColorTable(colors)
        if pixeldata:
            args = (0, 0) + dimensions + (pixeldata,)
            rcopy.WriteRaster(*args)
        return rcopy

    # TODO: Decide on envelope tuple format, or maybe just distinguish between
    # .envelope and .extent, create Envelope class.
    @property
    def extent(self):
        """Returns the minimum bounding rectangle as a tuple of min X, min Y,
        max X, max Y.
        """
        if self._extent is None:
            gt = self.GetGeoTransform()
            origin = gt[0], gt[3]
            ur_x = origin[0] + (self.RasterXSize * gt[1])
            ll_y = origin[1] + (self.RasterYSize * gt[5])
            self._extent = (origin[0], ll_y, ur_x, origin[1])
        return self._extent

    def mask(self, geom):
        """Returns a pixel buffer as a str, and a dict including the new
        geotransformation and pixel dimensions.

        Arguments:
        geom -- OGR Polygon or MultiPolygon
        """
        return self._mask(geom)

    def masked_array(self):
        return np.ma.masked_values(self.ReadAsArray(), self.nodata)

    def mask_asarray(self, geom):
        """Returns a numpy MaskedArray for the intersecting geometry.

        Arguments:
        geom -- OGR Polygon or MultiPolygon
        """
        #return self._mask(geom)[0]
        with self.mask(geom) as rast:
            m = rast.masked_array()
        return m

    @property
    def nodata(self):
        """Returns read only property for band nodata value, assuming single
        band rasters for now.
        """
        if self._nodata is None:
            self._nodata = self[1].GetNoDataValue()
        return self._nodata

    #def read(self, size=256):
        #"""Returns a list of pixel values"""
        #import struct

    def ReadRaster(self, *args, **kwargs):
        """Returns a string of raster data for partial or full extent.

        Overrides GDALDataset.ReadRaster() with the full raster dimensions by
        default.
        """
        if len(args) < 4:
            args = (0, 0, self.RasterXSize, self.RasterYSize)
        return self.ds.ReadRaster(*args, **kwargs)

    def resample(self, dimensions,
                 interpolation=gdalconst.GRA_NearestNeighbour):
        """Returns a new instance resampled to provided dimensions.

        Arguments:
        dimensions -- tuple of x,y image dimensions
        """
        # Find the scaling factor for pixel size.
        factors = (dimensions[0] / float(self.RasterXSize),
                   dimensions[1] / float(self.RasterYSize))
        affine = AffineTransform(self.GetGeoTransform())
        affine.pixel_size = (affine.pixel_size[0] * factors[0],
                             affine.pixel_size[1] * factors[1])
        # FIXME: affine, not affine.tuple
        dest = self.new(dimensions=dimensions, affine=affine.tuple)
        # Uses self and dest projection when set to None
        gdal.ReprojectImage(self.ds, dest.ds, None, None, interpolation)
        return dest

    def warp(self, to_sref, interpolation=gdalconst.GRA_NearestNeighbour):
        """Returns a new reprojected instance.

        Arguments:
        to_sref -- spatial reference as a proj4 or wkt string, or a
        SpatialReference
        """
        if not isinstance(to_sref, SpatialReference):
            to_sref = SpatialReference(to_sref)
        dest_wkt = to_sref.ExportToWkt()
        dtype = self[1].DataType
        err_thresh = 0.125
        # Call AutoCreateWarpedVRT() to fetch default values for target raster
        # dimensions and geotransform
        # src_wkt : left to default value --> will use the one from source
        vrt = gdal.AutoCreateWarpedVRT(self.ds, None, dest_wkt, interpolation,
                                       err_thresh)
        dst_xsize = vrt.RasterXSize
        dst_ysize = vrt.RasterYSize
        dst_gt = vrt.GetGeoTransform()
        vrt = None
        # FIXME: Should not set proj in new()?
        #dest = self.new(dimensions=(dst_xsize, dst_ysize))
        dest = self.io.create(dst_xsize, dst_ysize, self.RasterCount, dtype)
        print 'SELF', self.shape
        print 'DEST', dest.shape
        dest.SetGeoTransform(dst_gt)
        dest.SetProjection(to_sref)
        for band in dest:
            band.SetNoDataValue(self.nodata)
            band = None
        # Uses self and dest projection when set to None
        gdal.ReprojectImage(self.ds, dest.ds, None, None, interpolation)
        return dest

    def save(self, location):
        """Save this instance to the path and format from location.

        Arguments:
        location -- str or instance of a BaseImageIO subclass
        """
        try:
            r = location.copy_from(self)
        except AttributeError:
            path = getattr(location, 'name', location)
            ImageIO = contones.io.get_imageio_for(path)
            imgio = ImageIO(path)
            r = imgio.copy_from(self)
        finally:
            r.close()

    def SetProjection(self, to_sref):
        if not isinstance(to_sref, SpatialReference):
            to_sref = SpatialReference(to_sref)
        self.sref = to_sref
        self.ds.SetProjection(to_sref.ExportToWkt())

    def SetGeoTransform(self, geotrans_tuple):
        """Sets the affine transformation."""
        self.affine = AffineTransform(geotrans_tuple)
        self.ds.SetGeoTransform(geotrans_tuple)

    @property
    def shape(self):
        """Returns a tuple containing Y-axis, X-axis pixel counts."""
        return (self.RasterYSize, self.RasterXSize)

    @property
    def io(self):
        if self._io is None:
            d = self.ds.GetDriver()
            ImageIO = contones.io.get_imageio_for(d.ShortName)
            self._io = ImageIO()
        return self._io

    def _transform_maskgeom(self, geom):
        geom_sref = geom.GetSpatialReference()
        if geom_sref is None:
            raise Exception('Cannot transform from unknown spatial reference')
        # Reproject geom if necessary
        if not geom_sref.IsSame(self.sref):
            geom = geom.Clone()
            geom.TransformTo(self.sref)
        return geom

open = Raster
#@contextmanager
#def open(fpath):
    ##with Raster(fpath) as fpath:
        ##pass
    #return Raster(fpath)

