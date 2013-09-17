import os
import itertools

import numpy as np
from PIL import Image, ImageDraw
from osgeo import gdal, gdal_array, gdalconst, ogr, osr
import contones.io

def pixel_to_xy(coords, geotransform):
    """Convert image pixel/line coordinates to georeferenced x/y, return a
    generator of two-tuples.

    Arguments:
    coords -- input coordinates as iterable containing two-tuples/lists such as
    ((-120, 38), (-121, 39))
    geotransform -- GDAL GeoTransformation tuple
    """
    for x, y in coords:
        geo_x = geotransform[0] + geotransform[1] * x + geotransform[2] * y
        geo_y = geotransform[3] + geotransform[4] * x + geotransform[5] * y
        # Move the coordinate to the center of the pixel.
        geo_x += geotransform[1] / 2.0
        geo_y += geotransform[5] / 2.0
        yield geo_x, geo_y

def xy_to_pixel(coords, gt):
    """Transform from projection coordinates (Xp,Yp) space to pixel/line
    (P,L) raster space, based on the provided geotransformation.

    Arguments:
    coords -- input coordinates as iterable containing two-tuples/lists such as
    ((-120, 38), (-121, 39))
    gt -- GDAL GeoTransformation tuple
    """
    return [(int((x - gt[0]) / gt[1]), int((y - gt[3]) / gt[5]))
            for x, y in coords]

def geom_to_array(geom, matrix_size, geotrans):
    """Converts an OGR polygon to a 2D NumPy array.

    Arguments:
    geom -- OGR Polygon or MultiPolygon
    matrix_size -- array size in pixels as a tuple of (width, height)
    geotrans -- geotransformation as a five element tuple like
        (-124.625, 0.125, 0.0, 44.0, 0.0, -0.125,).
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
        draw.polygon(xy_to_pixel(coords, geotrans), 0)
    return np.asarray(img)

#def from_bbox(bbox):
def envelope_asgeom(bbox):
    #env[0], env[2], env[1], env[3]
    print 'BBOX:', bbox
    idxs = ((0, 2), (1, 2), (1, 3), (0, 3), (0, 2))
    #ring = ogr.Geometry(ogr.wkbLineString)
    ring = ogr.Geometry(ogr.wkbLinearRing)
    for idx in idxs:
        #print bbox[idx[0]], bbox[idx[1]]
        ring.AddPoint(bbox[idx[0]], bbox[idx[1]])
    polyg = ogr.Geometry(ogr.wkbPolygon)
    #polyg.AddGeometry(ring)
    polyg.AddGeometryDirectly(ring)
    print 'ENVELOPE:', polyg.GetEnvelope()
    return polyg

def count_unique(arr):
    """Returns a two-tuple of pixel count and bin value for every unique pixel
    value in the array.

    Arguments:
    arr -- numpy ndarray
    """
    return [a.tolist() for a in np.histogram(arr, np.unique(arr))]


class GeoTransform(object):
#class AffineTransform(object):

    def __init__(self, geotrans_tuple):
        self.origin = geotrans_tuple[0], geotrans_tuple[3]
        self.pixel_origin = (0, 0)
        self.pixel_dest = ()
        #self.pixel_x_size =
        #self.pixel_y_size =
        self.pixel_size = geotrans_tuple[1], geotrans_tuple[5]
        self.dims = ()
        #self.window = ()

    def resize(self, extent):
        #corners = (extent[0], extent[3]), (extent[2], extent[1])
        corners = (extent[0], extent[3]), (extent[1], extent[2])
        gt = list(self.as_tuple())
        #gt = self.as_tuple()
        ul, lr = xy_to_pixel(corners, gt)
        #self.px = ul, lr
        self.pixel_origin = ul
        self.pixel_dest = lr
        #new_gt = list(gt)
        # Set upper left x, y for new GeoTransformation.
        gt[0], gt[3] = corners[0]
        self.origin = gt[0], gt[3]
        # Find the pixel dimensions for the image.
        #nx, ny = lr[0] - ul[0], lr[1] - ul[1]
        self.dims = lr[0] - ul[0], lr[1] - ul[1]
        #return {'geotrans': new_gt, 'dims': (nx, ny), 'ul': ul, 'lr': lr}
        # Need ul_px,dims, geotrans

    #def lower_right(self):

    def reader_args(self):
        return self.pixel_origin + self.dims

    def as_tuple(self):
        # Assumes north up images.
        return (self.origin[0], self.pixel_size[0], 0.0, self.origin[1], 0.0,
                self.pixel_size[1])


class Raster(object):
    """Wrap a GDAL Dataset with additional behavior."""

    def __init__(self, dataset, mode=gdalconst.GA_ReadOnly):
        if not isinstance(dataset, gdal.Dataset):
            dataset = gdal.Open(dataset, mode)
        if dataset is None:
            raise IOError('Could not open %s' % dataset)
        self.ds = dataset
        self.sref = osr.SpatialReference(dataset.GetProjection())
        try:
            self.srid = int(self.sref.GetAuthorityCode('PROJCS'))
        except TypeError:
            self.srid = None
        self._nodata = None
        self._extent = None
        self._io = None
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
        envelope = envelope_asgeom(bbox.GetEnvelope())
        # Without a simple bounding box, this is really a masking operation
        # rather than a simple crop.
        if not bbox.Equals(envelope):
            pixbuf, pixwin = self.mask(bbox)
        else:
            bbox = self._transform_maskgeom(bbox)
            pixwin = self._pixelwin_from_extent(bbox.GetEnvelope())
            pixbuf = self.ReadRaster(*pixwin['ul_px'] + pixwin['dims'])
        clone = self.new(pixwin, pixbuf)
        #clone.WriteRaster(0, 0, pixwin['dims'][0], pixwin['dims'][1], pixbuf)
        return clone

    def new(self, pixwin=None, pixeldata=None):
        """Derive new Raster instances.

        Keyword args:
        pixwin -- dict of georef params
        pixeldata -- bytestring containing pixel data
        """
        pixels_x, pixels_y = (pixwin and pixwin['dims'] or
            (self.RasterXSize, self.RasterYSize))
        band = self.GetRasterBand(1)
        rcopy = self.io.create(
            pixels_x, pixels_y, self.RasterCount, band.DataType)
        rcopy.SetProjection(self.GetProjection())
        rcopy.SetGeoTransform(
            pixwin and pixwin['geotrans'] or self.GetGeoTransform())
        band_copy = rcopy.GetRasterBand(1)
        band_copy.SetNoDataValue(self.nodata)
        colors = band.GetColorTable()
        if colors:
            band_copy.SetColorTable(colors)
        # Flush written data
        #band_copy = None
        if pixeldata:
            args = (0, 0) + pixwin['dims'] + (pixeldata,)
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

    def _mask(self, geom):
        geom = self._transform_maskgeom(geom)
        pixwin = self._pixelwin_from_extent(geom.GetEnvelope())
        arr = self.ReadAsArray(*pixwin['ul_px'] + pixwin['dims'])
        if arr is None:
            raise IOError('Could not read {}'.format(self.GetDescription()))
        mask_arr = geom_to_array(geom, pixwin['dims'], pixwin['geotrans'])
        m = np.ma.masked_array(arr, mask=mask_arr)
        #m.set_fill_value(self.nodata)
        m = np.ma.masked_values(m, self.nodata)
        return m, pixwin

    def mask(self, geom):
        """Returns a pixel buffer as a str, and a dict including the new
        geotransformation and pixel dimensions.

        Arguments:
        geom -- OGR Polygon or MultiPolygon
        """
        m, pixwin = self._mask(geom)
        #TODO: return a new instance here.
        #return str(np.getbuffer(m.filled())), pixwin
        pixbuf = str(np.getbuffer(m.filled()))
        clone = self.new(pixwin, pixbuf)
        #args = (0, 0) + pixwin['dims'] + (pixbuf,)
        #clone.WriteRaster(0, 0, pixwin['dims'][0], pixwin['dims'][1], pixbuf)
        #clone.WriteRaster(*args)
        return clone

    def mask_asarray(self, geom):
        """Returns a numpy MaskedArray for the intersecting geometry.

        Arguments:
        geom -- OGR Polygon or MultiPolygon
        """
        return self._mask(geom)[0]

    @property
    def nodata(self):
        """Returns read only property for band nodata value, assuming single
        band rasters for now.
        """
        if self._nodata is None:
            self._nodata = self[1].GetNoDataValue()
        return self._nodata

    def __pixelwin_from_extent(self, extent):
        """Returns a dict containing a geotransformation tuple with its origin set
        to the upper left coord from 'extent', pixel dimensions as a tuple, and
        upper left and lower right coordinates in pixel space.

        This pixel window is useful for calling ReadRaster and ReadAsArray.

        Arguments:
        extent -- 4-tuple, consisting of (xmin, ymin, xmax, ymax) as returned
            by the GEOS/OGR Polygon or MultiPolygon 'extent' instance attribute
        """
        geotransform = self.GetGeoTransform()
        #corners = (extent[0], extent[3]), (extent[2], extent[1])
        corners = (extent[0], extent[3]), (extent[1], extent[2])
        ul, lr = xy_to_pixel(corners, geotransform)
        # Origin cannot be outside of raster dimensions.
        if not (0, 0) <= ul < (self.RasterXSize, self.RasterYSize):
            raise ValueError('Origin pixel out of bounds: {}'.format(ul))
        # FIXME: Should not be negative
        # Find the pixel dimensions for the image based on the corners and
        # reduce them if they are beyond the maximum image dimensions.
        nx = min(lr[0] - ul[0], self.RasterXSize - ul[0])
        ny = min(lr[1] - ul[1], self.RasterYSize - ul[1])
        new_gt = list(geotransform)
        # Set upper left x, y origin for new GeoTransformation.
        new_gt[0], new_gt[3] = corners[0]
        return {'geotrans': new_gt, 'dims': (nx, ny), 'ul_px': ul, 'lr_px': lr}

    def _pixelwin_from_extent(self, extent):
        geotrans = GeoTransform(self.GetGeoTransform())
        print 'TRANSFORM', geotrans
        #corners = (extent[0], extent[3]), (extent[1], extent[2])
        geotrans.resize(extent)
        ul, lr = geotrans.pixel_origin,geotrans.pixel_dest
        # Find the pixel dimensions for the image based on the corners and
        # reduce them if they are beyond the maximum image dimensions.
        nx = min(lr[0] - ul[0], self.RasterXSize - ul[0])
        ny = min(lr[1] - ul[1], self.RasterYSize - ul[1])
        #nx = min(geotrans.pixel_dest[0] - geotrans.pixel_origin[0], self.RasterXSize - geotrans.pixel_origin[0])
        #ny = min(geotrans.pixel_dest[1] - geotrans.pixel_origin[1], self.RasterYSize - geotrans.pixel_origin[1])
        return {'geotrans': geotrans.as_tuple(), 'dims': (nx, ny), 'ul_px': geotrans.pixel_origin}

    def ReadRaster(self, *args, **kwargs):
        """Returns a string of raster data for partial or full extent.

        Overrides GDALDataset.ReadRaster() with the full raster dimensions by
        default.
        """
        if len(args) < 4:
            args = (0, 0, self.RasterXSize, self.RasterYSize)
        return self.ds.ReadRaster(*args, **kwargs)

    def resample_to(self, to, dest=None,
                    interpolation=gdalconst.GRA_NearestNeighbour):
        dtype = self[1].DataType
        dest = dest or gdal.GetDriverByName('MEM').Create(
            '', to.RasterXSize, to.RasterYSize, to.RasterCount, dtype)
        dest.SetGeoTransform(to.GetGeoTransform())
        dest.SetProjection(to.GetProjection())
        band = dest.GetRasterBand(1)
        band.SetNoDataValue(self.nodata)
        band = None
        # Uses self and dest projection when set to None
        gdal.ReprojectImage(self, dest, None, None, eResampleAlg=interpolation)
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

