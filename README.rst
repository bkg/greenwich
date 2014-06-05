Greenwich
=========

|Build Status|_
|Coverage Status|_

Adding Python conveniences to the wonderful world of `GDAL <http://www.gdal.org/>`_.

Greenwich provides a wrapper for the GDAL SWIG Python bindings. The focus here is on
providing some higher level behavior mainly to the raster side of the GDAL/OGR fence.

Basic Usage
-----------

Open any raster data set you have lying around, perhaps some climate data from
`WorldClim <http://worldclim.org/CMIP5>`_.

.. code-block:: python

    from greenwich.raster import Raster
    from greenwich.geometry import Geometry
    from greenwich.io import MemFileIO

    with Raster('cc85tn701.tif') as tmax:
        # Save as a NetCDF file.
        tmax.save('cc85tn701.nc')
        geom = Geometry(
            wkt='POLYGON((-123 47,-123 48,-122 49,-121 48,-121 47,-123 47))',
            srs=4326)
        # Clip the raster with a geometry and save the result as a GeoTIFF.
        with tmax.clip(geom) as clipped:
            clipped.save('clipped.tif')

        # Return a NumPy MaskedArray using no data values for a given bounding box.
        m = tmax.masked_array((-120, 38, -118, 44))

        # Convert to an Erdas Imagine file in memory.
        imgio = MemFileIO(suffix='.img')
        tmax.save(imgio)
        imgdata = imgio.read()
        imgio.close()

        # Iterate over bands and retrieve the maximum pixel values.
        maxvals = [band.GetMaximum() for band in tmax]

Retrieve a NumPy array for a specific area by providing the extent as a 4-tuple of min/max x, y coordinates::

    arr = tmax.array((-120, 38, -118, 44))

Reproject the raster to another coordinate system. You may pass EPSG codes, WKT,
proj4 formatted projections, or a SpatialReference instance as an argument.::

    warped = tmax.warp(3857)

Perhaps you would like to resample your image to a new resolution which can be
achieved with::

    resampled = tmax.resample((100, 100))

Raster instances still behave like a gdal.Dataset.::

    meta = tmax.GetMetadata()
