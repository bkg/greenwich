import multiprocessing

import contones.gio
import contones.raster

def _run_encoder(path, encoder_cls, geom=None):
    encoder = encoder_cls()
    with contones.raster.Raster(path) as r:
        if geom:
            with r.crop(geom) as cropped:
                cropped.save(encoder)
        else:
            r.save(encoder)
    buff = encoder.getvalue()
    # Remove the dataset from memory
    encoder.unlink()
    return buff

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


class ImageIOPool(object):

    def __init__(self, drivername, pathnames, geom=None):
        self.geom = geom
        self.drivername = drivername
        self.pathnames = pathnames
        num_workers = multiprocessing.cpu_count()
        self.num_workers = num_workers / 2 if num_workers > 4 else num_workers
        self.ps = []
        self.outq = multiprocessing.Queue()
        self.sentinel = 'STOP'

    def run(self):
        paths = list(self.pathnames)
        # FIXME: Use a Queue of pathnames and read them off the stack.
        while paths:
            if len(self.ps) < self.num_workers:
                p = multiprocessing.Process(target=self.run_job, args=(paths.pop(),))
                self.ps.append(p)
                p.start()
        for p in self.ps:
            p.join()
        self.outq.put(self.sentinel)

    def run_job(self, path):
        encoder = contones.gio.ImageIO(driver=self.drivername)
        with contones.raster.Raster(path) as r:
            if self.geom:
                with r.crop(self.geom) as cropped:
                    cropped.save(encoder)
            else:
                r.save(encoder)
        buff = encoder.getvalue()
        # Remove the dataset from memory
        encoder.unlink()
        #return buff
        self.outq.put(buff)

    def get_results(self):
        self.run()
        return list(iter(self.outq.get, self.sentinel))
