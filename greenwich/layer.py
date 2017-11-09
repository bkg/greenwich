from osgeo import ogr


class MemoryLayer(object):
    """Wrap an in-memory Layer.

    Holds reference to Memory DataSource to avoid segfault in underlying
    GDAL lib when going out of scope.
    """

    def __init__(self, srs=None, geom_type=ogr.wkbUnknown):
        self.id = 'id'
        self.ds = ogr.GetDriverByName('Memory').CreateDataSource('')
        self.layer = self.ds.CreateLayer('', srs, geom_type)
        idfield = ogr.FieldDefn(self.id, ogr.OFTInteger)
        self.layer.CreateField(idfield)

    def __getattr__(self, attr):
        return getattr(self.layer, attr)

    def __getitem__(self, index):
        return self.layer[index]

    def __iter__(self):
        for feature in self.layer:
            yield feature

    def __len__(self):
        return self.layer.GetFeatureCount()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return True

    def close(self):
        self.ds.Destroy()

    @classmethod
    def copy(cls, layer):
        obj = cls(layer.GetSpatialRef(), layer.GetGeomType())
        obj.load(layer)
        return obj

    @classmethod
    def from_records(cls, records):
        geom = records[0][1]
        obj = cls(geom.GetSpatialReference(), geom.GetGeometryType())
        featdef = obj.GetLayerDefn()
        for record in records:
            feature = ogr.Feature(featdef)
            feature.SetFID(record[0])
            feature.SetField(obj.id, record[0])
            feature.SetGeometry(record[1])
            obj.CreateFeature(feature)
            feature.Destroy()
        return obj

    def load(self, layer):
        defn = self.GetLayerDefn()
        for feat in layer:
            feature = ogr.Feature(defn)
            g = feat.geometry()
            feature.SetGeometry(g)
            feature.SetField(self.id, feat.GetFID())
            self.layer.CreateFeature(feature)
            feature.Destroy()

    def transform(self, sref):
        for feature in self:
            geom = feature.geometry()
            geom.TransformTo(sref)
            feature.SetGeometry(geom)
