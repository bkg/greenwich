class Comparable(object):
    """A generic mixin providing type and attribute equality comparisons."""

    def __eq__(self, another):
        if type(another) is type(self):
            return self.__dict__ == another.__dict__
        return False

    def __ne__(self, another):
        return not self.__eq__(another)
