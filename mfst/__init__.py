import openfst_wrapper_backend as _backend


class FST(_backend.FSTBase):

    def __init__(self):
        super().__init__()
        self._weight_class = None


class Weight(object):

    def __init__(self):
        # This needs to happen otherwise the backing classes might not get set up properly
        super().__init__()
        self._value = 0  # the value

    def create(self):
        return type(self)()  # create a new instance of this class

    # These methods should be overriden to define a custom semiring
    def zero(self):
        return self.create()

    def one(self):
        w = self.create()
        w._value = 1
        return w

    def __add__(self, other):
        assert type(other) is type(self)
        w = self.create()
        w._value = self._value + other._value
        return w

    def __mul__(self, other):
        assert type(other) is type(self)
        w = self.create()
        w._value = self._value * other._value
        return w

    def __div__(self, other):
        assert type(other) is type(self)
        w = self.create()
        w._value = self._value / other._value
        return w

    def __pow__(self, n):
        w = self.create()
        w._value = w._value ** n
        return w

    # def (self, other):
    #     return self.__eq__(other)

    def _member(self):
        # check that this is a member of the semiring
        # this is just a nan check atm
        return self._value == self._value

    def _quantize(self, delta=.5):
        # quantize the weight into buckets
        return self

    def __truediv__(self, other):
        return self.__div__(other)
