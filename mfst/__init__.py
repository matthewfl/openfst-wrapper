import openfst_wrapper_backend as _backend


class FST(_backend.FSTBase):

    pass



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

    def add(self, other):
        assert type(other) is type(self)
        w = self.create()
        w._value = self._value + other._value
        return w

    def multiply(self, other):
        assert type(other) is type(self)
        w = self.create()
        w._value = self._value * other._value
        return w

    def divide(self, other):
        assert type(other) is type(self)
        w = self.create()
        w._value = self._value / other._value
        return w

    def power(self, n):
        w = self.create()
        w._value = w._value ** n
        return w

    def equal(self, other):
        return self.__eq__(other)

    def member(self):
        # check that this is a member of the semiring
        # this is just a nan check atm
        return self._value == self._value

    def quantize(self, delta=.5):
        # quantize the weight into buckets
        return self

    # These methods are provided to make the python interface work but are not directly used

    def __str__(self):
        return str(self._value)

    def __add__(self, other):
        return self.add(other)

    def __mul__(self, other):
        return self.multiply(other)

    def __div__(self, other):
        return self.divide(other)

    def __pow__(self, other):
        return self.power(other)
