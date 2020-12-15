from . import AbstractSemiringWeight
from math import exp as _exp

try:
    # use numpy to define the real values in the case that we can import it
    from numpy import isreal as _isreal, isscalar as _isscalar

    def _is_real(x):
        return _isreal(x) and _isscalar(x)
except ImportError:
    def _is_real(x):
        return isinstance(x, (int, float))


class PythonValueSemiringWeight(AbstractSemiringWeight):
    """
    Plus times semiring weight over python objects
    """

    def __init__(self, value=0):
        super().__init__()
        # the value should be immutable, so access via property
        self.__value = value

    @property
    def value(self):
        return self.__value

    def _create(self, v):
        r = type(self)(v)
        return r

    def __add__(self, other):
        assert type(other) is type(self)
        return self._create(self.value + other.value)

    def __mul__(self, other):
        assert type(other) is type(self)
        return self._create(self.value * other.value)

    def __div__(self, other):
        assert type(other) is type(self)
        try:
            return self._create(self.value / other.value)
        except ZeroDivisionError:
            return self._create(float('nan'))

    def __pow__(self, n):
        return self._create(self.value ** n)

    def member(self):
        # check that this is a member of the semiring
        # this is just a nan check atm
        return self.value == self.value

    def quantize(self, delta=.5):
        # quantize the weight into buckets
        return self

    def sampling_weight(self):
        # just make the sampling of these path weights uniform
        return 1

    def approx_eq(self, other, delta):
        return abs(self.value - other.value) < delta

    def __str__(self):
        return str(self.value)

    def __hash__(self):
        return hash(self.value)

    def __eq__(self, other):
        return type(self) is type(other) and self.value == other.value

    def __repr__(self):
        return f'{type(self).__name__}({self.value})'

# static semiring zero and one elements
PythonValueSemiringWeight.zero = PythonValueSemiringWeight(0)
PythonValueSemiringWeight.one = PythonValueSemiringWeight(1)


class RealSemiringWeight(PythonValueSemiringWeight):
    """
    The standard <+,*> semiring on real valued variables
    """

    def __init__(self, v):
        assert _is_real(v), f"Value {v} is not a real number"
        super().__init__(v)

    def quantize(self, delta=1.0 / 1024):
        return self._create(int(self.value / delta) * delta)

    def sampling_weight(self):
        assert self.value >= 0, "Value on semiring when sampling a path must be >= 0"
        return self.value

    def __float__(self):
        return float(self.value)

    def __int__(self):
        return int(self.value)

# static semiring zero and one elements
RealSemiringWeight.zero = RealSemiringWeight(0)
RealSemiringWeight.one = RealSemiringWeight(1)


class MinPlusSemiringWeight(RealSemiringWeight):

    # set that this semiring has the path property (idempotent)
    # a + a = a \forall a && a \le b \iff a + b = a
    semiring_properties = 'path'

    # let self.value be the path length of this edge

    def __add__(self, other):
        # Return the min value between these two weights
        if self.value < other.value:
            return self
        else:
            return other

    def __mul__(self, other):
        # self (*) other
        return self._create(self.value + other.value)

    def __div__(self, other):
        # self (/) other
        return self._create(self.value - other.value)

# static semiring zero and one elements
MinPlusSemiringWeight.zero = MinPlusSemiringWeight(float('inf'))
MinPlusSemiringWeight.one = MinPlusSemiringWeight(0)


class MaxPlusSemiringWeight(RealSemiringWeight):

    # set that this semiring has the path property (idempotent)
    # a + a = a \forall a && a \le b \iff a + b = a
    semiring_properties = 'path'

    # let self.value be the path length of this edge

    def __add__(self, other):
        # Return the max value between these two weights
        if self.value > other.value:
            return self
        else:
            return other

    def __mul__(self, other):
        # self (*) other
        return self._create(self.value + other.value)

    def __div__(self, other):
        # self (/) other
        return self._create(self.value - other.value)

# static semiring zero and one elements
MaxPlusSemiringWeight.zero = MaxPlusSemiringWeight(float('-inf'))
MaxPlusSemiringWeight.one = MaxPlusSemiringWeight(0)


class TropicalSemiringWeight(MinPlusSemiringWeight):

    def sampling_weight(self):
        return _exp(-self.value)

# static semiring zero and one elements
TropicalSemiringWeight.zero = TropicalSemiringWeight(float('inf'))
TropicalSemiringWeight.one = TropicalSemiringWeight(0)


class BooleanSemiringWeight(PythonValueSemiringWeight):
    """
    Boolean Semiring with two elements True and False
    """

    semiring_properties = 'path'

    def __init__(self, v):
        assert isinstance(v, bool)
        super().__init__(v)

    def sampling_weight(self):
        if self.value:
            return 1
        return 0

    def approx_eq(self, other, delta):
        return self.value == other.value

    def __add__(self, other):
        # logical or
        if self.value:
            return self
        return other

    def __mul__(self, other):
        # logical and
        if not self.value:
            return self
        return other

    def __div__(self, other):
        if not other.value:
            raise ZeroDivisionError()
        return self

    def __pow__(self, n):
        return self

BooleanSemiringWeight.zero = BooleanSemiringWeight(False)
BooleanSemiringWeight.one = BooleanSemiringWeight(True)
