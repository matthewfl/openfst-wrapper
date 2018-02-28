import openfst_wrapper_backend as _backend

class WeightBase(object):

    @classmethod
    def zero(cls):
        """
        Returns the zero element of the semiring
        The semiring should interpret int(0) as the zero when being constructed
        """
        return cls(0)

    @classmethod
    def one(cls):
        """
        Returns the one element of the semiring
        The semiring should interpret int(1) as the one when being constructed
        """
        return cls(1)

    def __add__(self, other):
        """
        Semiring add
        """
        raise NotImplementedError()

    def __mul__(self, other):
        """
        Semiring multiply
        """
        raise NotImplementedError()

    def __div__(self, other):
        """
        Division used in weight pushing
        """
        raise NotImplementedError()

    def __pow__(self, n):
        """
        Power of self for n is an int
        """
        raise NotImplementedError()

    def _member(self):
        """
        Checks if the current item is a member of the semiring
        (Eg float is not nan)
        """
        raise NotImplementedError()

    def _quantize(self):
        """
        Return a new instance that is bucketed
        """
        raise NotImplementedError()

    def __hash__(self):
        """
        Return the hash code for this instance
        (Will be used by openfst)
        """
        raise NotImplementedError()

    def __eq__(self, other):
        """
        Return if this object is equal to another
        (Will be used by openfst)
        """
        raise NotImplementedError()

    def __truediv__(self, other):
        return self.__div__(other)



class _WeightWrapper(WeightBase):

    def __init__(self, value=0):
        # This needs to happen otherwise the backing classes might not get set up properly
        super().__init__()
        self._value = value  # the value

    @classmethod
    def _create(cls):
        return cls()

    def zero(self):
        return self._create()

    def one(self):
        w = self._create()
        w._value = 1
        return w

    def __add__(self, other):
        assert type(other) is type(self)
        w = self._create()
        w._value = self._value + other._value
        return w

    def __mul__(self, other):
        assert type(other) is type(self)
        w = self._create()
        w._value = self._value * other._value
        return w

    def __div__(self, other):
        assert type(other) is type(self)
        w = self._create()
        w._value = self._value / other._value
        return w

    def __pow__(self, n):
        w = self._create()
        w._value = w._value ** n
        return w

    def _member(self):
        # check that this is a member of the semiring
        # this is just a nan check atm
        return self._value == self._value

    def _quantize(self, delta=.5):
        # quantize the weight into buckets
        return self

    def __str__(self):
        return str(self._value)

    def __hash__(self):
        return hash(self._value)

    def __eq__(self, other):
        return isinstance(other, _WeightWrapper) and self._value == other._value



class FST(_backend.FSTBase):
    """
    Wraps a mutable FST class
    """

    def __init__(self, weight_class=None):
        super().__init__()
        if not weight_class:
            self._weight_class = _WeightWrapper
        else:
            assert issubclass(weight_class, WeightBase)
            weight_class(0)  # check that we can construct zero and one
            weight_class(1)
            self._weight_class = weight_class

    def _make_weight(self, w):
        if isinstance(w, self._weight_class):
            return w
        assert not isinstance(w, WeightBase), "Can not mix different types of weights in a FST"
        if isinstance(w, int):
            if w == 0:
                return self._weight_class.zero()
            elif w == 1:
                return self._weight_class.one()
        if isinstance(w, str) and w == 'FST INVALID':
            return None
        return self._weight_class(w)

    @property
    def num_states(self):
        """
        Return the number of states currently set on the fst
        """
        return self._NumStates()

    @property
    def num_arcs(self):
        """
        Return the number of arcs in the fst
        """
        return self._NumArcs()

    def add_state(self):
        """
        Add a new state to the FST
        Return this state's id
        """
        return self._AddState()

    def add_arc(self, from_state, to_state,
                weight=1, input_label=-1, output_label=-1):
        """
        Add an arc between states from->to with weight (default 1).
        input_label and output label should be ints that map to a label (-1 == epsilon)
        """
        # assert if the state is valid otherwise openfst calls exit(1)
        assert (from_state >= 0 and from_state < self.num_states and
                to_state >= 0 and to_state < self.num_states), "Invalid state id"
        return self._AddArc(from_state, to_state, input_label, output_label,
                            self._make_weight(weight))

    def delete_arcs(self, state):
        """
        Delete all arcs coming out of state
        """
        assert (state >= 0 and state < self.num_states), "Invalid state id"
        return self._DeleteArcs(state)

    def delete_states(self):
        """
        Delete all states in the FST
        """
        return self._DeleteStates()

    def set_final_weight(self, state, weight):
        """
        Set the weight that this state transisions to the final state
        """
        assert (state >= 0 and state < self.num_states), "Invalid state id"
        return self._SetFinal(state, self._make_weight(weight))

    def get_final_weight(self, state):
        """
        Get the weight of transistioning to the final state
        """
        assert (state >= 0 and state < self.num_states), "Invalid state id"
        return self._make_weight(self._FinalWeight(state))

    def set_start_state(self, state):
        """
        Mark a state as the start state
        """
        assert (state >= 0 and state < self.num_states), "Invalid state id"
        return self._SetStart(state)
