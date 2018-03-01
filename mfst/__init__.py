import openfst_wrapper_backend as _backend
from collections import namedtuple as _namedtuple

ArcType = _namedtuple(['ilabel', 'olabel', 'nextstate', 'weight'])

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

    def _reverse(self):
        """
        Return a weight that represent reversing this edge
        """
        return self

    def _sampling_weight(self):
        """
        Return a positive unnormalized floating point value that can be used to sample this arc
        """
        raise NotImplementedError()

    def _approx_eq(self, other, delta):
        """
        Returns if this weight is approximatly equal to another other less than delta
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

    def _sampling_weight(self):
        # just make the sampling of these path weights uniform
        return 1

    def _approx_eq(self, other, delta):
        return abs(self._value - other._value) < delta

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
        if isinstance(w, str) and w == '__FST_INVALID__':
            # this can be returned by the C++ binding in the case that there is an invalid state
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

    @property
    def start_state(self):
        """
        Return the state id of the starting state
        """
        return self._Start()

    @start_state.setter
    def start_state(self, state):
        """
        Mark a state as the start state
        """
        assert (state >= 0 and state < self.num_states), "Invalid state id"
        return self._SetStart(state)

    def set_start_state(self, state):
        self.start_state = state

    def add_state(self):
        """
        Add a new state to the FST
        Return this state's id
        """
        return self._AddState()

    def add_arc(self, from_state, to_state,
                weight=1, input_label=0, output_label=0):
        """
        Add an arc between states from->to with weight (default 1).
        input_label and output label should be ints that map to a label (0 == epsilon)
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

    def get_arcs(self, state):
        """
        Return the arcs coming out of some state
        """
        assert (state >= 0 and state < self.num_states), "Invalid state id"
        for arc in self._ArcList(state):
            yield ArcType(*arc)

    def isomorphic(self, other, delta=1.0/1024):
        """
        This operation determines if two transducers with a certain required
        determinism have the same states, irrespective of numbering, and the
        same transitions with the same labels and weights, irrespective of
        ordering. In other words, Isomorphic(A, B) is true if and only if the
        states of A can be renumbered and the transitions leaving each state
        reordered so that Equal(A, B) is true.

        http://www.openfst.org/twiki/bin/view/FST/IsomorphicDoc

        uses WeightBase._approx_eq to compare wieghts.
        delta: 32 bit floating point number that is passed to _approx_eq
        """
        return self._Isomorphic(other, delta)

    # methods for changing the fst given anther fst
    def concat(self, other):
        """
        This operation computes the concatenation (product) of two FSTs. If A
        transduces string x to y with weight a and B transduces string w to v
        with weight b, then their concatenation transduces string xw to yv with
        weight a (times) b.

        http://www.openfst.org/twiki/bin/view/FST/ConcatDoc
        """
        assert isinstance(other, FST)
        return self._Concat(other)

    def compose(self, other):
        """
        This operation computes the composition of two transducers. If A transduces
        string x to y with weight a and B transduces y to z with weight b, then their
        composition transduces string x to z with weight a (times) b.

        http://www.openfst.org/twiki/bin/view/FST/ComposeDoc
        """
        assert isinstance(other, FST)
        return self._Compose(other)

    def determinize(self, delta=1.0/1024, weight_threshold=None):
        """
        This operation determinizes a weighted transducer. The result will be an
        equivalent FST that has the property that no state has two transitions
        with the same input label. For this algorithm, epsilon transitions are
        treated as regular symbols (cf. RmEpsilon).

        http://www.openfst.org/twiki/bin/view/FST/DeterminizeDoc

        delta: Quantization delta for subset weights.
        weight_threshold: Pruning weight threshold.
        """
        if weight_threshold is None:
            weight_threshold = self._weight_class()

        return self._Determinize(delta, self._make_weight(weight_threshold))

    def project(self, type='input'):
        """
        This operation projects an FST onto its domain or range by either copying
        each arc's input label to its output label or vice versa.

        http://www.openfst.org/twiki/bin/view/FST/ProjectDoc
        """

        if type == 'output':
            t = 0
        elif type == 'intput':
            t = 1
        else:
            raise RuntimeError("unknown project type " + type)
        return self._Project(t)

    def difference(self, other):
        """
        This operation computes the difference between two FSAs. Only strings that
        are in the first automaton but not in second are retained in the
        result.

        http://www.openfst.org/twiki/bin/view/FST/DifferenceDoc
        """
        assert isinstance(other, FST)
        assert self._Difference(other)

    def invert(self):
        """
        This operation inverts the transduction corresponding to an FST by
        exchanging the FST's input and output labels.

        http://www.openfst.org/twiki/bin/view/FST/InvertDoc
        """
        return self._Invert()

    def prune(self, weight):
        """
        This operation deletes states and arcs in the input FST that do not belong
        to a successful path whose weight is no more (w.r.t the natural the
        natural semiring order) than the threshold t otimes the weight of the
        shortest path in the input FST.

        http://www.openfst.org/twiki/bin/view/FST/PruneDoc
        """
        return self._Prune(self._make_weight(weight))

    def random_path(self, arc_selector=None):
        """


        http://www.openfst.org/twiki/bin/view/FST/RandGenDoc
        """
        pass