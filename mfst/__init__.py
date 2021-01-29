# Written by Matthew Francis-Landau (2018)
#
# Wrapper for OpenFst that supports defining custom semirings in python
# and drawing FSTs in ipython notebooks
#
# Edit 2021 Eric Fosler-Lussier to create SymbolTable class, allow for different input/output symbol tables,
# allow compilation

import openfst_wrapper_backend as _backend
from collections import namedtuple as _namedtuple, deque as _deque
from random import randint as _randint


ArcType = _namedtuple('Arc', ['input_label', 'output_label', 'nextstate', 'weight'])
PathType = _namedtuple('Path', ['input_path', 'output_path', 'weight'])


class AbstractSemiringWeight(object):
    """
    Defines the base class that all semirings should inherit from.


    Attributes:
        semiring_properties: {'base', 'path'} defines which properties this semiring has as this will construct a different backing FST
    """
    semiring_properties = 'base'

    """
    The zero element of the semiring
    """
    zero = None

    """
    The one element of the semiring
    """
    one = None

    def __add__(self, other):
        """
        Semiring add (oplus).

        Return a new instance of this class that corresponds with: self (+) other
        """
        print('not implemented __add__')
        raise NotImplementedError('__add__')

    def __mul__(self, other):
        """
        Semiring multiply (otimes).

        Return a new instance of this class that corresponds with: self (*) other
        """
        print('not implemented __mul__')
        raise NotImplementedError('__mul__')

    def __div__(self, other):
        """
        Division used in weight pushing

        Return a new instance of this class that corresponds with: self (*) (other)^{-1}
        If not a member of the semiring, then raise an exception
        """
        print('not implemented __div__')
        raise NotImplementedError('__div__')

    def __pow__(self, n):
        """
        Power of self
        n is an int
        """
        print('not implemented __pow__')
        raise NotImplementedError('__pow__')

    def member(self):
        """
        Checks if the current instance is a member of the semiring
        (Eg float is not nan)

        Called from openFst
        """
        return True

    def quantize(self, delta=.5):
        """
        Return a new instance that is bucketed
        """
        print('not implemented _quantize')
        raise NotImplementedError('_quantize')

    def reverse(self):
        """
        Return a weight that represent reversing this edge
        """
        return self

    def sampling_weight(self):
        """
        Return a positive unnormalized floating point value that can be used to sample this arc

        Locally normalized outgoing from a particular state
        """
        print('not implemented sampling_weight')
        raise NotImplementedError('sampling_weight')

    def approx_eq(self, other, delta):
        """
        Returns if this weight is approximately equal to another other less than delta
        """
        print('not implemented approx_eq')
        raise NotImplementedError('approx_eq')

    def __hash__(self):  # hash is required if defining __eq__
        """
        Return the hash code for this instance
        (Will be used by openfst)
        """
        print('not implemented __hash__')
        raise NotImplementedError('__hash__')

    def __eq__(self, other):
        """
        Return if this object is equal to another
        (Will be used by openfst)
        """
        print('not implemented __eq__')
        raise NotImplementedError('__eq__')

    def __truediv__(self, other):
        return self.__div__(other)

    def openfst_str(self):
        """
        Returns a string that is used by OpenFst when performing ToString operations
        """
        return str(self)

    def __repr__(self):
        """
        A representation for printing on the command line
        """
        return f'{type(self).__name__}({str(self)})'

    def __bool__(self):
        return not (self == self.zero)


from . import semirings
from .semirings import (
    PythonValueSemiringWeight,  # semiring for general python values
    RealSemiringWeight,         # Standard <+,*> semiring for real valued scalars
    MinPlusSemiringWeight,      # <min, +> semiring with real value scalars
    MaxPlusSemiringWeight,      # <max, +> semiring with real value scalars
    TropicalSemiringWeight,     # <min, +> semiring with real value scalars
    BooleanSemiringWeight,      # <or, and> semiring for boolean values (special handling by compose and lift)
)


class FST(object):
    """
    Wraps a mutable FST class
    """

    def __init__(self, semiring_class=None, acceptor=False, string_mapper=None, output_string_mapper=None, *, _fst=None):
        if semiring_class is None:
            semiring_class = PythonValueSemiringWeight
        elif type(semiring_class) is not type:
            assert hasattr(semiring_class, '__iter__'), "first argument is not iterable or a semiring class"
            # handle special case where we are dealing with a string construction
            # just build a BooleanFST that can be composed with other machines
            f = FST(BooleanSemiringWeight).create_from_string(semiring_class)
            _fst = f._fst
            semiring_class = BooleanSemiringWeight
            acceptor = True
            string_mapper = f._string_mapper
            output_string_mapper = f._output_string_mapper
        else:
            assert issubclass(semiring_class, AbstractSemiringWeight), "first argument is not iterable or a semiring class"

        if _fst is None:
            # quick sanity check that this implements the semiring class
            zero = semiring_class.zero
            one = semiring_class.one
            assert zero is not None and isinstance(zero, semiring_class), "Zero weight not set for semiring"
            assert one  is not None and isinstance(one , semiring_class), "One weight not set for semiring"

            assert isinstance(zero + one, semiring_class), "Semiring operator + returns the wrong type"
            assert isinstance(zero * one, semiring_class), "Semiring operator * returns the wrong type"

        self._semiring_class = semiring_class

        if _fst is not None:
            self._fst = _fst
        else:
            self._fst = {
                # these classes are defined in c++ to wrap openfst
                'base': _backend.FSTBase,
                'path': _backend.FSTPath,
            }[self._semiring_class.semiring_properties]()

        # if we are an acceptor machine, then we want the input and output labels to be the same
        # setting this to true will automattically copy the input label to the output label in add_arc
        self._acceptor = acceptor
        # the arcs input and output labels are in general ints, but we had support for using a single character string
        # and passing that through ord() and chr() to print them in the graphics that we are drawing.  So if that is being
        # used, then this will get set to true inside of add_arc
        self._string_mapper = string_mapper
        # add output string mapper, which can also be the same alphabet
        self._output_string_mapper = output_string_mapper
        if self._output_string_mapper is None:
            self._output_string_mapper = string_mapper

    def _make_weight(self, w):
        if isinstance(w, self._semiring_class):
            return w
        assert not isinstance(w, AbstractSemiringWeight), "Can not mix different types of weights in a FST"
        if isinstance(w, str):
            # this can be returned by the C++ binding in the case that there is an invalid state
            if w == '__FST_INVALID__':
                return None
            elif w == '__FST_ONE__':
                return self._semiring_class.one
            elif w == '__FST_ZERO__':
                return self._semiring_class.zero
        return self._semiring_class(w)

    def _check_same_fst(self, other):
        """Check that the other fst is the same type as us, otherwise we will run into problems"""
        assert isinstance(other, FST)
        assert self._semiring_class is other._semiring_class, "Can not mix FSTs with different semirings.  Use FST.lift to change between semirings"
        assert type(self._fst) is type(other._fst), "Can not mix FSTs with different properties"

    def constructor(self, _fst=None, **kwargs):
        """Return a new instance of the FST using the same parameters"""
        if _fst is not None:
            assert type(_fst) is type(self._fst), "type of fst differs from our own type"
        else:
            _fst = type(self._fst)()
        params = dict(
            _fst=_fst,
            semiring_class=self._semiring_class,
            acceptor=self._acceptor,
            string_mapper=self._string_mapper,
            output_string_mapper=self._output_string_mapper
        )
        params.update(kwargs)
        return type(self)(**params)

    @property
    def semiring_one(self):
        """Return the semiring's one element"""
        return self._semiring_class.one

    @property
    def semiring_zero(self):
        """Return the semiring's zero element"""
        return self._semiring_class.zero

    @property
    def semiring(self):
        """Return the semiring associated with this FST"""
        return self._semiring_class

    def create_from_string(self, string):
        """
        Creates a FST which converts the empty string (epsilon) to string.
        String can be a normal python string or an iterable (list or tuple) of integers
        """
        ret = self.constructor(acceptor=True)
        last = ret.add_state()
        ret.initial_state = last
        for s in string:  # string can be any iterable object, eg (a normal string or a tuple of ints)
            state = ret.add_state()
            ret.add_arc(last, state, output_label=s, input_label=s)
            last = state
        ret.set_final_weight(last)
        return ret

    def get_unique_output_string(self):
        """
        Returns the string representation in the case that there is only a single path in the fst
        """
        if self.num_states == 0:
            return None  # this is an empty FST
        state = self.initial_state
        seen = set()
        ret = []

        if self._string_mapper is not None:
            mapper = self._string_mapper
        else:
            mapper = lambda x: x

        if self._output_string_mapper is not None:
            omapper = self._output_string_mapper
        else:
            omapper = lambda x: x


        while state != -1:
            edges = list(self.get_arcs(state))
            if len(edges) != 1:
                raise RuntimeError("FST does not contain exactly one path")
            l = edges[0].output_label
            if l != 0:  # the epsilon state
                ret.append(omapper(l))
            if edges[0].nextstate in seen:
                raise RuntimeError("FST contains cycle")
            seen.add(state)
            state = edges[0].nextstate
        if omapper is chr:
            return ''.join(ret)
        return ret

    @property
    def num_states(self):
        """
        Return the number of states currently set on the fst
        """
        return self._fst.NumStates()

    @property
    def states(self):
        """
        An iterator over state ids inside of the FST.
        """
        yield from range(self.num_states)

    def num_arcs(self, state):
        """
        Return the number of arcs in the fst
        """
        return self._fst.NumArcs(state)

    @property
    def initial_state(self):
        """
        Return the state id of the starting state
        """
        return self._fst.Start()

    @initial_state.setter
    def initial_state(self, state):
        """
        Mark a state as the start state
        """
        assert (state >= 0 and state < self.num_states), "Invalid state id"
        return self._fst.SetStart(state)

    def set_initial_state(self, state):
        self.initial_state = state

    def add_state(self):
        """
        Add a new state to the FST
        Return this state's id
        """
        return self._fst.AddState()

    def add_arc(self, from_state, to_state,
                weight='__FST_ONE__', input_label=0, output_label=0):
        """
        Add an arc between states from->to with weight (default 1).
        input_label and output label should be ints that map to a label (0 == epsilon)
        """
        assert (from_state >= 0 and from_state < self.num_states and
                to_state >= 0 and to_state < self.num_states), "Invalid state id"
        if isinstance(input_label, str):
            assert len(input_label) == 1, "FST string labels can only be a single character"
            input_label = ord(input_label)
            if self._string_mapper is None:
                self._string_mapper = chr
        if isinstance(output_label, str):
            assert len(output_label) == 1, "FST string labels can only be a single character"
            output_label = ord(output_label)
            if self._output_string_mapper is None:
                self._output_string_mapper = chr
        if self._acceptor:
            # acceptors are machines with the same input and output label
            if output_label == 0:  # if not set just copy the value
                output_label = input_label
            else:
                assert output_label == input_label, "a FSA requires that the input and output labels are equal on all arcs"
        return self._fst.AddArc(from_state, to_state, input_label, output_label,
                                self._make_weight(weight))

    def delete_arcs(self, state):
        """
        Delete all arcs coming out of state
        """
        assert (state >= 0 and state < self.num_states), "Invalid state id"
        return self._fst.DeleteArcs(state)

    def delete_states(self):
        """
        Delete all states in the FST
        """
        return self._fst.DeleteStates()

    def set_final_weight(self, state, weight='__FST_ONE__'):
        """
        Set the weight that this state transisions to the final state (default weight 1)
        """
        assert (state >= 0 and state < self.num_states), "Invalid state id"
        return self._fst.SetFinal(state, self._make_weight(weight))

    def get_final_weight(self, state):
        """
        Get the weight of transistioning to the final state
        """
        assert (state >= 0 and state < self.num_states), "Invalid state id"
        return self._make_weight(self._fst.FinalWeight(state))

    def get_arcs(self, state):
        """
        Return the arcs coming out of some state
        """
        assert (state >= 0 and state < self.num_states), "Invalid state id"
        return [
            ArcType(ilabel, olabel, nextstate, self._make_weight(weight))
            for ilabel, olabel, nextstate, weight in self._fst.ArcList(state)
        ]

    def isomorphic(self, other, delta=1.0/1024):
        """
        This operation determines if two transducers with a certain required
        determinism have the same states, irrespective of numbering, and the
        same transitions with the same labels and weights, irrespective of
        ordering. In other words, Isomorphic(A, B) is true if and only if the
        states of A can be renumbered and the transitions leaving each state
        reordered so that Equal(A, B) is true.

        http://www.openfst.org/twiki/bin/view/FST/IsomorphicDoc

        uses AbstractSemiring._approx_eq to compare wieghts.
        delta: 32 bit floating point number that is passed to _approx_eq
        """
        self._check_same_fst(other)
        return self._fst.Isomorphic(other._fst, delta)

    def concat(self, other):
        """
        This operation computes the concatenation (product) of two FSTs. If A
        transduces string x to y with weight a and B transduces string w to v
        with weight b, then their concatenation transduces string xw to yv with
        weight a (otimes) b.

        http://www.openfst.org/twiki/bin/view/FST/ConcatDoc
        """
        self._check_same_fst(other)
        return self.constructor(self._fst.Concat(other._fst))

    def compose(self, *other_fsts):
        """
        This operation computes the composition of two transducers. If A transduces
        string x to y with weight a and B transduces y to z with weight b, then their
        composition transduces string x to z with weight a (otimes) b.

        http://www.openfst.org/twiki/bin/view/FST/ComposeDoc

        Will efficiently handle one or more FSTs composing self :: arg1 :: arg2 :: .... :: argN
        """
        assert len(other_fsts) > 0, "Require at least one other FST to compose with"
        # deal with the case that there may be a boolean semiring in which case we want to
        # convert this into the other semiring that we are using
        fsts = [self] + list(other_fsts)
        target_semiring = None
        for fst in fsts:
            target_semiring = fst.semiring
            if target_semiring is not BooleanSemiringWeight:
                break
        if target_semiring is not BooleanSemiringWeight:
            fsts = [f if not (f.semiring is BooleanSemiringWeight and f._acceptor) else
                    f.disambiguate().lift(target_semiring)
                    for f in fsts]
        s, *others = fsts
        for fst in others:
            s._check_same_fst(fst)
        return s.constructor(
            s._fst.Compose([f._fst for f in others]),
            acceptor=all([f._acceptor for f in fsts])
        )

    def determinize(self, delta=1.0/1024, weight_threshold=None, *, allow_non_functional=False):
        """
        This operation determinizes a weighted transducer. The result will be an
        equivalent FST that has the property that no state has two transitions
        with the same input label. For this algorithm, epsilon transitions are
        treated as regular symbols (cf. RmEpsilon).

        http://www.openfst.org/twiki/bin/view/FST/DeterminizeDoc

        delta: Quantization delta for binning weights.
        weight_threshold: Pruning weight threshold.
        allow_non_functional: Only works on path semirings.  In the case that there are two
          output sequences for a given input the one with the shorter path
          will be used and the longer path discarded.
        """
        if weight_threshold is None:
            weight_threshold = self._semiring_class.zero

        return self.constructor(self._fst.Determinize(
            self._semiring_class,
            delta,
            self._make_weight(weight_threshold),
            allow_non_functional
        ))

    def disambiguate(self):
        """
        This operation disambiguates a weighted transducer. The result will be an
        equivalent FST that has the property that no two successful paths have
        the same input labeling. For this algorithm, epsilon transitions are
        treated as regular symbols (cf. RmEpsilon).
        """
        return self.constructor(self._fst.Disambiguate(self._semiring_class))

    def project(self, side='input'):
        """
        This operation projects an FST onto its domain or range by either copying
        each arc's input label to its output label or vice versa.

        http://www.openfst.org/twiki/bin/view/FST/ProjectDoc
        """
        if side == 'output':
            t = 0
        elif side == 'input':
            t = 1
        else:
            raise RuntimeError("Unknown project side, expected input or output labels")
        return self.constructor(self._fst.Project(t))

    def difference(self, other):
        """
        This operation computes the difference between two FSAs. Only strings that
        are in the first automaton but not in second are retained in the
        result.

        http://www.openfst.org/twiki/bin/view/FST/DifferenceDoc
        """
        self._check_same_fst(other)
        return self.constructor(self._fst.Difference(other._fst))

    def invert(self):
        """
        This operation inverts the transduction corresponding to an FST by
        exchanging the FST's input and output labels.

        http://www.openfst.org/twiki/bin/view/FST/InvertDoc
        """
        return self.constructor(self._fst.Invert())

    def prune(self, weight):
        """
        This operation deletes states and arcs in the input FST that do not belong
        to a successful path whose weight is no more (w.r.t the natural the
        natural semiring order) than the threshold t (otimes) the weight of the
        shortest path in the input FST.

        http://www.openfst.org/twiki/bin/view/FST/PruneDoc
        """
        return self.constructor(self._fst.Prune(self._make_weight(weight)))

    def union(self, other):
        """
        This operation computes the union (sum) of two FSTs. If A transduces string
        x to y with weight a and B transduces string w to v with weight b, then their
        union transduces x to y with weight a and w to v with weight b.

        http://www.openfst.org/twiki/bin/view/FST/UnionDoc
        """
        self._check_same_fst(other)
        return self.constructor(self._fst.Union(other._fst))

    def intersect(self, other):
        """
        This operation computes the intersection (Hadamard product) of two
        FSAs. Only strings that are in both automata are retained in the
        result.

        http://www.openfst.org/twiki/bin/view/FST/IntersectDoc
        """
        s = self
        if other.semiring is BooleanSemiringWeight and other._acceptor:
            if s.semiring is not BooleanSemiringWeight:
                other = other.disambiguate().lift(s.semiring)
        elif s.semiring is BooleanSemiringWeight and s._acceptor:
            s = s.disambiguate().lift(other.semiring)
        s._check_same_fst(other)
        return s.constructor(s._fst.Intersect(other._fst))

    def push(self, towards='initial'):
        """
        This operation produces an equivalent transducer by pushing the weights
        and/or the labels towards the initial state or toward the final states.

        http://www.openfst.org/twiki/bin/view/FST/PushDoc
        """
        if towards == 'initial':
            t = 0
        elif towards == 'final':
            t = 1
        else:
            raise RuntimeError("Unknown direction for weight pushing, expected final or initial")
        return self.constructor(self._fst.Push(self._semiring_class, t))

    def minimize(self, delta=1./1024):
        """
        This operation performs the minimization of deterministic weighted automata
        and transducers.

        If the input FST A is an automaton (acceptor), this operation produces the
        minimal automaton B equivalent to A, i.e. the automata with a minimal number of
        states that is equivalent to A.

        If the input FST A is a transducer, this operation internally builds an
        equivalent transducer with a minimal number of states. However, this minimality
        is obtained by allowing transition having strings of symbols as output labels,
        this known in the litterature as a real-time transducer. Such transducers are
        not directly supported by the library. By defaut, Minimize will convert such
        transducer by expanding each string-labeled transition into a sequence of
        transitions. This will results in the creation of new states, hence losing the
        minimality property. If a second output argument is given to Minimize, then the
        first output B will be the minimal real-time transducer with each strings that
        is the output label of a transition being mapped to a new output symbol, the
        second output transducer C represents the mapping between new output labels and
        old output labels. Hence, we will have that A is equivalent to B o C.

        http://www.openfst.org/twiki/bin/view/FST/MinimizeDoc
        """
        return self.constructor(self._fst.Minimize(delta))

    def shortest_path(self, count=1):
        """
        This operation produces an FST containing the n -shortest paths in the input
        FST. The n -shortest paths are the n -lowest weight paths w.r.t. the
        natural semiring order. The single path that can be read from the ith of
        at most n transitions leaving the initial state of the resulting FST is
        the ith shortest path.

        The weights need to be right distributive and have the path property. They also
        need to be left distributive as well for n -shortest with n > 1 (valid for
        TropicalWeight).

        http://www.openfst.org/twiki/bin/view/FST/ShortestPathDoc

        This uses the ShortestFirstQueue.  It works in the case that there are cycles
        and no negative weights
        """
        return self.constructor(self._fst.ShortestPath(self._semiring_class, count))

    def shortest_distance(self, reverse=False):
        """
        This operation computes the shortest distance from the initial state to
        every state (when reverse is false) or from every state to the final
        states (when reverse is true). The shortest distance from p to q is the
        (oplus)-sum of the weights of all the paths between p and q.
        """
        return [self._make_weight(w) for w in self._fst.ShortestDistance(reverse)]

    def sum_paths(self):
        """
        Return the sum of the weight of all successful paths in an FST, i.e., the
        shortest-distance from the initial state to the final states. Returns a
        weight such that Member() is false if an error was encountered.
        """
        return self._make_weight(self._fst.SumPaths())

    def topo_sort(self):
        """
        This operation topologically sorts its input if acyclic.
        When sorted, all transitions are from lower to higher state IDs.
        """
        return self.constructor(self._fst.TopSort())

    def random_path(self, count=1):
        """
        This operation randomly generates a set of successful paths in the input
        FST. The operation relies on an ArcSelector object for randomly
        selecting an outgoing transition at a given state in the input FST. The
        default arc selector, UniformArcSelector, randomly selects a transition
        using the uniform distribution. LogProbArcSelector randomly selects a
        transition w.r.t. the weights treated as negative log probabilities
        after normalizing for the total weight leaving the state. In all cases,
        finality is treated as a transition to a super-final state.

        This uses Weight._sampling_weight to get an unormalized weight for each arc

        http://www.openfst.org/twiki/bin/view/FST/RandGenDoc
        """
        # use python random, so if it is seeded then it will be consistent across uses
        s = _randint(0, 2 ** 63)
        return self.constructor(self._fst.RandomPath(count, s))

    def remove_epsilon(self):
        """
        This operation removes epsilon-transitions (when both the input and output
        label are an epsilon) from a transducer. The result will be an
        equivalent FST that has no such epsilon transitions.

        http://www.openfst.org/twiki/bin/view/FST/RmEpsilonDoc
        """
        return self.constructor(self._fst.RmEpsilon(self._semiring_class))

    def lift(self, semiring=None, converter=None):
        """
        This operation builds a new FST that accepts the same inputs and outputs as this FST
        but the weights have been converted into the new semiring
        """
        if not semiring:
            semiring = self._semiring_class
        if not converter:
            if self._semiring_class is BooleanSemiringWeight:
                converter = lambda x: semiring.one if x else semiring.zero
            else:
                converter = lambda x: x

        ret = FST(semiring, acceptor=self._acceptor, string_mapper=self._string_mapper, output_string_mapper=self._output_string_mapper)
        zero = self.semiring_zero
        for i in range(self.num_states):
            ret.add_state()  # would be nice if this did not need to be called in a loop
        for i in range(self.num_states):
            for arc in self.get_arcs(i):
                if arc.nextstate == -1:
                    if arc.weight != zero:  # some openfst algorithms add zero weights to the final edge (like determinize)
                        # then this is a final state
                        w = converter(arc.weight)
                        if w is not None:
                            ret.set_final_weight(i, weight=w)
                else:
                    w = converter(arc.weight)
                    if w is not None:
                        ret.add_arc(i, arc.nextstate,
                                    weight=w,
                                    input_label=arc.input_label,
                                    output_label=arc.output_label)
        ret.initial_state = self.initial_state
        return ret

    def verify(self):
        """
        This operation checks the sanity of a FST's contents. It returns false if
        the transducer is incomplete or ill-formed (e.g., a non-trivial FST
        that has no initial state or transitions to non-existent destination
        states).
        """
        return self._fst.Verify()

    def closure(self, mode='star'):
        """
        This operation computes the concatenative closure. If A transduces string x
        to y with weight a, then the closure transduces x to y with weight a, xx
        to yy with weight a (otimes) a, xxx to yyy with weight a (otimes) a
        (otimes) a, etc. If closure_type is CLOSURE_STAR, then the empty string
        is transduced to itself with weight 1 as well.
        """
        if mode == 'star':
            t = 0
        elif mode == 'plus':
            t = 1
        else:
            raise RuntimeError('closure expects mode of star or plus')
        return self.constructor(self._fst.Closure(t))

    def reverse(self):
        """
        This operation reverses an FST. If A transduces string x to y with weight a,
        then the reverse of A transduces the reverse of x to the reverse of y
        with weight a.Reverse().
        """
        return self.constructor(self._fst.Reverse())

    def iterate_paths(self, start=None):
        """
        Return an iterator over paths through the FST.
        Each path is of the PathType and contains the (input sequence, output sequence, path weight sum)
        """
        if start is None:
            start = self.initial_state

        # basic sanity check before we start trying to construct lists of states
        assert self.verify()

        zero = self.semiring_zero
        if self._string_mapper is None:
            mapper = lambda x: x
        elif self._string_mapper is chr:
            def mapper(x):
                return ''.join(chr(y) for y in x)
        else:
            def mapper(x):
                return tuple(self._string_mapper(y) for y in x)

        # run BFS
        queue = _deque([(tuple(), tuple(), self.semiring_one, start)])

        while queue:
            input_path, output_path, sweight, state = queue.popleft()
            for input_label, output_label, nextstate, weight in self.get_arcs(state):
                if zero != weight:
                    if nextstate == -1:
                        # this is a final state
                        yield PathType(mapper(input_path), mapper(output_path), sweight * weight)
                    else:
                        ip = input_path
                        op = output_path
                        if input_label != 0:
                            ip += (input_label,)
                        if output_label != 0:
                            op += (output_label,)
                        queue.append((
                            ip, op,
                            sweight * weight,
                            nextstate
                        ))

    def __str__(self):
        if self.num_states < 10 and sum(self.num_arcs(s) for s in range(self.num_states)) < 300:
            # if the FST is small enough that we want to print the whole thing in the string
            return self.full_str()
        else:
            return 'FST(num_states={})'.format(self.num_states)

    def full_str(self):
        return 'FST {\n' + self._fst.ToString() + '}'

    def __repr__(self):
        return str(self)

    def __bool__(self):
        return self.num_states > 0

    def __getstate__(self):
        return {
            'semiring_class': self._semiring_class,
            'acceptor': self._acceptor,
            'string_mapper': self._string_mapper,
            'output_string_mapper': self._output_string_mapper,
            'num_states': self.num_states,
            'arcs': [[tuple(x) for x in self.get_arcs(s)] for s in self.states],
            'initial_state': self.initial_state,
        }

    def __setstate__(self, d):
        f = FST(
            semiring_class=d['semiring_class'],
            acceptor=d['acceptor'],
            string_mapper=d['string_mapper'],
            output_string_mapper=d['output_string_mapper']
        )
        self._fst = f._fst
        self._semiring_class = f._semiring_class
        self._acceptor = f._acceptor
        self._string_mapper = f._string_mapper
        self._output_string_mapper = f._output_string_mapper
        for i in range(d['num_states']):
            self.add_state()
        self.initial_state = d['initial_state']
        for i, arcs in enumerate(d['arcs']):
            for input_label, output_label, nextstate, weight in arcs:
                if nextstate == -1:
                    self.set_final_weight(state=i, weight=weight)
                else:
                    self.add_arc(from_state=i, to_state=nextstate, weight=weight,
                                 input_label=input_label, output_label=output_label)

    def _repr_html_(self):
        """
        When returned from a Jupyter cell, this will generate the FST visualization
        """
        # mostly copied from dagre-d3 tutorial / demos
        from uuid import uuid4
        import json
        from collections import defaultdict
        ret = []
        if self.num_states == 0:
            return '<code>Empty FST</code>'

        if self.num_states > 1200:
            return f'FST too large to draw graphic, use fst.full_str()<br /><code>FST(num_states={self.num_states})</code>'

        # here we are actually going to read the states from the FST and generate nodes for them
        # in the output source code
        zero = self._make_weight('__FST_ZERO__')
        one = self._make_weight('__FST_ONE__')
        initial_state = self.initial_state

        for sid in range(self.num_states):
            finalW = ''
            is_final = False
            ww = self._fst.FinalWeight(sid)
            if ww is not None and (not isinstance(ww, str) or '__FST_ONE__' == ww):  # look at the raw returned value to see if it is zero (unset)
                ww = self._make_weight(ww)
                if zero != ww:
                    is_final = True
                    if not (one == ww and sid != initial_state):
                        finalW = f'\n({ww})'
            label = f'{sid}{finalW}'

            ret.append(f'g.setNode("state_{sid}", {{ label: {json.dumps(label)} , shape: "circle" }});\n')
            if is_final:
                # make the final states red
                ret.append(f'g.node("state_{sid}").style = "fill: #f77"; \n')

        if self._string_mapper is not None:
            if self._string_mapper is chr:
                def make_label(x):
                    if x == 32:
                        return '(spc)'
                    elif x < 32:
                        return str(x)
                    else:
                        return chr(x)
            else:
                make_label = self._string_mapper
        else:
            make_label = str

        if self._output_string_mapper is not None:
            if self._output_string_mapper is chr:
                def make_olabel(x):
                    if x == 32:
                        return '(spc)'
                    elif x < 32:
                        return str(x)
                    else:
                        return chr(x)
            else:
                make_olabel = self._output_string_mapper
        else:
            make_olabel = make_label

        for sid in range(self.num_states):
            to = defaultdict(list)
            for arc in self.get_arcs(sid):
                if arc.nextstate == -1:
                    continue

                label = 'L:'
                if arc.input_label == 0:
                    label += '\u03B5'  # epsilon
                else:
                    label += make_label(arc.input_label)
                if arc.output_label == 0:
                    olabel = '\u03B5'  # epsilon
                else:
                    olabel = make_olabel(arc.output_label)
                
                if label != olabel:
                    label += ':'+olabel
                
                if one != arc.weight:
                    label += f'/{arc.weight}'
                to[arc.nextstate].append(label)
            for dest, values in to.items():
                if len(values) > 3:
                    values = values[0:2] + ['. . .']
                label = '\n'.join(values)
                ret.append(f'g.setEdge("state_{sid}", "state_{dest}", {{ arrowhead: "vee", label: {json.dumps(label)} }});\n')

        if initial_state >= 0:
            # make the start state green
            ret.append(f'g.node("state_{initial_state}").style = "fill: #7f7"; \n')

        # if the machine is too big, do not attempt to make the web browser display it
        # otherwise it ends up crashing and stuff...
        if len(ret) > 1200:
            return f'FST too large to draw graphic, use fst.full_str()<br /><code>FST(num_states={self.num_states})</code>'


        ret2 = ['''
        <script>
        try {
        require.config({
        paths: {
        "d3": "https://cdnjs.cloudflare.com/ajax/libs/d3/4.13.0/d3",
        "dagreD3": "https://cdnjs.cloudflare.com/ajax/libs/dagre-d3/0.6.1/dagre-d3.min"
        }
        });
        } catch {
          ["https://cdnjs.cloudflare.com/ajax/libs/d3/4.13.0/d3.js",
           "https://cdnjs.cloudflare.com/ajax/libs/dagre-d3/0.6.1/dagre-d3.min.js"].forEach(function (src) {
            var tag = document.createElement('script');
            tag.src = src;
            document.body.appendChild(tag);
          })
        }
        try {
        requirejs(['d3', 'dagreD3'], function() {});
        } catch (e) {}
        try {
        require(['d3', 'dagreD3'], function() {});
        } catch (e) {}
        </script>
        <style>
        .node rect,
        .node circle,
        .node ellipse {
        stroke: #333;
        fill: #fff;
        stroke-width: 1px;
        }

        .edgePath path {
        stroke: #333;
        fill: #333;
        stroke-width: 1.5px;
        }
        </style>
        ''']


        obj = 'fst_' + uuid4().hex
        ret2.append(f'<center><svg width="850" height="600" id="{obj}"><g/></svg></center>')
        ret2.append('''
        <script>
        (function render_d3() {
        var d3, dagreD3;
        try { // requirejs is broken on external domains
          d3 = require('d3');
          dagreD3 = require('dagreD3');
        } catch (e) {
          // for google colab
          if(typeof window.d3 !== "undefined" && typeof window.dagreD3 !== "undefined") {
            d3 = window.d3;
            dagreD3 = window.dagreD3;
          } else { // not loaded yet, so wait and try again
            setTimeout(render_d3, 50);
            return;
          }
        }
        //alert("loaded");
        var g = new dagreD3.graphlib.Graph().setGraph({ 'rankdir': 'LR' });
        ''')
        ret2.append(''.join(ret))

        ret2.append(f'var svg = d3.select("#{obj}"); \n')
        ret2.append('''
        var inner = svg.select("g");

        // Set up zoom support
        var zoom = d3.zoom().scaleExtent([0.3, 5]).on("zoom", function() {
        inner.attr("transform", d3.event.transform);
        });
        svg.call(zoom);

        // Create the renderer
        var render = new dagreD3.render();

        // Run the renderer. This is what draws the final graph.
        render(inner, g);

        // Center the graph
        var initialScale = 0.75;
        svg.call(zoom.transform, d3.zoomIdentity.translate((svg.attr("width") - g.graph().width * initialScale) / 2, 20).scale(initialScale));

        svg.attr('height', g.graph().height * initialScale + 50);
        })();
        </script>
        ''')
        return ''.join(ret2)

class SymbolTable(object):
    """
    Creates a symbol table that maps strings to ids; can function as a callable
    to pass as input to FST constructor.
    """
    def __init__(self,mutableOnFly=False,strict=False):
        self.__sym2id = {}
        self.__id2sym = {}
        self.__mutableOnFly = mutableOnFly
        self.__strict = strict

    def add_symbol(self,sym):
        if sym in self.__sym2id:
            if strict:
                raise Exception('Symbol '+sym+' already in symbol table')
            return self.__sym2id[sym]
        newid=len(self.__sym2id)
        self.__sym2id[sym]=newid
        self.__id2sym[newid]=sym
        return newid
    
    def get_symbol(self,sym):
        if sym in self.__sym2id:
            return self.__sym2id[sym]
        elif self.__mutableOnFly:
            return self.add_symbol(sym)
        else:
            raise Exception('Symbol '+sym+' not in symbol table')

    def __getitem__(self, sym):
        return self.get_symbol(sym)
    
    def __call__(self,symid):
        if symid in self.__id2sym:
            return self.__id2sym[symid]
        else:
            return symid

import re

def compiler(strings,isymbols=None,osymbols=None,acceptor=False,add_symbols=True):

    # create isymbols, osymbols
    if add_symbols and isymbols is None:
        isymbols=SymbolTable(mutableOnFly=add_symbols)
        isymbols.add_symbol("-")
    if add_symbols and not acceptor and osymbols is None:
        osymbols=SymbolTable(mutableOnFly=add_symbols)
        osymbols.add_symbol("-")

    if (acceptor):
        f=FST(acceptor=acceptor,string_mapper=isymbols)
    else:
        f=FST(acceptor=acceptor, string_mapper=isymbols, output_string_mapper=osymbols)

    # always have 0 be the initial state per OpenFST compilation standards
    states={0: f.add_state()}
    f.set_initial_state(states[0])

    for s in strings:
        parts=re.split('\s+',s)
        weight=1
        max=5
        if acceptor:
            max=4

        parts[0]=int(parts[0])
        if len(parts)>2:
            parts[1]=int(parts[1])
        if len(parts)>max:
            raise Exception('Syntax error: '+s)
        if len(parts)==max:
            weight=float(parts[max-1])
        if len(parts)>=max-1:
            if parts[0] not in states:
                states[parts[0]]=f.add_state()
            s0=states[parts[0]]

            if parts[1] not in states:
                states[parts[1]]=f.add_state()
            s1=states[parts[1]]
            isym=isymbols[parts[2]]
            if acceptor:
                f.add_arc(s0,s1,weight,isym)
            else:
                osym=osymbols[parts[3]]
                f.add_arc(s0,s1,weight,isym,osym)
        elif len(parts)==2:
            if parts[0] not in states:
                states[parts[0]]=f.add_state()
            s0=states[parts[0]]
            f.set_final_weight(s0,parts[1])
        elif len(parts)==1:
            if parts[0] not in states:
                states[parts[0]]=f.add_state()
            s0=states[parts[0]]
            f.set_final_weight(s0)
        else:
            raise Exception('Syntax error: '+s)
    return f
