# Written by Matthew Francis-Landau (2018)
#
# Wrapper for OpenFST that supports defining custom semirings in python
# and drawing FSTs in ipython notebooks


import openfst_wrapper_backend as _backend
from collections import namedtuple as _namedtuple
from random import randint as _randint

ArcType = _namedtuple('Arc', ['ilabel', 'olabel', 'nextstate', 'weight'])


class AbstractSemiring(object):

    @classmethod
    def zero(cls):
        """
        Returns the zero element of the semiring
        """
        return cls(0)

    @classmethod
    def one(cls):
        """
        Returns the one element of the semiring
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

    def _quantize(self, delta=.5):
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


class ValueSemiring(AbstractSemiring):

    def __init__(self, value=0):
        super().__init__()
        # the value should be immutable, so access via property
        self.__value = value

    @property
    def value(self):
        return self.__value

    @classmethod
    def _create(cls, v):
        return cls(v)

    @classmethod
    def zero(cls):
        return cls(0)

    @classmethod
    def one(cls):
        return cls(1)

    def __add__(self, other):
        assert type(other) is type(self)
        return self._create(self.value + other.value)

    def __mul__(self, other):
        assert type(other) is type(self)
        return self._create(self.value * other.value)

    def __div__(self, other):
        assert type(other) is type(self)
        return self._create(self.value / other.value)

    def __pow__(self, n):
        return self._create(self.value ** n)

    def _member(self):
        # check that this is a member of the semiring
        # this is just a nan check atm
        return self.value == self.value

    def _quantize(self, delta=.5):
        # quantize the weight into buckets
        return self

    def _sampling_weight(self):
        # just make the sampling of these path weights uniform
        return 1

    def _approx_eq(self, other, delta):
        return abs(self.value - other.value) < delta

    def __str__(self):
        return str(self.value)

    def __hash__(self):
        return hash(self.value)

    def __eq__(self, other):
        return isinstance(other, ValueSemiring) and self.value == other.value

    def __float__(self):
        return float(self.value)

    def __int__(self):
        return int(self.value)

    def __repr__(self):
        return f'{type(self).__name__}({self.value})'


class FST(object):
    """
    Wraps a mutable FST class
    """

    def __init__(self, fst='base', semiring_class=None, acceptor=False, _character_fst=False):
        if isinstance(fst, str):
            self._fst = {
                # these classes are defined in c++ to wrap openfst
                'base': _backend.FSTBase,
                'path': _backend.FSTPath,
            }[fst]()
        else:
            self._fst = fst

        # if we are an acceptor machine, then we want the input and output labels to be the same
        # setting this to true will automattically copy the input label to the output label in add_arc
        self._acceptor = acceptor
        # the arcs input and output labels are in general ints, but we had support for using a single character string
        # and passing that through ord() and chr() to print them in the graphics that we are drawing.  So if that is being
        # used, then this will get set to true inside of add_arc
        self._character_fst = _character_fst

        if not semiring_class:
            self._semiring_class = ValueSemiring
        else:
            # quick sanity check that this implements the semiring class
            assert issubclass(semiring_class, AbstractSemiring)
            zero = semiring_class.zero()
            one = semiring_class.one()
            assert isinstance(zero, semiring_class)
            assert isinstance(one, semiring_class)
            assert isinstance(zero + one, semiring_class)
            assert isinstance(zero * one, semiring_class)

            self._semiring_class = semiring_class

    def _make_weight(self, w):
        if isinstance(w, self._semiring_class):
            return w
        assert not isinstance(w, AbstractSemiring), "Can not mix different types of weights in a FST"
        if isinstance(w, str):
            # this can be returned by the C++ binding in the case that there is an invalid state
            if w == '__FST_INVALID__':
                return None
            elif w == '__FST_ONE__':
                return self._semiring_class.one()
            elif w == '__FST_ZERO__':
                return self._semiring_class.zero()
        return self._semiring_class(w)

    def _check_same_fst(self, other):
        """Check that the other fst is the same type as us, otherwise we will run into problems"""
        assert isinstance(other, FST)
        assert type(self._fst) is type(other._fst), "Can not mix FSTs with different properties"
        assert self._semiring_class is other._semiring_class, "Can not mix FSTs with different semirings"

    def constructor(self, _fst=None):
        """Return a new instance of the FST using the same parameters"""
        if _fst:
            assert type(_fst) is type(self._fst), "type of fst differs from our own type"
        else:
            _fst = type(self._fst)()
        return type(self)(
            fst=_fst,
            semiring_class=self._semiring_class,
            acceptor=self._acceptor,
            _character_fst=self._character_fst
        )

    @property
    def semiring_one(self):
        """Return the semiring's one element"""
        return self._semiring_class.one()

    @property
    def semiring_zero(self):
        """Return the semiring's zero element"""
        return self._semiring_class.zero()

    @property
    def semiring(self):
        """Return the semiring associated with this FST"""
        return self._semiring_class

    def create_from_string(self, string):
        """
        Creates a FST which converts the empty string (epsilon) to string.
        String can be a normal python string or an iterable (list or tuple) of integers
        """
        ret = self.constructor()
        last = ret.add_state()
        ret.start_state = last
        for s in string:  # string can be any iterable object, eg (a normal string or a tuple of ints)
            state = ret.add_state()
            if self._acceptor:
                ret.add_arc(last, state, output_label=s, input_label=s)
            else:
                ret.add_arc(last, state, output_label=s)
            last = state
        if last:
            ret.set_final_weight(last)
        return ret

    def get_string(self):
        """
        Returns the string representation in the case that there is only a single path in the fst
        """
        if self.num_states == 0:
            return None  # this is an empty FST
        state = self.start_state
        seen = set()
        ret = []
        while state != -1:
            edges = list(self.get_arcs(state))
            if len(edges) != 1:
                raise RuntimeError("FST does not contain exactly one path")
            l = edges[0].olabel
            if l != 0:  # the epsilon state
                if self._character_fst:
                    ret.append(chr(l))
                else:
                    ret.append(l)
            if edges[0].nextstate in seen:
                raise RuntimeError("FST contains cycle")
            seen.add(state)
            state = edges[0].nextstate
        if self._character_fst:
            return ''.join(ret)
        else:
            # return a tuple of ints that represents which labels where generated by the fst
            return tuple(ret)

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
    def start_state(self):
        """
        Return the state id of the starting state
        """
        return self._fst.Start()

    @start_state.setter
    def start_state(self, state):
        """
        Mark a state as the start state
        """
        assert (state >= 0 and state < self.num_states), "Invalid state id"
        return self._fst.SetStart(state)

    def set_start_state(self, state):
        self.start_state = state

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
        # assert if the state is valid otherwise openfst calls exit(1)
        assert (from_state >= 0 and from_state < self.num_states and
                to_state >= 0 and to_state < self.num_states), "Invalid state id"
        if isinstance(input_label, str):
            assert len(input_label) == 1, "FST string labels can only be a single character"
            input_label = ord(input_label)
            self._character_fst = True
        if isinstance(output_label, str):
            assert len(output_label) == 1, "FST string labels can only be a single character"
            output_label = ord(output_label)
            self._character_fst = True
        if self._acceptor:
            # acceptors are machines with the same input and output label
            output_label = input_label
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
        Set the weight that this state transisions to the final state
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

    # methods for changing the fst given anther fst
    def concat(self, other):
        """
        This operation computes the concatenation (product) of two FSTs. If A
        transduces string x to y with weight a and B transduces string w to v
        with weight b, then their concatenation transduces string xw to yv with
        weight a (times) b.

        http://www.openfst.org/twiki/bin/view/FST/ConcatDoc
        """
        self._check_same_fst(other)
        return self.constructor(self._fst.Concat(other._fst))

    def compose(self, other):
        """
        This operation computes the composition of two transducers. If A transduces
        string x to y with weight a and B transduces y to z with weight b, then their
        composition transduces string x to z with weight a (times) b.

        http://www.openfst.org/twiki/bin/view/FST/ComposeDoc
        """
        self._check_same_fst(other)
        return self.constructor(self._fst.Compose(other._fst))

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
            weight_threshold = self._semiring_class()

        return self.constructor(self._fst.Determinize(delta, self._make_weight(weight_threshold)))

    def project(self, type='input'):
        """
        This operation projects an FST onto its domain or range by either copying
        each arc's input label to its output label or vice versa.

        http://www.openfst.org/twiki/bin/view/FST/ProjectDoc
        """
        if type == 'output':
            t = 0
        elif type == 'input':
            t = 1
        else:
            raise RuntimeError("unknown project type " + type)
        return self.constructor(self._fst.Project(t))

    def difference(self, other):
        """
        This operation computes the difference between two FSAs. Only strings that
        are in the first automaton but not in second are retained in the
        result.

        http://www.openfst.org/twiki/bin/view/FST/DifferenceDoc
        """
        self._check_same_fst(other)
        assert self.constructor(self._fst.Difference(other._fst))

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
        natural semiring order) than the threshold t otimes the weight of the
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
        self._check_same_fst(other)
        return self.constructor(self._fst.Intersect(other._fst))

    def push(self, towards='final'):
        """
        This operation produces an equivalent transducer by pushing the weights
        and/or the labels towards the initial state or toward the final states.

        http://www.openfst.org/twiki/bin/view/FST/PushDoc
        """
        if towards == 'final':
            t = 1
        elif towards == 'initial':
            t = 0
        else:
            raise RuntimeError("Unknown direction for weight pushing")
        return self.constructor(self._fst.Push(t))

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
        """
        return self.constructor(self._fst.ShortestPath(count))

    def shortest_distance(self, reverse=False):
        """
        This operation computes the shortest distance from the initial state to
        every state (when reverse is false) or from every state to the final
        states (when reverse is true). The shortest distance from p to q is the
        oplus-sum of the weights of all the paths between p and q.
        """
        return [self._make_weight(w) for w in self._fst.ShortestDistance(reverse)]

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
        return self.constructor(self._fst.RmEpsilon())

    def __str__(self):
        if self.num_states < 10:
            # if the FST is small enough that we might want to print the whole thing in the string
            return self.full_str()
        else:
            return 'FST(num_states={})'.format(self.num_states)

    def full_str(self):
        return 'FST {\n' + self._fst.ToString() + '\n}'

    def __repr__(self):
        return str(self)

    def __bool__(self):
        return self.num_states > 0

    def _repr_html_(self):
        """
        When returned from an ipython cell, this will generate the FST visualization
        """
        # mostly copied from dagre-d3 tutorial / demos
        from uuid import uuid4
        import json
        ret = ''
        if self.num_states == 0:
            return '<code>Empty FST</code>'
        # sigh...loading these as external files
        # ipython is loading with require js
        ret += '''
        <script>
        require.config({
        paths: {
        "d3": "https://cdnjs.cloudflare.com/ajax/libs/d3/4.13.0/d3",
        "dagreD3": "https://cdnjs.cloudflare.com/ajax/libs/dagre-d3/0.6.1/dagre-d3.min"
        }
        });
        </script>
        <!--script>
        (function() {
        if(typeof d3 == "undefined") {
        var script   = document.createElement("script");
        script.type  = "text/javascript";
        script.src   = "https://cdnjs.cloudflare.com/ajax/libs/d3/4.13.0/d3.js";
        document.body.appendChild(script);

        var script   = document.createElement("script");
        script.type  = "text/javascript";
        script.src   = "https://cdnjs.cloudflare.com/ajax/libs/dagre-d3/0.6.1/dagre-d3.min.js";
        document.body.appendChild(script);
        }
        })();
        </script-->
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
        '''


        obj = 'fst_' + uuid4().hex
        ret += f'<center><svg width="850" height="600" id="{obj}"><g/></svg></center>'
        ret += '''
        <script>
        requirejs(['d3', 'dagreD3'], function() {});
        (function render_d3() {
        var d3, dagreD3;
        try {
        d3 = require('d3');
        dagreD3 = require('dagreD3');
        } catch { setTimeout(render_d3, 50); return; } // requirejs is broken on external domains
        //alert("loaded");
        var g = new dagreD3.graphlib.Graph().setGraph({});

        '''

        # here we are actually going to read the states from the FST and generate nodes for them
        # in the output source code
        zero = self._make_weight('__FST_ZERO__')

        for sid in range(self.num_states):
            finalW = ''
            ww = self._fst.FinalWeight(sid)
            if '__FST_ZERO__' != ww:  # look at at the raw returned value to see if it is zero (unset)
                ww = self._make_weight(ww)
                if zero != ww:
                    finalW = f'\n({ww})'
            label = f'{sid}{finalW}'

            ret += f'g.setNode("state_{sid}", {{ label: {json.dumps(label)} , shape: "circle" }});\n'
            if finalW:
                # make the final states red
                ret += f'g.node("state_{sid}").style = "fill: #f77"; \n'

        if self._character_fst:
            make_chr = chr
        else:
            make_chr = str

        for sid in range(self.num_states):
            to = set()
            for arc in self.get_arcs(sid):
                if arc.nextstate == -1:
                    continue
                # if there are multiple arcs between a state, just draw one
                # otherwise the drawing system is going to have problems
                if arc.nextstate in to:
                    continue
                to.add(arc.nextstate)

                label = ''
                if arc.ilabel == 0:
                    label += '\u03B5'  # epsilon
                elif arc.ilabel == 32 and self._character_fst:
                    label += '(spc)'
                elif arc.ilabel < 32:
                    label += str(arc.ilabel)
                else:
                    label += make_chr(arc.ilabel)
                label += ':'
                if arc.olabel == 0:
                    label += '\u03B5'
                elif arc.olabel == 32 and self._character_fst:
                    label += '(spc)'
                elif arc.olabel < 32:
                    label += str(arc.olabel)
                else:
                    label += make_chr(arc.olabel)
                label += f'/{arc.weight}'

                ret += f'g.setEdge("state_{sid}", "state_{arc.nextstate}", {{ arrowhead: "vee", label: {json.dumps(label)} }});\n'

            # TODO: figure out ending weight and how to encode that in the graph?
            # atm this is just skipping the ending weight


        # make the start state green
        ret += f'g.node("state_{self.start_state}").style = "fill: #7f7"; \n'

        ret += f'var svg = d3.select("#{obj}"); \n'
        ret += '''
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
        '''
        return ret
