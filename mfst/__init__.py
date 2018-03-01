import openfst_wrapper_backend as _backend
from collections import namedtuple as _namedtuple

ArcType = _namedtuple('Arc', ['ilabel', 'olabel', 'nextstate', 'weight'])


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


class ValueWeight(WeightBase):

    def __init__(self, value=0):
        # This needs to happen otherwise the backing classes might not get set up properly
        super().__init__()
        self._value = value  # the value

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
        return self._create(self._value + other._value)

    def __mul__(self, other):
        assert type(other) is type(self)
        return self._create(self._value * other._value)

    def __div__(self, other):
        assert type(other) is type(self)
        return self._create(self._value / other._value)

    def __pow__(self, n):
        return self._create(self._value ** n)

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
        return isinstance(other, ValueWeight) and self._value == other._value


class FST(object):
    """
    Wraps a mutable FST class
    """

    def __init__(self, fst_type='base', weight_class=None, _fst=None):
        if _fst:
            self._fst = _fst
        else:
            self._fst = {
                # these classes are defined in c++ to wrap openfst
                'base': _backend.FSTBase,
                'path': _backend.FSTPath
            }[fst_type]()

        if not weight_class:
            self._weight_class = ValueWeight
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

    def _wrap_fst(self, fst):
        return type(self)(weight_class=self._weight_class, _fst=fst)

    def create_from_string(self, string):
        ret = FST(weight_class=self._weight_class, _fst=type(self._fst)())
        last = ret.add_state()
        ret.start_state = last
        for s in string:
            state = ret.add_state()
            ret.add_arc(last, state, output_label=ord(s))
            last = state
        if last:
            ret.set_final_weight(last)
        return ret

    def get_string(self):
        """
        Returns the string representation in the case that there is only a single path in the fst
        """
        state = self.start_state
        seen = set()
        ret = []
        while state != -1:
            edges = list(self.get_arcs(state))
            if len(edges) != 1:
                raise RuntimeError("FST does not contain exactly one path")
            l = edges[0].olabel
            if l != 0:  # the epsilon state
                ret.append(chr(l))
            if edges[0].nextstate in seen:
                raise RuntimeError("FST contains cycle")
            seen.add(state)
            state = edges[0].nextstate
        return ''.join(ret)

    @property
    def num_states(self):
        """
        Return the number of states currently set on the fst
        """
        return self._fst._NumStates()

    def num_arcs(self, state):
        """
        Return the number of arcs in the fst
        """
        return self._fst._NumArcs()

    @property
    def start_state(self):
        """
        Return the state id of the starting state
        """
        return self._fst._Start()

    @start_state.setter
    def start_state(self, state):
        """
        Mark a state as the start state
        """
        assert (state >= 0 and state < self.num_states), "Invalid state id"
        return self._fst._SetStart(state)

    def set_start_state(self, state):
        self.start_state = state

    def add_state(self):
        """
        Add a new state to the FST
        Return this state's id
        """
        return self._fst._AddState()

    def add_arc(self, from_state, to_state,
                weight=1, input_label=0, output_label=0):
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
        if isinstance(output_label, str):
            assert len(input_label) == 1, "FST string labels can only be a single character"
            output_label = ord(output_label)
        return self._fst._AddArc(from_state, to_state, input_label, output_label,
                                 self._make_weight(weight))

    def delete_arcs(self, state):
        """
        Delete all arcs coming out of state
        """
        assert (state >= 0 and state < self.num_states), "Invalid state id"
        return self._fst._DeleteArcs(state)

    def delete_states(self):
        """
        Delete all states in the FST
        """
        return self._fst._DeleteStates()

    def set_final_weight(self, state, weight=1):
        """
        Set the weight that this state transisions to the final state
        """
        assert (state >= 0 and state < self.num_states), "Invalid state id"
        return self._fst._SetFinal(state, self._make_weight(weight))

    def get_final_weight(self, state):
        """
        Get the weight of transistioning to the final state
        """
        assert (state >= 0 and state < self.num_states), "Invalid state id"
        return self._make_weight(self._fst._FinalWeight(state))

    def get_arcs(self, state):
        """
        Return the arcs coming out of some state
        """
        assert (state >= 0 and state < self.num_states), "Invalid state id"
        for arc in self._fst._ArcList(state):
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
        return self._fst._Isomorphic(other._fst, delta)

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
        return self._wrap_fst(self._fst._Concat(other._fst))

    def compose(self, other):
        """
        This operation computes the composition of two transducers. If A transduces
        string x to y with weight a and B transduces y to z with weight b, then their
        composition transduces string x to z with weight a (times) b.

        http://www.openfst.org/twiki/bin/view/FST/ComposeDoc
        """
        assert isinstance(other, FST)
        return self._wrap_fst(self._fst._Compose(other._fst))

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

        return self._wrap_fst(self._fst._Determinize(delta, self._make_weight(weight_threshold)))

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
        return self._wrap_fst(self._fst._Project(t))

    def difference(self, other):
        """
        This operation computes the difference between two FSAs. Only strings that
        are in the first automaton but not in second are retained in the
        result.

        http://www.openfst.org/twiki/bin/view/FST/DifferenceDoc
        """
        assert isinstance(other, FST)
        assert self._wrap_fst(self._fst._Difference(other))

    def invert(self):
        """
        This operation inverts the transduction corresponding to an FST by
        exchanging the FST's input and output labels.

        http://www.openfst.org/twiki/bin/view/FST/InvertDoc
        """
        return self._wrap_fst(self._fst._Invert())

    def prune(self, weight):
        """
        This operation deletes states and arcs in the input FST that do not belong
        to a successful path whose weight is no more (w.r.t the natural the
        natural semiring order) than the threshold t otimes the weight of the
        shortest path in the input FST.

        http://www.openfst.org/twiki/bin/view/FST/PruneDoc
        """
        return self._wrap_fst(self._fst._Prune(self._make_weight(weight)))

    def union(self, other):

        pass

    def intersect(self, other):
        pass


    def random_path(self, arc_selector=None):
        """


        http://www.openfst.org/twiki/bin/view/FST/RandGenDoc
        """
        pass

    def __str__(self):
        if self.num_states < 10:
            # if the FST is small enough that we might want to print the whole thing in the string
            return 'FST {\n' + self._fst._toString() + '\n}'
        else:
            return 'FST(num_states={})'.format(self.num_states)

    def __repr__(self):
        return str(self)

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
        ret += f'<center><svg width=700 height="600" id="{obj}"><g/></svg></center>'
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

        for sid in range(self.num_states):
            ret += f'g.setNode("state_{sid}", {{ label: "{sid}" , shape: "circle" }});\n'
        for sid in range(self.num_states):
            for arc in self.get_arcs(sid):
                if arc.nextstate == -1:
                    continue
                label = ''
                if arc.ilabel == 0:
                    label += '\u03B5'  # epsilon
                elif arc.ilabel < 33:
                    label += str(arc.ilabel)
                else:
                    label += chr(arc.ilabel)
                label += ':'
                if arc.olabel == 0:
                    label += '\u03B5'
                elif arc.olabel < 33:
                    label += str(arc.olabel)
                else:
                    label += chr(arc.olabel)
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
        var zoom = d3.zoom().on("zoom", function() {
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

        svg.attr('height', g.graph().height * initialScale + 140);
        })();
        </script>
        '''
        return ret
