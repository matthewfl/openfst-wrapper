#include <memory>
#include <string>
#include <sstream>
#include <random>
#include <vector>
#include <tuple>

#undef NDEBUG
#include <assert.h>


#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <fst/fst.h>
#include <fst/mutable-fst.h>
#include <fst/vector-fst.h>
#include <fst/arc.h>

// fst operations that we are supporting
#include <fst/isomorphic.h>
#include <fst/concat.h>
#include <fst/compose.h>
#include <fst/determinize.h>
#include <fst/project.h>
#include <fst/difference.h>
#include <fst/invert.h>
#include <fst/prune.h>


#include <fst/script/print.h>


using namespace std;
using namespace fst;
namespace py = pybind11;

// this error should just become a runtime error in python land
class fsterror : public exception {
private:
  const char* error;
public:
  fsterror(const char *e): error(e) {}
  virtual const char* what() { return error; }
};


bool check_is_weight(py::object &weight) {
  return
    !weight.is_none() &&
    hasattr(weight, "__add__") &&
    hasattr(weight, "__mul__") &&
    hasattr(weight, "__div__") &&
    hasattr(weight, "__pow__") &&
    hasattr(weight, "__hash__") &&
    hasattr(weight, "__eq__") &&
    hasattr(weight, "__str__") &&
    hasattr(weight, "_member") &&
    hasattr(weight, "_quantize") &&
    hasattr(weight, "_reverse");
  // zero & one ??
}

class FSTWeight {
private:
  FSTWeight(int32 g) : impl(), flags(g) {} // the static value constructor
public:

  using ReverseWeight = FSTWeight;

  enum {
    isZero = 0x1,
    isOne = 0x2,
    isNoWeight = 0x4,
    isSet = 0x8
  };

  // the python object that we are wrapping
  int16_t flags;
  py::object impl;

  FSTWeight() : impl() {
    flags = isNoWeight;
  }

  FSTWeight(py::object i) : impl(i) {
    if(!check_is_weight(impl))
      throw fsterror("Value does not implement Weight interface");
    flags = isSet;
  }

  static const FSTWeight& Zero() {
    static const FSTWeight zero = FSTWeight(isZero);
    return zero;
  }
  static const FSTWeight& One() {
    static const FSTWeight one = FSTWeight(isOne);
    return one;
  }
  static const FSTWeight& NoWeight() {
    static const FSTWeight no_weight = FSTWeight(isNoWeight);
    return no_weight;
  }

  bool isBuiltIn() const {
    assert(flags != 0);
    return (flags & (isZero | isOne | isNoWeight)) != 0;
  }

  static const string& Type() {
    // TODO: this is wrong
    // we should register our own type of weight, but for some reason that isn't working????
    static const string type("python"); //tropical");
    return type;
  }

  // this might not be what the user wants to use for this???
  static constexpr uint64 Properties () { return kSemiring | kCommutative; }

  bool Member() const {
    py::object r = impl.attr("_member")();
    return r.cast<bool>();
  }

  FSTWeight Reverse() const {
    py::object r = impl.attr("_reverse")();
    return FSTWeight(r);
  }

  FSTWeight Quantize(float delta = kDelta) const {
    py::object r = impl.attr("_quantize")(delta);
    return FSTWeight(r);
  }

  std::istream &Read(std::istream &strm) const { throw fsterror("not implemented"); }
  std::ostream &Write(std::ostream &os) const {
    if(isBuiltIn()) {
      if(flags == isOne) {
        return os << "(1)" ;
      } else if(flags == isZero) {
        return os << "(0)";
      } else {
        return os << "(invalid)";
      }
    } else {
      py::object s = impl.attr("__str__")();
      return os << s.cast<string>();
    }
  }

  size_t Hash() const {
    py::object r = impl.attr("__hash__")();
    return r.cast<size_t>();
  }

  FSTWeight& operator=(const FSTWeight& other) {
    impl = other.impl;
    flags = other.flags;
  }

  virtual ~FSTWeight() {
    //cout << "delete weight\n";
  }

};

inline FSTWeight Plus(const FSTWeight &w1, const FSTWeight &w2) {
  // these identity elements have to be handled specially
  // as they are the same for all semirings that are defined in python and thus contain no value
  if(w1.flags == FSTWeight::isZero) {
    return w2;
  }
  if(w2.flags == FSTWeight::isZero) {
    return w1;
  }
  if(w1.flags == FSTWeight::isOne || w2.flags == FSTWeight::isOne) {
    throw fsterror("Trying to add with the static semiring one");
  }
  return FSTWeight(w1.impl.attr("__add__")(w2.impl));
}

inline FSTWeight Times(const FSTWeight &w1, const FSTWeight &w2) {
  if(w1.flags == FSTWeight::isOne) {
    return w2;
  }
  if(w2.flags == FSTWeight::isOne) {
    return w1;
  }
  if(w1.flags == FSTWeight::isZero || w2.flags == FSTWeight::isZero) {
    return FSTWeight::Zero();
  }
  return FSTWeight(w1.impl.attr("__mul__")(w2.impl));
}

inline FSTWeight Divide(const FSTWeight &w1, const FSTWeight &w2) {
  if(w2.flags == FSTWeight::isOne) {
    return w1;
  }
  // so this could construct a 1 element or try and use rdiv
  return FSTWeight(w1.impl.attr("__div__")(w2.impl));
}

inline FSTWeight Divide(const FSTWeight &w1, const FSTWeight &w2, const DivideType &type) {
  // TODO: there are left and right divide types that could be passed to python???
  // that might be important depending on which ring we are in
  return Divide(w1, w2);
}

inline FSTWeight Power(const FSTWeight &w1, int n) {
  return FSTWeight(w1.impl.attr("__pow__")(n));
}

inline bool operator==(FSTWeight const &w1, FSTWeight const &w2) {
  if(w1.isBuiltIn()) {
    return w1.flags == w2.flags;  // if this is a built in value then it will have the same pointer
  } else {
    if(w2.isBuiltIn())
      return false;
    py::object r = w1.impl.attr("__eq__")(w2.impl);
    return r.cast<bool>();
  }
}

inline bool operator!=(FSTWeight const &w1, FSTWeight const &w2) {
  return !(w1 == w2);
}

inline bool ApproxEqual(FSTWeight const &w1, FSTWeight const &w2, const float &delta) {
  if(w1.isBuiltIn()) {
    return w1.flags == w2.flags;  // if this is a built in value then it will have the same pointer
  } else {
    if(w2.isBuiltIn())
      return false;
    py::object r = w1.impl.attr("_approx_eq")(w2.impl, delta);
    return r.cast<bool>();
  }
}

inline bool ApproxEqual(FSTWeight const &w1, FSTWeight const &w2, const float &delta, const bool &error) {
  return ApproxEqual(w1, w2, delta);
}

std::ostream &operator<<(std::ostream &os, const FSTWeight &w) {
  return w.Write(os);
}

std::istream &operator>>(std::istream &is, const FSTWeight &w) {
  throw fsterror("python weights do not support loading");
}


using PyArc = ArcTpl<FSTWeight>;
using PyFST = VectorFst<PyArc, VectorState<PyArc> >;

PyFST* create_fst() {
  return new PyFST();
}


void add_arc(PyFST &self, int64 from, int64 to,
             int64 input_label, int64 output_label, py::object weight) {
  // TODO: check if the weight is the correct python instance

  if(!check_is_weight(weight)) {
    throw fsterror("weight is missing required method");
  }

  FSTWeight w1(weight);
  PyArc a(input_label, output_label, w1, to);
  //cout << "before add\n";
  self.AddArc(from, a);
  //cout << "after add\n";
}

void set_final(PyFST &self, int64 state, py::object weight) {
  if(!check_is_weight(weight)) {
    throw fsterror("weight is missing required method");
  }

  FSTWeight w1(weight);
  self.SetFinal(state, w1);
}

py::object final_weight(PyFST &self, int64 state) {
  FSTWeight finalW = self.Final(state);
  if(finalW.isBuiltIn()) {
    if(finalW.flags == FSTWeight::isZero) {
      return py::cast(0);
    } else if(finalW.flags == FSTWeight::isOne) {
      return py::cast(1);
    } else {
      // invalid
      return py::cast("__FST_INVALID__");
    }
  }

  assert(finalW.flags == FSTWeight::isSet);

  py::object r =  finalW.impl; // this should still be holding onto the handle

  return r;
}
/*

class PythonArcSelector {
public:
  using StateId = typename PyFST::StateId;
  using Weight = typename PyFST::Weight;

  PythonArcSelector() {}

  explicit PythonArcSelector(uint64 seed) : rand_(seed) {}

  size_t operator()(const Fst<PyFST> &fst, StateId s) const {
    const auto n = fst.NumArcs(s) + (fst.Final(s) != Weight::Zero());
    //vector<float> scores;  // the unweighted scores for the arcs

    // TODO:
    return static_cast<size_t>(
        std::uniform_int_distribution<>(0, n - 1)(rand_));
  }

 private:
  mutable std::mt19937_64 rand_;

};
*/

PYBIND11_MODULE(openfst_wrapper_backend, m) {
  m.doc() = "Backing wrapper for OpenFST";

  py::class_<PyFST>(m, "FSTBase")
    .def(py::init<>(&create_fst))
#define d(name) .def("_" #name, &PyFST:: name)
    .def("_AddArc", &add_arc) // keep the weight alive when added
    d(AddState)
    // there are more than one method with this name but different type signatures
    .def("_DeleteArcs", [](PyFST *m, int64 v) { if(m) m->DeleteArcs(v); })
    .def("_DeleteStates", [](PyFST *m) { if(m) m->DeleteStates(); })

    d(NumStates)
    d(ReserveArcs)
    d(ReserveStates)

    .def("_SetFinal", &set_final)
    d(SetStart)

    d(NumArcs)
    d(Start)

    .def("_FinalWeight", &final_weight)
#undef d


    // compare two FST methods
    .def("_Isomorphic", [](PyFST &a, PyFST  &b, float delta) {
        return Isomorphic(a, b, delta);
      })

    // methods that will generate a new FST or
    .def("_Concat", [](const PyFST &a, PyFST &b) {
        PyFST *ret = b.Copy();
        Concat(a, ret);
        return ret;
      })


    .def("_Compose", [](const PyFST &a, const PyFST &b) {
        PyFST *ret = create_fst();
        Compose(a,b, ret);
        return ret;
      })

    .def("_Determinize", [](const PyFST &a, float delta, py::object weight) {
        FSTWeight weight_threshold(weight);

        PyFST *ret = create_fst();

        DeterminizeOptions<PyArc> ops(delta, weight_threshold);
        Determinize(a, ret, ops);
        return ret;
      })

    .def("_Project", [](const PyFST &a, int type) {
        PyFST *ret = create_fst();

        ProjectType typ = type ? PROJECT_INPUT : PROJECT_OUTPUT;
        Project(a, ret, typ);
        return ret;
      })

    .def("_Difference", [](const PyFST &a, const PyFST &b) {
        PyFST *ret = create_fst();
        Difference(a, b, ret);
        return ret;
      })

    .def("_Invert", [](const PyFST &a) {
        PyFST *ret = a.Copy();
        Invert(ret);
        return ret;
      })

    .def("_Prune", [](const PyFST &a, py::object weight) {
        FSTWeight weight_threshold(weight);
        PyFST *ret = create_fst();
        Prune(a, ret, weight_threshold);
        return ret;
      })

    .def("_RandomPath", [](const PyFST &a) {
        // TODO: have something that is going to call back into the
        // python process and use that to generate the weights that are along the path

        assert(false); // TODO:
        return false;
      })

    .def("_ArcList", [](const PyFST &a, int64 state) {
        // input label, output label, to state, weight
        vector<py::tuple> ret;

        ArcIterator<PyFST> iter(a, state);
        while(!iter.Done()) {
          auto &v = iter.Value();
          assert(v.weight.flags == FSTWeight::isSet);
          // we are just returning the pure python object, so if it gets held
          // that will still be ok
          ret.push_back(make_tuple(v.ilabel, v.olabel, v.nextstate, v.weight.impl));
          iter.Next();
        }

        FSTWeight finalW = a.Final(state);
        if(finalW.flags == FSTWeight::isSet) {
          // then there is something here that we should push I guess?
          ret.push_back(make_tuple(0, 0, -1, finalW.impl));
        }

        return ret;
      })

    .def("_toString", [](const PyFST &a) {
        cout << "asdf\n";
        ostringstream out;
        fst::script::PrintFst(a, out);
        return out.str();
      })


    ;

}
