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
#include <fst/intersect.h>
#include <fst/union.h>
#include <fst/push.h>


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

template<uint64 S>
class FSTWeight {
private:
  FSTWeight(int32 g) : impl(), flags(g) {} // the static value constructor
public:
  using ReverseWeight = FSTWeight<S>;

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

  static const FSTWeight<S>& Zero() {
    static const FSTWeight<S> zero = FSTWeight<S>(isZero);
    return zero;
  }
  static const FSTWeight<S>& One() {
    static const FSTWeight<S> one = FSTWeight<S>(isOne);
    return one;
  }
  static const FSTWeight<S>& NoWeight() {
    static const FSTWeight<S> no_weight = FSTWeight<S>(isNoWeight);
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
  // this has to be a constexpr which means that we can't change it from python
  static constexpr uint64 Properties () {
    return S;
    //return kSemiring | kCommutative | kPath;
  }

  bool Member() const {
    if(isBuiltIn()) {
      return (flags & (isOne | isZero)) != 0;
    } else {
      py::object r = impl.attr("_member")();
      return r.cast<bool>();
    }
  }

  FSTWeight<S> Reverse() const {
    if(isBuiltIn()) {
      return *this;
    } else {
      py::object r = impl.attr("_reverse")();
      return FSTWeight<S>(r);
    }
  }

  FSTWeight<S> Quantize(float delta = kDelta) const {
    if(isBuiltIn()) {
      return *this;
    } else {
      py::object r = impl.attr("_quantize")(delta);
      return FSTWeight<S>(r);
    }
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
    if(isBuiltIn()) {
      return flags;
    } else {
      py::object r = impl.attr("__hash__")();
      return r.cast<size_t>();
    }
  }

  FSTWeight<S>& operator=(const FSTWeight<S>& other) {
    impl = other.impl;
    flags = other.flags;
  }

  py::object PythonObject() const {
    if(flags == isSet) {
      return impl;
    } else if(flags == isOne) {
      return py::cast(1);
    } else if(flags == isZero) {
      return py::cast(0);
    } else {
      return py::cast("__FST_INVALID__");
    }
  }

  py::object buildObject(py::object &other) const {
    // in the case that we are just wrapping a count, we still want to construct some object
    // that can be used to track the semiring operations
    py::object ret = other.attr("zero")(); // get the zero element
    py::object v = other.attr("one")();
    uint64 i = 0;
    while(i) {
      if(i & 0x1) {
        ret = ret.attr("__add__")(v);
      }
      i >>= 1;
      if(!i) break;
      v = v.attr("__add__")(v);
    }
    return ret;
  }

  virtual ~FSTWeight<S>() {
    //cout << "delete weight\n";
  }

};


template<uint64 S>
inline FSTWeight<S> Plus(const FSTWeight<S> &w1, const FSTWeight<S> &w2) {
  // these identity elements have to be handled specially
  // as they are the same for all semirings that are defined in python and thus contain no value
  if(w1.flags == FSTWeight<S>::isZero) {
    return w2;
  }
  if(w2.flags == FSTWeight<S>::isZero) {
    return w1;
  }
  if(w1.flags == FSTWeight<S>::isOne || w2.flags == FSTWeight<S>::isOne) {
    throw fsterror("Trying to add with the static semiring one");
  }
  return FSTWeight<S>(w1.impl.attr("__add__")(w2.impl));
}

template<uint64 S>
inline FSTWeight<S> Times(const FSTWeight<S> &w1, const FSTWeight<S> &w2) {
  if(w1.flags == FSTWeight<S>::isOne) {
    return w2;
  }
  if(w2.flags == FSTWeight<S>::isOne) {
    return w1;
  }
  if(w1.flags == FSTWeight<S>::isZero || w2.flags == FSTWeight<S>::isZero) {
    return FSTWeight<S>::Zero();
  }
  return FSTWeight<S>(w1.impl.attr("__mul__")(w2.impl));
}

template<uint64 S>
inline FSTWeight<S> Divide(const FSTWeight<S> &w1, const FSTWeight<S> &w2) {
  if(w2.flags == FSTWeight<S>::isOne) {
    return w1;
  }
  // so this could construct a 1 element or try and use rdiv
  return FSTWeight<S>(w1.impl.attr("__div__")(w2.impl));
}

template<uint64 S>
inline FSTWeight<S> Divide(const FSTWeight<S> &w1, const FSTWeight<S> &w2, const DivideType &type) {
  // TODO: there are left and right divide types that could be passed to python???
  // that might be important depending on which ring we are in
  return Divide(w1, w2);
}

template<uint64 S>
inline FSTWeight<S> Power(const FSTWeight<S> &w1, int n) {
  if(w1.isBuiltIn()) {
    return w1;
  } else {
    return FSTWeight<S>(w1.impl.attr("__pow__")(n));
  }
}

template<uint64 S>
inline bool operator==(FSTWeight<S> const &w1, FSTWeight<S> const &w2) {
  if(w1.isBuiltIn()) {
    return w1.flags == w2.flags;  // if this is a built in value then it will have the same pointer
  } else {
    if(w2.isBuiltIn())
      return false;
    py::object r = w1.impl.attr("__eq__")(w2.impl);
    return r.cast<bool>();
  }
}

template<uint64 S>
inline bool operator!=(FSTWeight<S> const &w1, FSTWeight<S> const &w2) {
  return !(w1 == w2);
}

template<uint64 S>
inline bool ApproxEqual(FSTWeight<S> const &w1, FSTWeight<S> const &w2, const float &delta) {
  if(w1.isBuiltIn()) {
    return w1.flags == w2.flags;  // if this is a built in value then it will have the same pointer
  } else {
    if(w2.isBuiltIn())
      return false;
    py::object r = w1.impl.attr("_approx_eq")(w2.impl, delta);
    return r.cast<bool>();
  }
}

template<uint64 S>
inline bool ApproxEqual(FSTWeight<S> const &w1, FSTWeight<S> const &w2, const float &delta, const bool &error) {
  return ApproxEqual(w1, w2, delta);
}

template<uint64 S>
inline std::ostream &operator<<(std::ostream &os, const FSTWeight<S> &w) {
  return w.Write(os);
}

template<uint64 S>
inline std::istream &operator>>(std::istream &is, const FSTWeight<S> &w) {
  throw fsterror("python weights do not support loading");
}

template<uint64 S>
using PyArc = ArcTpl<FSTWeight<S>>;
template<uint64 S>
using PyFST = VectorFst<PyArc<S>, VectorState<PyArc<S> > >;

// PyFST<S>* create_fst() {
//   return new PyFST();
// }

template<uint64 S>
void add_arc(PyFST<S> &self, int64 from, int64 to,
             int64 input_label, int64 output_label, py::object weight) {
  // TODO: check if the weight is the correct python instance

  if(!check_is_weight(weight)) {
    throw fsterror("weight is missing required method");
  }

  FSTWeight<S> w1(weight);
  PyArc<S> a(input_label, output_label, w1, to);
  //cout << "before add\n";
  self.AddArc(from, a);
  //cout << "after add\n";
}

template<uint64 S>
void set_final(PyFST<S> &self, int64 state, py::object weight) {
  if(!check_is_weight(weight)) {
    throw fsterror("weight is missing required method");
  }

  FSTWeight<S> w1(weight);
  self.SetFinal(state, w1);
}

template<uint64 S>
py::object final_weight(PyFST<S> &self, int64 state) {
  FSTWeight<S> finalW = self.Final(state);
  py::object r = finalW.PythonObject();
  // if(finalW.isBuiltIn()) {
  //   if(finalW.flags == FSTWeight<S>::isZero) {
  //     return py::cast(0);
  //   } else if(finalW.flags == FSTWeight<S>::isOne) {
  //     return py::cast(1);
  //   } else {
  //     // invalid
  //     return py::cast("");
  //   }
  // }

  // assert(finalW.flags == FSTWeight<S>::isSet);

  // py::object r =  finalW.impl; // this should still be holding onto the handle

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

template<uint64 S>
void define_class(pybind11::module &m, const char *name) {

  py::class_<PyFST<S> >(m, name)
    .def(py::init<>())

#define d(name) .def("_" #name, & PyFST<S> :: name)
    .def("_AddArc", &add_arc<S>) // keep the weight alive when added
    d(AddState)
    // there are more than one method with this name but different type signatures
    .def("_DeleteArcs", [](PyFST<S> *m, int64 v) { if(m) m->DeleteArcs(v); })

    .def("_DeleteStates", [](PyFST<S> *m) { if(m) m->DeleteStates(); })

    d(NumStates)
    d(ReserveArcs)
    d(ReserveStates)

    .def("_SetFinal", &set_final<S>)
    d(SetStart)

    d(NumArcs)
    d(Start)

    .def("_FinalWeight", &final_weight<S>)
#undef d



    // compare two FST methods
    .def("_Isomorphic", [](const PyFST<S> &a, const PyFST<S> &b, float delta) {
        return Isomorphic(a, b, delta);
      })

    // methods that will generate a new FST or
    .def("_Concat", [](const PyFST<S> &a, PyFST<S> &b) {
        PyFST<S> *ret = b.Copy();
        Concat(a, ret);
        return ret;
      })


    .def("_Compose", [](const PyFST<S> &a, const PyFST<S> &b) {
        PyFST<S> *ret = new PyFST<S>();
        Compose(a,b, ret);
        return ret;
      })

    .def("_Determinize", [](const PyFST<S> &a, float delta, py::object weight) {
        FSTWeight<S> weight_threshold(weight);

        PyFST<S> *ret = new PyFST<S>();

        DeterminizeOptions<PyArc<S> > ops(delta, weight_threshold);
        Determinize(a, ret, ops);
        return ret;
      })

    .def("_Project", [](const PyFST<S> &a, int type) {
        PyFST<S> *ret = new PyFST<S>();

        ProjectType typ = type ? PROJECT_INPUT : PROJECT_OUTPUT;
        Project(a, ret, typ);
        return ret;
      })

    .def("_Difference", [](const PyFST<S> &a, const PyFST<S> &b) {
        PyFST<S> *ret = new PyFST<S>();
        Difference(a, b, ret);
        return ret;
      })

    .def("_Invert", [](const PyFST<S> &a) {
        PyFST<S> *ret = a.Copy();
        Invert(ret);
        return ret;
      })

    .def("_Prune", [](const PyFST<S> &a, py::object weight) {
        FSTWeight<S> weight_threshold(weight);
        PyFST<S> *ret = new PyFST<S>();
        Prune(a, ret, weight_threshold);
        return ret;
      })

    .def("_Intersect", [](const PyFST<S> &a, const PyFST<S> &b) {
        PyFST<S> *ret = new PyFST<S>();
        Intersect(a, b, ret);
        return ret;
      })

    .def("_Union", [](const PyFST<S> &a, const PyFST<S> &b) {
        PyFST<S> *ret = a.Copy();
        Union(ret, b);
        return ret;
      })

    .def("_Push", [](const PyFST<S> &a, int mode) {
        PyFST<S> *ret = new PyFST<S>();
        if(mode == 0) {
          Push<PyArc<S>, REWEIGHT_TO_INITIAL>(a, ret, kPushWeights);
        } else {
          Push<PyArc<S>, REWEIGHT_TO_FINAL>(a, ret, kPushWeights);
        }
        return ret;
      })

    .def("_RandomPath", [](const PyFST<S> &a) {
        // TODO: have something that is going to call back into the
        // python process and use that to generate the weights that are along the path

        assert(false); // TODO:
        return false;
      })

    .def("_ArcList", [](const PyFST<S> &a, int64 state) {
        // input label, output label, to state, weight
        vector<py::tuple> ret;

        ArcIterator<PyFST<S> > iter(a, state);
        while(!iter.Done()) {
          auto &v = iter.Value();
          //assert(v.weight.flags == FSTWeight<S>::isSet);
          // we are just returning the pure python object, so if it gets held
          // that will still be ok
          const FSTWeight<S> &w = v.weight;
          py::object oo = w.PythonObject();
          ret.push_back(make_tuple(v.ilabel, v.olabel, v.nextstate, oo));
          iter.Next();
        }

        FSTWeight<S> finalW = a.Final(state);
        if(finalW.flags == FSTWeight<S>::isSet) {
          // then there is something here that we should push I guess?
          ret.push_back(make_tuple(0, 0, -1, finalW.impl));
        }

        return ret;
      })

    .def("_toString", [](const PyFST<S> &a) {
        ostringstream out;
        fst::script::PrintFst(a, out);
        return out.str();
      });

    ;
}

PYBIND11_MODULE(openfst_wrapper_backend, m) {
  m.doc() = "Backing wrapper for OpenFST";

  define_class<kSemiring | kCommutative>(m, "FSTBase");
  define_class<kSemiring | kCommutative | kPath | kIdempotent>(m, "FSTPath");

}
