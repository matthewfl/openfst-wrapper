/**
   Written by Matthew Francis-Landau (2018)

   Wrapper for OpenFst that supports defining custom semirings in python
   and drawing FSTs in ipython notebooks
*/

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

#ifdef __GNUC__
// openfst causes this warning to go off a lot
#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma GCC diagnostic ignored "-Wreorder"
#endif

#include <fst/fst.h>
#include <fst/mutable-fst.h>
#include <fst/vector-fst.h>
#include <fst/arc.h>
#include <fst/util.h>

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
#include <fst/arcsort.h>
#include <fst/minimize.h>
#include <fst/shortest-path.h>
#include <fst/rmepsilon.h>
#include <fst/randgen.h>
#include <fst/shortest-distance.h>
#include <fst/topsort.h>
#include <fst/disambiguate.h>
#include <fst/verify.h>
#include <fst/closure.h>
#include <fst/reverse.h>

#include <fst/script/print.h>


using namespace std;
using namespace fst;
namespace py = pybind11;

// this error should just become a runtime error in python land
class fsterror : public std::exception {
private:
  string str;
  const char* error;
public:
  fsterror(const char *e): error(e) {}
  fsterror(string s) : str(s), error(str.c_str()) { }
  virtual const char* what() { return error; }
};

class ErrorCatcher {
  // hijack the std::cerr buffer and get any errors that openfst produces.
  // this will then be thrown to the python process
  std::streambuf *backup;
  ostringstream buff;
public:
  ErrorCatcher() {
    backup = cerr.rdbuf();
    cerr.rdbuf(buff.rdbuf());
  }
  ~ErrorCatcher() noexcept(false) {
    // reset
    cerr.rdbuf(backup);
    string e = buff.str();
    if(!e.empty() && std::current_exception() == nullptr &&
       !std::uncaught_exception() && PyErr_Occurred() == nullptr) {
      PyErr_SetString(PyExc_RuntimeError, e.c_str());
      //throw fsterror(e);  // TODO: try this again to see if this will work
      throw py::error_already_set();
    }
  }
};


bool check_is_weight(py::object &weight) {
  return
    !weight.is_none() &&
    hasattr(weight, "__add__") &&
    hasattr(weight, "__mul__") &&
    hasattr(weight, "__div__") &&
    hasattr(weight, "__pow__") &&
    // need to check that this isn't null
    !getattr(weight, "__hash__").is_none() &&
    hasattr(weight, "__eq__") &&
    hasattr(weight, "openfst_str") &&
    hasattr(weight, "member") &&
    hasattr(weight, "quantize") &&
    hasattr(weight, "reverse") &&
    hasattr(weight, "approx_eq") &&
    hasattr(weight, "sampling_weight") &&
    hasattr(weight, "one") &&
    hasattr(weight, "zero");
}


// this is SUCH a HACK
class StaticPythonWeights {
private:
  //py::handle old;
  py::handle semiring_class;
  StaticPythonWeights *old;

  // // TODO maybe remove?
  // py::object one_cache;
  // py::object zero_cache;

  static StaticPythonWeights *active;

public:
  StaticPythonWeights(py::handle semiring) {
    if(!hasattr(semiring, "zero") || !hasattr(semiring, "one")) {
      throw fsterror("Semiring does not have zero and one method to construct new instances");
    }
    //old = semiring_class;
    semiring_class = semiring;
    old = active;
    assert(active == nullptr); // TODO allow for nested static weight classes
    active = this;
  }
  ~StaticPythonWeights() {
    assert(active == this);
    active = old;
    semiring_class = nullptr;
    // one_cache.release();
    // zero_cache.release();
  }

  static bool contains() {
    return active != nullptr && active->semiring_class.ptr() != nullptr;
  }

  static py::object One() {
    // should have already checked
    //if(active->one_cache.ptr() == nullptr) {
    return active->semiring_class.attr("one");
    // active->one_cache =
    //   //}
    // return active->one_cache;
  }

  static py::object Zero() {
    //if(active->zero_cache.ptr() == nullptr) {
    return active->semiring_class.attr("zero");
    // active->zero_cache =
    //   //}
    // return active->zero_cache;
  }
};

StaticPythonWeights* StaticPythonWeights::active = nullptr;


template<uint64 S>
class FSTWeight final {
private:
  FSTWeight(int32 g, bool __) : flags(g), impl() {} // the static value constructor
public:
  using ReverseWeight = FSTWeight<S>;

  enum {
    isZero = 0x1,
    isOne = 0x2,
    isNoWeight = 0x4,
    isSet = 0x8,
  };

  // the python object that we are wrapping
  py::object impl;

  int16_t flags;


  FSTWeight() : flags(isZero), impl() {}


  FSTWeight(py::object i) : flags(isSet), impl(i) {
    if(!check_is_weight(impl))
      throw fsterror("Value does not implement Weight interface");
  }

  // these need to be "true" static values, as these are cached by OpenFst as static variables in various places
  static FSTWeight<S> Zero() {
    static const FSTWeight<S> zero = FSTWeight<S>(isZero, true);
    return zero;
  }
  static FSTWeight<S> One() {
    static const FSTWeight<S> one = FSTWeight<S>(isOne, true);
    return one;
  }
  static const FSTWeight<S>& NoWeight() {
    static const FSTWeight<S> no_weight = FSTWeight<S>(isNoWeight, true);
    return no_weight;
  }

  bool isBuiltIn() const {
    assert(flags != 0);
    return (flags & (isZero | isOne | isNoWeight)) != 0;
  }

  static const string& Type() {
    static const string type("python");
    return type;
  }

  // controlled by the template parameter which can be somewhat choosen via
  // different classes from python
  // see define_class
  static constexpr uint64 Properties () {
    return S;
  }

  bool Member() const {
    if(isBuiltIn()) {
      return (flags & (isOne | isZero)) != 0;
    } else {
      assert(flags == isSet);
      py::object r = impl.attr("member")();
      return r.cast<bool>();
    }
  }

  FSTWeight<S> Reverse() const {
    if(isBuiltIn()) {
      // TODO: this should print a warning or something
      return *this;
    } else {
      py::object r = impl.attr("reverse")();
      return FSTWeight<S>(r);
    }
  }

  FSTWeight<S> Quantize(float delta = kDelta) const {
    if(isBuiltIn()) {
      return *this;
    } else {
      py::object r = impl.attr("quantize")(delta);
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
      // it appears that openfst uses this string inside of its determinize and disambiguate
      // if there are minor non matching floating point errors then openfst breaks
      py::object s = impl.attr("openfst_str")();
      return os << s.cast<string>();
    }
  }

  size_t Hash() const {
    if(isBuiltIn()) {
      return flags;
    } else {
      return py::hash(impl);
    }
  }

  FSTWeight<S>& operator=(const FSTWeight<S>& other) {
    impl = other.impl;
    flags = other.flags;
    return *this;
  }

  py::object PythonObject() const {
    if(flags == isSet) {
      return impl;
    } else if(flags == isOne) {
      if(StaticPythonWeights::contains()) {
        return StaticPythonWeights::One();
      } else {
        return py::cast("__FST_ONE__");
      }
    } else if(flags == isZero) {
      if(StaticPythonWeights::contains()) {
        return StaticPythonWeights::Zero();
      } else {
        return py::cast("__FST_ZERO__");
      }
    } else {
      return py::cast("__FST_INVALID__");
    }
  }

  py::object buildObject(const py::object &other) const {
    // that can be used to track the semiring operations
    if(flags == isSet) {
      return impl;
    } else if(flags == isOne) {
      return other.attr("one");
    } else if(flags == isZero) {
      return other.attr("zero");
    } else {
      assert(flags == isNoWeight);
      // unsure what could be done here
      return py::cast("__FST_INVALID__");
    }
  }

  ~FSTWeight<S>() {
    //cout << "delete weight\n";
  }

};


template<uint64 S>
FSTWeight<S> Plus(const FSTWeight<S> &w1, const FSTWeight<S> &w2) {
  // these identity elements have to be handled specially
  // as they are the same for all semirings that are defined in python and thus contain no value
  if(w1.flags == FSTWeight<S>::isZero) {
    return w2;
  }
  if(w2.flags == FSTWeight<S>::isZero) {
    return w1;
  }
  if(w1.flags == FSTWeight<S>::isOne && w2.flags == FSTWeight<S>::isOne) {
    // TODO: remove the counting weights from the semirings as they are awkward and a bad idea..
    if(StaticPythonWeights::contains()) {
      // this can just return the two object
      py::object one = StaticPythonWeights::One();
      py::object r = one.attr("__add__")(one);
      return FSTWeight<S>(r);
    }
    throw fsterror("Trying to add with the static semiring one");
  }
  py::object o1 = w1.impl;
  if(w1.flags != FSTWeight<S>::isSet) {
    if(w2.flags == FSTWeight<S>::isSet) {
      o1 = w1.buildObject(w2.impl);
    } else if(StaticPythonWeights::contains() && w1.flags == FSTWeight<S>::isOne) {
      o1 = StaticPythonWeights::One();
    }else {
      throw fsterror("Undefined addition between two static elements of the field");
    }
  }
  py::object o2 = w2.buildObject(o1);
  py::object r = o1.attr("__add__")(o2);
  return FSTWeight<S>(r);
}

template<uint64 S>
FSTWeight<S> Times(const FSTWeight<S> &w1, const FSTWeight<S> &w2) {
  if(w1.flags == FSTWeight<S>::isOne) {
    return w2;
  }
  if(w2.flags == FSTWeight<S>::isOne) {
    return w1;
  }
  if(w1.flags == FSTWeight<S>::isZero || w2.flags == FSTWeight<S>::isZero) {
    return FSTWeight<S>::Zero();
  }
  if(w1.flags == FSTWeight<S>::isNoWeight || w2.flags == FSTWeight<S>::isNoWeight) {
    return FSTWeight<S>::NoWeight();
  }
  py::object o1 = w1.impl;
  if(w1.flags != FSTWeight<S>::isSet) {
    if(w2.flags != FSTWeight<S>::isSet) {
      throw fsterror("Undefined multiplication between two static elements of the field");
    } else {
      //assert(w2.flags == FSTWeight<S>::isSet);
      o1 = w1.buildObject(w2.impl);
    }
  }
  py::object o2 = w2.buildObject(o1);
  return FSTWeight<S>(o1.attr("__mul__")(o2));
}

template<uint64 S>
FSTWeight<S> Divide(const FSTWeight<S> &w1, const FSTWeight<S> &w2) {
  if(w2.flags == FSTWeight<S>::isOne) {
    return w1;
  }
  if(w1.flags == FSTWeight<S>::isZero) {
    return w1; // zero / anything = 0
  }
  if(w1.flags == FSTWeight<S>::isNoWeight || w2.flags == FSTWeight<S>::isNoWeight) {
    return FSTWeight<S>::NoWeight();
  }
  py::object o1 = w1.impl;
  if(w1.flags == FSTWeight<S>::isOne) {
    if(w2.flags != FSTWeight<S>::isSet) {
      throw fsterror("Undefined division between two static elements of the field");
    } else {
      o1 = w1.buildObject(w2.impl);
    }
  }

  py::object o2 = w2.buildObject(o1);

  return FSTWeight<S>(o1.attr("__div__")(o2));
}

template<uint64 S>
inline FSTWeight<S> Divide(const FSTWeight<S> &w1, const FSTWeight<S> &w2, const DivideType &type) {
  // TODO: there are left and right divide types that could be passed to python???
  // that might be important depending on which ring we are in
  return Divide(w1, w2);
}

template<uint64 S>
FSTWeight<S> Power(const FSTWeight<S> &w1, int n) {
  if(w1.isBuiltIn()) {
    return w1;
  } else {
    return FSTWeight<S>(w1.impl.attr("__pow__")(n));
  }
}

template<uint64 S>
bool operator==(FSTWeight<S> const &w1, FSTWeight<S> const &w2) {
  if(&w1 == &w2) return true;
  if(w1.flags == FSTWeight<S>::isNoWeight || w2.flags == FSTWeight<S>::isNoWeight)
    return w1.flags == FSTWeight<S>::isNoWeight && w2.flags == FSTWeight<S>::isNoWeight; // then this is invalid object, that should not be operated on
  py::object o1 = w1.impl;
  py::object o2 = w2.impl;
  if(w1.isBuiltIn() && !w2.isBuiltIn()) {
    o1 = w1.buildObject(o2);
  } else if(!w1.isBuiltIn() && w2.isBuiltIn()) {
    o2 = w2.buildObject(o1);
  } else if(w1.isBuiltIn() && w2.isBuiltIn()) {
    return w1.flags == w2.flags;
  }

  // this is the same underlying python object, these should return equal
  if(o1.ptr() == o2.ptr()) return true;

  py::object r = o1.attr("__eq__")(o2);
  return r.cast<bool>();

}

template<uint64 S>
inline bool operator!=(FSTWeight<S> const &w1, FSTWeight<S> const &w2) {
  return !(w1 == w2);
}

template<uint64 S>
bool ApproxEqual(FSTWeight<S> const &w1, FSTWeight<S> const &w2, const float &delta) {
  if(&w1 == &w2) return true;
  if(w1.flags == FSTWeight<S>::isNoWeight || w2.flags == FSTWeight<S>::isNoWeight)
    return w1.flags == FSTWeight<S>::isNoWeight && w2.flags == FSTWeight<S>::isNoWeight; // then this is invalid object, that should not be operated on
  py::object o1 = w1.impl;
  py::object o2 = w2.impl;
  if(w1.isBuiltIn() && !w2.isBuiltIn()) {
    o1 = w1.buildObject(o2);
  } else if(!w1.isBuiltIn() && w2.isBuiltIn()) {
    o2 = w2.buildObject(o1);
  } else if(w1.isBuiltIn() && w2.isBuiltIn()) {
    return w1.flags == w2.flags;
  }

  // this is the same underlying python object, these should return equal
  if(o1.ptr() == o2.ptr()) return true;

  py::object r = o1.attr("approx_eq")(o2, delta);
  return r.cast<bool>();
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
using PyArc = ArcTpl<FSTWeight<S> >;
template<uint64 S>
using PyFST = VectorFst<PyArc<S>, VectorState<PyArc<S> > >;



template<uint64 S>
class PythonArcSelector {
public:
  using StateId = typename PyFST<S>::StateId;
  using Weight = typename PyFST<S>::Weight;

  PythonArcSelector() {}

  explicit PythonArcSelector(uint64 seed) : random_engine(seed) {}

  size_t operator()(const Fst<PyArc<S> > &fst, StateId s) const {
    vector<double> scores;  // the unweighted scores for the arcs
    double sum = 0;

    ArcIterator<Fst<PyArc<S> > > iter(fst, s);
    while(!iter.Done()) {
      auto &v = iter.Value();
      const FSTWeight<S> &w = v.weight;
      if(w.isBuiltIn()) {
        if(w.flags != FSTWeight<S>::isZero) {
          // this could be fixed by using the static semiring class on the methods which make use of this
          throw fsterror("Unable to sample from fst that contains static but not zero weights");
        }
        scores.push_back(0);
      } else {
        py::object o = w.PythonObject();
        py::object r = o.attr("sampling_weight")();
        double f = r.cast<double>();
        if(f < 0) {
          throw fsterror("sampling_weight on edge must be >= 0");
        }
        sum += f;
        scores.push_back(f);
      }
      iter.Next();
    }

    // hopefully final is the last state, otherwise there is going to be some off by one error
    const FSTWeight<S> &w = fst.Final(s);
    if(w != Weight::Zero()) {
      if(w.isBuiltIn()) {
        throw fsterror("Unable to sample from fst that contains static but not zero weights");
      }
      py::object o = w.PythonObject();
      py::object r = o.attr("sampling_weight")();
      double f = r.cast<double>();
      sum += f;
      scores.push_back(f);
    }

    if(sum != sum) {  // check if the value is nan
      throw fsterror("Overflow when computing sampling_weight normalizer");
    }

    uniform_real_distribution<double> uniform(0.0, 1.0);
    double U = sum * uniform(random_engine);
    for(size_t i = 0; i < scores.size(); i++) {
      U -= scores[i];
      if(U <= 0) { return i; }
    }
    return scores.size() - 1; // always return something in the case that we got all the way to the end
  }

 private:
  // this is seeded when this class is constructed.  The seed comes from python random.randint
  mutable default_random_engine random_engine;

};


template<typename Arc>
class IOLabelCompare {
public:
  IOLabelCompare() {}

  bool operator()(const Arc &arc1, const Arc &arc2) const {
    // try the input first otherwise the output
    return arc1.ilabel == arc2.ilabel ? arc1.olabel < arc2.olabel : arc1.ilabel < arc2.ilabel;
  }

  uint64 Properties(uint64 props) const {
    return (props & kArcSortProperties) | kILabelSorted |
           (props & kAcceptor ? kOLabelSorted : 0);
  }
};


template<uint64 S>
void add_arc(PyFST<S> &self, int64 from, int64 to,
             int64 input_label, int64 output_label, py::object weight) {
  // TODO: check if the weight is the correct python instance
  ErrorCatcher e;

  if(!check_is_weight(weight)) {
    throw fsterror("weight is missing required method");
  }

  FSTWeight<S> w1(weight);
  PyArc<S> a(input_label, output_label, w1, to);
  self.AddArc(from, a);
}

template<uint64 S>
void set_final(PyFST<S> &self, int64 state, py::object weight) {
  ErrorCatcher e;

  if(!check_is_weight(weight)) {
    throw fsterror("weight is missing required method");
  }

  FSTWeight<S> w1(weight);
  self.SetFinal(state, w1);
}

template<uint64 S>
py::object final_weight(PyFST<S> &self, int64 state) {
  ErrorCatcher e;

  FSTWeight<S> finalW = self.Final(state);
  py::object r = finalW.PythonObject();

  return r;
}

template<uint64 S>
unique_ptr<PyFST<S> > compose(const PyFST<S> & a, const vector<const PyFST<S>*> args) {
  ErrorCatcher e;

  unique_ptr<PyFST<S> > ret (new PyFST<S>());
  // we have to sort the arcs such that this can be compared between objects
  IOLabelCompare<PyArc<S> > comp;

  // create all of the sorted FSTs as this is required for
  vector<unique_ptr<ArcSortFst<PyArc<S>, IOLabelCompare<PyArc<S> > > > > sorted;
  sorted.reserve(args.size() + 1);
  sorted.emplace_back(new ArcSortFst<PyArc<S>, IOLabelCompare<PyArc<S> > >(a, comp));
  for(const PyFST<S> *f : args) {
    sorted.emplace_back(new ArcSortFst<PyArc<S>, IOLabelCompare<PyArc<S> > >(*f, comp));
  }

  vector<unique_ptr<ComposeFst<PyArc<S>, DefaultCacheStore<PyArc<S> > > > > lazy_composed;
  if(args.size() > 1) {
    lazy_composed.reserve(sorted.size() - 1);
    const Fst<PyArc<S> > &s1 = *sorted[0];
    const Fst<PyArc<S> > &s2 = *sorted[1];
    lazy_composed.emplace_back(new ComposeFst<PyArc<S>, DefaultCacheStore<PyArc<S> > > (s1, s2));
    for(int i = 2; i < sorted.size() - 1; i++) {
      const Fst<PyArc<S> > &s3 = *lazy_composed.back();
      const Fst<PyArc<S> > &s4 = *sorted[i];
      lazy_composed.emplace_back(new ComposeFst<PyArc<S>, DefaultCacheStore<PyArc<S> > >(s3, s4));
    }

    // do the final compose operation into the ret object
    Compose(*lazy_composed.back(), *sorted.back(), ret.get());
  } else {
    Compose(*sorted[0], *sorted[1], ret.get());
  }

  return ret;
}



template<uint64 S>
void define_class(pybind11::module &m, const char *name) {

  // prevent fst from killing the process
  // we just need to set this somewhere before using openfst
  // TODO: if an exception is thrown, then this will probably leak the object it was constructing
  FLAGS_fst_error_fatal = false;

  py::class_<PyFST<S> >(m, name)
    .def(py::init<>())

#define d(name) .def(#name, & PyFST<S> :: name)
    .def("AddArc", &add_arc<S>) // keep the weight alive when added
    d(AddState)
    // there are more than one method with this name but different type signatures
    .def("DeleteArcs", [](PyFST<S> *m, int64 v) { if(m) m->DeleteArcs(v); })

    .def("DeleteStates", [](PyFST<S> *m) { if(m) m->DeleteStates(); })

    d(NumStates)
    d(ReserveArcs)
    d(ReserveStates)

    .def("SetFinal", &set_final<S>)
    d(SetStart)


    d(NumArcs)
    d(Start)

    .def("FinalWeight", &final_weight<S>)
#undef d



    // compare two FST methods
    .def("Isomorphic", [](const PyFST<S> &a, const PyFST<S> &b, float delta) {
        ErrorCatcher e;
        return Isomorphic(a, b, delta);
      })

    // methods that will generate a new FST
    .def("Concat", [](const PyFST<S> &a, const PyFST<S> &b) {
        ErrorCatcher e;
        unique_ptr<PyFST<S> > ret(b.Copy());
        Concat(a, ret.get());
        return ret;
      })

    .def("Compose", &compose<S>)

    .def("Determinize", [](const PyFST<S> &a, py::object semiring, float delta, py::object weight, bool allow_non_functional) {
        ErrorCatcher e;
        StaticPythonWeights w(semiring);
        FSTWeight<S> weight_threshold(weight);

        unique_ptr<PyFST<S> > ret(new PyFST<S>());

        DeterminizeOptions<PyArc<S> > ops
          (delta, weight_threshold,
           kNoStateId, 0,
           allow_non_functional ? DETERMINIZE_DISAMBIGUATE : DETERMINIZE_FUNCTIONAL,
           false);

        Determinize(a, ret.get(), ops);
        return ret;
      })

    .def("Disambiguate", [](const PyFST<S> &a, py::object semiring) {
        ErrorCatcher e;
        StaticPythonWeights w(semiring);
        unique_ptr<PyFST<S> > ret(new PyFST<S>());

        Disambiguate(a, ret.get());

        return ret;
      })

    .def("Project", [](const PyFST<S> &a, int type) {
        ErrorCatcher e;
        unique_ptr<PyFST<S> > ret(new PyFST<S>());

        ProjectType typ = type ? PROJECT_INPUT : PROJECT_OUTPUT;
        Project(a, ret.get(), typ);
        return ret;
      })

    .def("Difference", [](const PyFST<S> &a, const PyFST<S> &b) {
        ErrorCatcher e;
        unique_ptr<PyFST<S> > ret(new PyFST<S>());

        if((a.Properties(kNotAcceptor, false) & b.Properties(kNotAcceptor, false) & kNotAcceptor) != 0) {
          throw fsterror("Difference only works on FSA acceptors");
        }

        IOLabelCompare<PyArc<S> > comp;
        //const ArcSortFst<PyArc<S>, IOLabelCompare<PyArc<S> > > as(a, comp);
        // sorted by the input lables
        const ArcSortFst<PyArc<S>, IOLabelCompare<PyArc<S> > > bs(b, comp);

        Difference(a, bs, ret.get());
        return ret;
      })

    .def("Invert", [](const PyFST<S> &a) {
        ErrorCatcher e;
        unique_ptr<PyFST<S> > ret(a.Copy());
        Invert(ret.get());
        return ret;
      })

    .def("Prune", [](const PyFST<S> &a, py::object weight) {
        ErrorCatcher e;
        FSTWeight<S> weight_threshold(weight);

        unique_ptr<PyFST<S> > ret(new PyFST<S>());
        Prune(a, ret.get(), weight_threshold);
        return ret;
      })

    .def("Intersect", [](const PyFST<S> &a, const PyFST<S> &b) {
        unique_ptr<PyFST<S> > ret(new PyFST<S>());

        if((a.Properties(kNotAcceptor, false) & b.Properties(kNotAcceptor, false) & kNotAcceptor) != 0) {
          throw fsterror("Intersect only works on FSA acceptors");
        }

        IOLabelCompare<PyArc<S> > comp;
        //const ArcSortFst<PyArc<S>, IOLabelCompare<PyArc<S> > > as(a, comp);
        const ArcSortFst<PyArc<S>, IOLabelCompare<PyArc<S> > > bs(b, comp);

        Intersect(a, bs, ret.get());
        return ret;
      })

    .def("Union", [](const PyFST<S> &a, const PyFST<S> &b) {
        ErrorCatcher e;
        unique_ptr<PyFST<S> > ret(a.Copy());
        Union(ret.get(), b);
        return ret;
      })

    .def("Minimize", [](const PyFST<S> &a, double delta) {
        ErrorCatcher e;
        unique_ptr<PyFST<S> > ret(a.Copy());
        Minimize(ret.get(), static_cast<PyFST<S>* >(nullptr), delta);
        return ret;
      })

    .def("Push", [](const PyFST<S> &a, py::object semiring, int mode) {
        ErrorCatcher e;
        StaticPythonWeights w(semiring);
        unique_ptr<PyFST<S> > ret(new PyFST<S>());
        if(mode == 0) {
          Push<PyArc<S>, REWEIGHT_TO_INITIAL>(a, ret.get(), kPushWeights);
        } else {
          Push<PyArc<S>, REWEIGHT_TO_FINAL>(a, ret.get(), kPushWeights);
        }
        return ret;
      })

    .def("ShortestPath", [](const PyFST<S> &a, py::object semiring, int count) {
        ErrorCatcher e;
        StaticPythonWeights w(semiring);
        unique_ptr<PyFST<S> > ret(new PyFST<S>());

        ShortestPath(a, ret.get(), count);

        return ret;
      })

    .def("ShortestDistance", [](const PyFST<S> &a, bool reverse) {
        ErrorCatcher e;
        vector<FSTWeight<S> > distances;

        ShortestDistance(a, &distances, reverse);

        vector<py::object> ret;
        ret.reserve(distances.size());
        for(auto &w : distances) {
          ret.push_back(w.PythonObject());
        }

        return ret;
      })

    .def("SumPaths", [](const PyFST<S> &a) {
        ErrorCatcher e;
        FSTWeight<S> ret = ShortestDistance(a);

        return ret.PythonObject();
      })

    .def("TopSort", [](const PyFST<S> &a) {
        ErrorCatcher e;
        unique_ptr<PyFST<S> > ret(a.Copy());

        TopSort(ret.get());

        return ret;
      })

    .def("RmEpsilon", [](const PyFST<S> &a, py::object semiring) {
        ErrorCatcher e;
        StaticPythonWeights w(semiring);
        unique_ptr<PyFST<S> > ret(a.Copy());
        RmEpsilon(ret.get());
        return ret;
      })

    .def("RandomPath", [](const PyFST<S> &a, int count, uint64 rand_seed) {
        ErrorCatcher e;
        unique_ptr<PyFST<S> > ret(new PyFST<S>());

        PythonArcSelector<S>  selector(rand_seed);
        // unsure how having the count as the weight will work?  The output semiring is potentially the same as this one??
        // but we could get the counts back??
        // maybe we should just wrap the FST with the value class instead of having the customized seminring?
        RandGenOptions<PythonArcSelector<S> > ops(selector, std::numeric_limits<int32>::max(), count, false /* don't use weighted, as this would just return an fst with the prob of the path */);
        RandGen(a, ret.get(), ops);

        return ret;
      })

    .def("Verify", [](const PyFST<S> &a) {
        ErrorCatcher e;
        return Verify(a);
      })

    .def("Closure", [](const PyFST<S> &a, int mode) {
        ErrorCatcher e;
        unique_ptr<PyFST<S> > ret(a.Copy());
        Closure(ret.get(), mode == 0 ? CLOSURE_STAR : CLOSURE_PLUS);
        return ret;
      })

    .def("Reverse", [](const PyFST<S> &a) {
        ErrorCatcher e;
        unique_ptr<PyFST<S> > ret(new PyFST<S>());
        Reverse(a, ret.get());
        return ret;
      })

    .def("ArcList", [](const PyFST<S> &a, int64 state) {
        // input label, output label, to state, weight
        ErrorCatcher e;
        vector<py::tuple> ret;

        ArcIterator<PyFST<S> > iter(a, state);
        while(!iter.Done()) {
          auto &v = iter.Value();
          const FSTWeight<S> &w = v.weight;
          py::object oo = w.PythonObject();
          ret.push_back(py::make_tuple(v.ilabel, v.olabel, v.nextstate, oo));
          iter.Next();
        }

        FSTWeight<S> finalW = a.Final(state);
        if(finalW.flags != FSTWeight<S>::isZero && finalW.flags != FSTWeight<S>::isNoWeight) {
          // then there is something here that we should push I guess?
          ret.push_back(py::make_tuple(0, 0, -1, finalW.PythonObject()));
        }

        return ret;
      })

    .def("ToString", [](const PyFST<S> &a) {
        ErrorCatcher e;
        ostringstream out;
        fst::script::PrintFst(a, out);
        return out.str();
      });

    ;
}

PYBIND11_MODULE(openfst_wrapper_backend, m) {
  m.doc() = "Backing wrapper for OpenFst";

  define_class<kSemiring | kCommutative>(m, "FSTBase");
  define_class<kSemiring | kCommutative | kPath | kIdempotent>(m, "FSTPath");

  static py::exception<fsterror> ex(m, "FSTError");
  py::register_exception_translator([](std::exception_ptr p) {
      try {
        if (p) std::rethrow_exception(p);
      } catch (fsterror &e) {
        ex(e.what());
      }
    });
}
