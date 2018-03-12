/**
   Written by Matthew Francis-Landau (2018)

   Wrapper for OpenFST that supports defining custom semirings in python
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

#include <fst/script/print.h>


using namespace std;
using namespace fst;
namespace py = pybind11;

// this error should just become a runtime error in python land
class fsterror : public std::exception {
private:
  const char* error;
  string str;
public:
  fsterror(const char *e): error(e) {}
  fsterror(string s) : str(s), error(s.c_str()) { }
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
      //throw fsterror(e);
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
    //hasattr(weight, "__hash__") &&
    hasattr(weight, "__eq__") &&
    hasattr(weight, "_openfst_str") &&
    hasattr(weight, "_member") &&
    hasattr(weight, "_quantize") &&
    hasattr(weight, "_reverse") &&
    hasattr(weight, "_approx_eq") &&
    hasattr(weight, "_sampling_weight") &&
    hasattr(weight, "one") &&
    hasattr(weight, "zero");
}


// this is SUCH a HACK
namespace {
  static py::handle semiring_class;
  static py::object one_cache;
  static py::object zero_cache;
}
class StaticPythonWeights {
private:
  //py::handle old;
public:
  StaticPythonWeights(py::handle semiring) {
    if(!hasattr(semiring, "zero") || !hasattr(semiring, "one")) {
      throw fsterror("Semiring does not have zero and one method to construct new instances");
    }
    //old = semiring_class;
    assert(semiring_class.ptr() == nullptr);
    semiring_class = semiring;
  }
  ~StaticPythonWeights() {
    semiring_class = nullptr;
    one_cache.release();
    zero_cache.release();
  }

  static bool contains() {
    // if(semiring_class.ptr() != nullptr) {
    //   py::object foo = semiring_class.attr("one")();
    // }
    //return false;
    return semiring_class.ptr() != nullptr;
  }

  static py::object One() {
    // should have already checked
    if(one_cache.ptr() == nullptr) {
      one_cache = semiring_class.attr("one")();
    }
    return one_cache;
  }

  static py::object Zero() {
    if(zero_cache.ptr() == nullptr) {
      zero_cache = semiring_class.attr("zero")();
    }
    return zero_cache;
  }
};

template<uint64 S>
class FSTWeight {
private:
  FSTWeight(int32 g, bool __) : flags(g), impl(), count(0) {} // the static value constructor
public:
  using ReverseWeight = FSTWeight<S>;

  enum {
    isZero = 0x1,
    isOne = 0x2,
    isNoWeight = 0x4,
    isSet = 0x8,
    isCount = 0x10,
  };

  // the python object that we are wrapping
  int16_t flags;
  py::object impl;
  uint32 count;  // represents a count in the case that it added two elements of the static semiring

  FSTWeight() : flags(isNoWeight), impl(), count(0) {
  }

  FSTWeight(uint32 count) : flags(isCount), count(count) {
    assert(count > 0);
    if(count  == 0)
      throw fsterror("Invalid static count");
  }

  FSTWeight(py::object i) : flags(isSet), impl(i), count(0) {
    if(!check_is_weight(impl))
      throw fsterror("Value does not implement Weight interface");
  }

  static FSTWeight<S> Zero() {
    static const FSTWeight<S> zero = FSTWeight<S>(isZero, true);
    if(StaticPythonWeights::contains()) {
      return FSTWeight<S>(StaticPythonWeights::Zero());
    }
    return zero;
  }
  static FSTWeight<S> One() {
    static const FSTWeight<S> one = FSTWeight<S>(isOne, true);
    if(StaticPythonWeights::contains()) {
      return FSTWeight<S>(StaticPythonWeights::One());
    }
    return one;
  }
  static const FSTWeight<S>& NoWeight() {
    static const FSTWeight<S> no_weight = FSTWeight<S>(isNoWeight, true);
    return no_weight;
  }

  bool isBuiltIn() const {
    assert(flags != 0);
    return (flags & (isZero | isOne | isNoWeight | isCount)) != 0;
  }

  static const string& Type() {
    static const string type("python");
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
      return (flags & (isOne | isZero | isCount)) != 0;
    } else {
      py::object r = impl.attr("_member")();
      return r.cast<bool>();
    }
  }

  FSTWeight<S> Reverse() const {
    if(isBuiltIn()) {
      // TODO: this should print a warning or something
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
      } else if(flags == isCount) {
        return os << "(" << count << ")";
      } else {
        return os << "(invalid)";
      }
    } else {
      // it appears that openfst uses this string inside of its determinize and disambiguate
      // if there are minor non matching floating point errors then openfst breaks
      py::object s = impl.attr("_openfst_str")();
      return os << s.cast<string>();
    }
  }

  size_t Hash() const {
    if(isBuiltIn()) {
      return flags + count;
    } else {
      return py::hash(impl);
    }
  }

  FSTWeight<S>& operator=(const FSTWeight<S>& other) {
    impl = other.impl;
    flags = other.flags;
    count = other.count;
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
    } else if(flags == isCount) {
      throw fsterror("trying to statically build counting python object");
    } else {
      return py::cast("__FST_INVALID__");
    }
  }

  py::object buildObject(const py::object &other) const {
    // in the case that we are just wrapping a count, we still want to construct some object
    // that can be used to track the semiring operations
    if(flags == isSet) {
      return impl;
    } else if(flags == isOne) {
      return other.attr("one")();
    } else if(flags == isZero) {
      return other.attr("zero")();
    } else if(flags == isNoWeight) {
      // unsure what could be done here
      return py::cast("__FST_INVALID__");
    } else {
      assert(flags == isCount);
      py::object ret = other.attr("zero")(); // get the zero element
      py::object v = other.attr("one")();
      if(!check_is_weight(ret) || !check_is_weight(v))
        throw fsterror("Operation return non weight");
      uint32 i = count;
      // build this object using multiple add instructions as we do not know how the prod and multiply interact with eachother otherwise
      // TODO: check properties of the semiring and determine if we need to actually do this adding operation
      while(i) {
        if(i & 0x1) {
          ret = ret.attr("__add__")(v);
        }
        i >>= 1;
        if(!i) break;
        v = v.attr("__add__")(v);
      }
      if(!check_is_weight(ret))
        throw fsterror("Operation return non weight");
      return ret;
    }
  }

  virtual ~FSTWeight<S>() {
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
  if(w1.flags == FSTWeight<S>::isOne || w2.flags == FSTWeight<S>::isOne) {
    return FSTWeight<S>(2); // the counting element
    //throw fsterror("Trying to add with the static semiring one");
  }
  py::object o1 = w1.impl;
  if(w1.flags != FSTWeight<S>::isSet) {
    if(w2.flags == FSTWeight<S>::isCount) {
      return FSTWeight<S>(w1.count + w2.count);
    } else if(w2.flags == FSTWeight<S>::isSet) {
      o1 = w1.buildObject(w2.impl);
    } else {
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
  py::object o1 = w1.impl;
  if(w1.flags != FSTWeight<S>::isSet) {
    if(w2.flags != FSTWeight<S>::isSet) {
      throw fsterror("Undefined multiplication between two static count elements of the field");
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
  py::object o1 = w1.impl;
  if(w1.flags == FSTWeight<S>::isCount || w1.flags == FSTWeight<S>::isOne) {
    if(w2.flags != FSTWeight<S>::isSet) {
      throw fsterror("Undefined division between two static count elements of the field");
    } else {
      //assert(w2.flags == FSTWeight<S>::isSet);
      o1 = w1.buildObject(w2.impl);
    }
  }

  // else if(w1.flags != FSTWeigh<S>::isSet) {
  //   throw fsterror("Unable to divide unset weights");
  // }
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
    if(w1.flags == FSTWeight<S>::isCount)
      throw fsterror("Unable to construct power of static count");
    return w1;
  } else {
    return FSTWeight<S>(w1.impl.attr("__pow__")(n));
  }
}

template<uint64 S>
bool operator==(FSTWeight<S> const &w1, FSTWeight<S> const &w2) {
  py::object o1 = w1.impl;
  py::object o2 = w2.impl;
  if(w1.isBuiltIn() && !w2.isBuiltIn()) {
    o1 = w1.buildObject(o2);
  } else if(!w1.isBuiltIn() && w2.isBuiltIn()) {
    o2 = w2.buildObject(o1);
  } else if(w1.flags == FSTWeight<S>::isCount) {
    return w2.flags == FSTWeight<S>::isCount && w1.count == w2.count;
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

// template<uint64 S>
// inline bool operator<=(FSTWeight<S> const &w1, FSTWeight<S> const &w2) {
//   // py::object o1 = w1.impl;
//   // py::object o2 = w2.impl;
//   // if(w1.isBuiltIn() && !w2.isBuiltIn()) {
//   //   o1 = w1.buildObject(o2);
//   // } else if(!w1.isBuiltIn() && w2.isBuiltIn()) {
//   //   o2 = w2.buildObject(o1);
//   // } else if(w1.flags == FSTWeight<S>::isCount) {
//   //   throw fsterror("not defined with count");
//   //   //return w2.flags == FSTWeight<S>::isCount && w1.count <= w2.count;
//   // } else {
//   //   return w1.flags == w2.flags;
//   // }

//   // py::object r = o1.attr("_openfst_le")(o2);
//   // bool rr = r.cast<bool>();

//   // return rr;

//   auto r = Plus(w1, w2);
//   assert(r == w1 || r == w2);
//   assert(r != w1 || r != w2 || w1 == w2);
//   return Plus(w1, w2) == w1;

// }

template<uint64 S>
bool ApproxEqual(FSTWeight<S> const &w1, FSTWeight<S> const &w2, const float &delta) {
  py::object o1 = w1.impl;
  py::object o2 = w2.impl;
  if(w1.isBuiltIn() && !w2.isBuiltIn()) {
    o1 = w1.buildObject(o2);
  } else if(!w1.isBuiltIn() && w2.isBuiltIn()) {
    o2 = w2.buildObject(o1);
  } else if(w1.flags == FSTWeight<S>::isCount) {
    return w2.flags == FSTWeight<S>::isCount && w1.count == w2.count;
  } else if(w1.isBuiltIn() && w2.isBuiltIn()) {
    return w1.flags == w2.flags;
  }

  py::object r = o1.attr("_approx_eq")(o2, delta);
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
    //const auto n = fst.NumArcs(s) + (fst.Final(s) != Weight::Zero());
    vector<double> scores;  // the unweighted scores for the arcs
    double sum = 0;

    ArcIterator<Fst<PyArc<S> > > iter(fst, s);
    while(!iter.Done()) {
      auto &v = iter.Value();
      const FSTWeight<S> &w = v.weight;
      if(w.isBuiltIn()) {
        if(w.flags != FSTWeight<S>::isZero) {
          throw fsterror("Unable to sample from fst that contains static but not zero weights");
        }
        scores.push_back(0);
      } else {
        py::object o = w.PythonObject();
        py::object r = o.attr("_sampling_weight")();
        double f = r.cast<double>();
        if(f < 0) {
          throw fsterror("_sampling_weight on edge must be >= 0");
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
      py::object r = o.attr("_sampling_weight")();
      double f = r.cast<double>();
      sum += f;
      scores.push_back(f);
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
  //mutable std::mt19937_64 rand_;
  // TODO: this should just be a global that can be seeded at some point?
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

// template<uint64 S, bool b> struct path_runner {};
// template <uint64 S> struct path_runner<S, false> {
//   static void run(const PyFST<S> &a, PyFST<S> *r, int c) {
//     throw fsterror("Shortest path only defined on path semirings");
//   }
// };

// template <uint64 S> struct path_runner<S, true> {
//   static void run(const PyFST<S> &a, PyFST<S> *r, int c) {
//     vector<FSTWeight<S> > distances;
//     fst::internal::NShortestPath(a, r, distances, c);
//   }
// };


// #include <queue>

// // there is a problem with open fsts implementation of getting the single best path,
// // this has the wrong runtime due to using shortest distance which is O(V+E)
// template<uint64 S>
// void shortest_path_hack(const PyFST<S> &self, PyFST<S> &out) {
//   vector<FSTWeight<S> > distances;

//   // compute the shortest distance from all states to the ending state
//   ShortestDistance(self, &distances, true);

//   typedef typename PyFST<S>::StateId StateId;

//   StateId last = out.AddState();
//   out.SetStart(last);

//   StateId s = self.Start();

//   while(true) {
//     FSTWeight<S> finalW = self.Final(s);
//     if(distances[s] == finalW) {
//       // then this is the final state
//       out.SetFinal(last, finalW);
//       return;
//     }

//     ArcIterator<PyFST<S> > iter(self, s);
//     // loop through the start states and identify the one with the lowest final weight in the distance
//     StateId next = -1;
//     //FSTWeight<S> dist;
//     PyArc<S> arc;
//     while(!iter.Done()) {
//       auto &v = iter.Value();
//       if(distances[v.nextstate] <= distances[s]) {
//         arc = v;
//         next = arc.nextstate;
//         //dist = distances[arc.nextstate];
//         break;
//       }
//       iter.Next();
//     }
//     if(next == -1)
//       return;
//     StateId nn = out.AddState();
//     PyArc<S> adding(arc.ilabel, arc.olabel, arc.weight, nn);
//     out.AddArc(last, adding);
//     last = nn;
//     s = arc.nextstate;
//   }
// }


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

    // methods that will generate a new FST or
    .def("Concat", [](const PyFST<S> &a, PyFST<S> &b) {
        ErrorCatcher e;
        PyFST<S> *ret = b.Copy();
        Concat(a, ret);
        return ret;
      })

    .def("Compose", &compose<S>)

    .def("Determinize", [](const PyFST<S> &a, py::object semiring, float delta, py::object weight, bool allow_non_functional) {
        ErrorCatcher e;
        StaticPythonWeights w(semiring);
        FSTWeight<S> weight_threshold(weight);

        unique_ptr<PyFST<S> > ret(new PyFST<S>());

        const DeterminizeOptions<PyArc<S> > ops
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

    .def("Push", [](const PyFST<S> &a, int mode) {
        ErrorCatcher e;
        unique_ptr<PyFST<S> > ret(new PyFST<S>());
        if(mode == 0) {
          Push<PyArc<S>, REWEIGHT_TO_INITIAL>(a, ret.get(), kPushWeights);
        } else {
          Push<PyArc<S>, REWEIGHT_TO_FINAL>(a, ret.get(), kPushWeights);
        }
        return ret;
      })

    // .def("ShortestPathHack", [](const PyFST<S> &a) {
    //     ErrorCatcher e;
    //     unique_ptr<PyFST<S> > ret(new PyFST<S>());

    //     if((S & kPath) == 0) {
    //       throw fsterror("Shortest path only defined on path semirings");
    //     }

    //     shortest_path_hack(a, *ret.get());
    //     return ret;
    //   })

    .def("ShortestPath", [](const PyFST<S> &a, int count) {
        ErrorCatcher e;
        unique_ptr<PyFST<S> > ret(new PyFST<S>());

        // if((S & kPath) == 0) {
        //   throw fsterror("Shortest path only defined on path semirings");
        // }

        // shortest_path_hack(a, *ret.get());

        // use the Astar implementation with no estimate of the future reward, this is just Dijkstra
        // the documentation claims that it is finding the shortest path based off the weights
        // but the default implementation is just using the state id at the sorting order

        // vector<FSTWeight<S> > distances;
        // typedef NaturalShortestFirstQueue<typename PyFST<S>::StateId, FSTWeight<S> > queue_t;
        // queue_t queue(distances);

        // // struct astart_estimate_s {
        // //   const FSTWeight<S> one = FSTWeight<S>::One();
        // //   const FSTWeight<S> &operator()(typename PyFST<S>::StateId) const { return one; }
        // // };
        // // astart_estimate_s estimate;
        // // typedef NaturalAStarQueue<typename PyFST<S>::StateId, FSTWeight<S>, astart_estimate_s> queue_t;
        // // queue_t queue(distances, estimate);

        // AnyArcFilter<PyArc<S> > arc_filter;

        // // precompute the distances from the start state to all of the states
        // // without this openFST seems to hit some bug where it causes it to return the wrong answer
        // //ShortestDistance(a, &distances, false);

        // ShortestPathOptions<PyArc<S>, queue_t, AnyArcFilter<PyArc<S> > > opts(&queue, arc_filter, count, false);

        // ShortestPath(a, ret.get(), &distances, opts);

        // for(int d = 0; d < distances.size(); d++) {
        //   cout << "distance: " << d << " " << distances[d] << endl;
        // }

        // path_runner<S, (S & kPath) != 0>::run(a, ret.get(), count);

        ShortestPath(a, ret.get(), count);

        return ret;
      })

    .def("ShortestDistance", [](const PyFST<S> &a, bool reverse) {
        ErrorCatcher e;
        vector<FSTWeight<S> > distances;

        // TODO: same problem as shortest path

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

    .def("RmEpsilon", [](const PyFST<S> &a) {
        ErrorCatcher e;
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
        RandGenOptions<PythonArcSelector<S> > ops(selector, std::numeric_limits<int32>::max(), count, true);
        RandGen(a, ret.get(), ops);

        return ret;
      })

    .def("ArcList", [](const PyFST<S> &a, int64 state) {
        // input label, output label, to state, weight
        ErrorCatcher e;
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
        if(finalW.flags != FSTWeight<S>::isZero && finalW.flags != FSTWeight<S>::isNoWeight) {
          // then there is something here that we should push I guess?
          ret.push_back(make_tuple(0, 0, -1, finalW.PythonObject()));
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
  m.doc() = "Backing wrapper for OpenFST";

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
