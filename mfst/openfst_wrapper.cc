#include <memory>
#include <string>
#include <sstream>

#include <pybind11/pybind11.h>

#include <fst/fst.h>
#include <fst/mutable-fst.h>
#include <fst/arc.h>
#include <fst/script/fst-class.h>
#include <fst/script/arc-class.h>



using namespace std;
using namespace fst;
using namespace fst::script;
namespace py = pybind11;

class fsterror : public exception {
private:
  const char* error;
public:
  fsterror(const char *e): error(e) {}
  virtual const char* what() { return error; }
};

class FSTWeight {
public:

  // the python object that we are wrapping
  py::object impl;

  FSTWeight() = default;
  FSTWeight(py::object i) : impl(i) {
    if(i.is_none())
      throw fsterror("Can not have a None edge weight");
  }

  static const FSTWeight& Zero() {
    static const FSTWeight zero = FSTWeight();
    return zero;
  }
  static const FSTWeight& One() {
    static const FSTWeight one = FSTWeight();
    return one;
  }
  static const FSTWeight& NoWeight() {
    static const FSTWeight no_weight = FSTWeight();
    return no_weight;
  }

  static const string& Type() {
    // TODO: this is wrong
    // we should register our own type of weight, but for some reason that isn't working????
    static const string type("tropical");
    return type;
  }

  // this might not be what the user wants to use for this???
  static const uint64 Properties () { return kSemiring | kCommutative; }

  bool Member() const {
    py::object r = impl.attr("member")();
    return r.cast<bool>();
  }

  // do nothing?
  FSTWeight Reverse() { return *this; }

  FSTWeight Quantize(float delta = kDelta) const {
    return FSTWeight(impl.attr("quantize")(delta));
  }

  std::istream &Read(std::istream &strm) { throw fsterror("not implemented"); }
  std::ostream &Write(std::ostream &strm) { throw fsterror("not implemented"); }

  size_t Hash() const {
    py::object r = impl.attr("__hash__")();
    return r.cast<size_t>();
  }

};

inline FSTWeight Plus(const FSTWeight &w1, const FSTWeight &w2) {
  // these identity elements have to be handled specially
  // as they are the same for all semirings that are defined in python and thus contain no value
  if(&w1 == &FSTWeight::Zero()) {
    return w2;
  }
  if(&w2 == &FSTWeight::Zero()) {
    return w1;
  }
  if(&w1 == &FSTWeight::One() || &w2 == &FSTWeight::One()) {
    throw fsterror("Trying to add with the static semiring one");
  }
  return FSTWeight(w1.impl.attr("add")(w2.impl));
}

inline FSTWeight Times(const FSTWeight &w1, const FSTWeight &w2) {
  if(&w1 == &FSTWeight::One()) {
    return w2;
  }
  if(&w2 == &FSTWeight::One()) {
    return w1;
  }
  if(&w1 == &FSTWeight::Zero() || &w2 == &FSTWeight::Zero()) {
    // I suppose that we could just return zero?
    // which could just be static instead of throwing this error
    throw fsterror("Trying to multiply with static semiring zero");
  }
  return FSTWeight(w1.impl.attr("multiply")(w2.impl));
}

inline FSTWeight Divide(const FSTWeight &w1, const FSTWeight &w2) {
  if(&w2 == &FSTWeight::One()) {
    return w1;
  }
  return FSTWeight(w1.impl.attr("divide")(w2.impl));
}

inline FSTWeight Power(const FSTWeight &w1, int n) {
  return FSTWeight(w1.impl.attr("power")(n));
}

inline bool operator==(FSTWeight const &w1, FSTWeight const &w2) {
  py::object r = w1.impl.attr("equal")(w2.impl);
  return r.cast<bool>();
}

std::ostream &operator<<(std::ostream &os, const FSTWeight &w) {
  return os << py::str(w.impl.attr("__str__"));
}


//using FSTClass = MutableFstClass<ArcTpl<FSTWeight> >;

// class WeightBase;

using Arc = ArcTpl<FSTWeight>;

static FstRegister < Arc > our_weights;


MutableFstClass* create_fst() {
  //return MutableFstClass::Create
  //return *nullptr;
  //return MutableFstClass::Read(""); // read the empty string file to get an empty FST instance

  // before we attempt to construct the FST, ensure that we are loaded into openfst list of classes
  // can't seem to do this "right"
  // auto r = FstClassIoRegister::GetRegister();
  // r->

  MutableFstClass* r = new VectorFstClass("standard") ;///Arc::Type());
  //r->GetMutableFst<Arc>()->SetArcType(Arc::Type());
  return r;

  // stringstream ss("");

  // std::unique_ptr<MutableFstClass> fst(MutableFstClass::Read(ss, true));
  // MutableFstClass *r = MutableFstClass::Read<Arc>(ss, FstReadOptions());
  // return r;

  // // this is apparently the way that openfst wants to do this....omfg
  // FstClass *fst = new FstClass();
  // return static_cast<MutableFstClass*>(fst);
}

// // wrap the python class object into the fst
// class FSTWeightWrap : public WeightImplBase {
//   py::object wrapped;

//   //WeightBase &self() { return wrapped.cast<WeightBase&>(); }
// public:

//   virtual FSTWeightWrap *Copy() const {
//     // FSTWeightWrap *r = new FSTWeightWrap;
//     // r->wrapped = wrapped.attr("clone")();
//     // return r;
//     return nullptr;
//   }
//   virtual void Print(std::ostream *o) const {
//     *o << ToString();
//   }
//   virtual const string &Type() const {
//     static const string t("WrappedPythonWeight");
//     return t;
//   }
//   virtual string ToString() const {
//     return py::str(wrapped.attr("__str__")());
//   }
//   virtual bool operator==(const WeightImplBase &other) const {
//     py::object r = wrapped.attr("equal")(static_cast<FSTWeightWrap>(&other)->wrapped);
//     return r.cast<bool>();
//   }
//   virtual bool operator!=(const WeightImplBase &other) const {
//     return !((*this) == other);
//   }
//   virtual WeightImplBase &PlusEq(const WeightImplBase &other) {
//     wrapped = wrapped.attr("plus")(static_cast<FSTWeightWrap>(&other)->wrapped);
//     return *this;
//   }
//   virtual WeightImplBase &TimesEq(const WeightImplBase &other) {
//     wrapped = wrapped.attr("times")(static_cast<FSTWeightWrap>(&other)->wrapped);
//     return *this;
//   }
//   virtual WeightImplBase &DivideEq(const WeightImplBase &other) {
//     wrapped = wrapped.attr("divide")(static_cast<FSTWeightWrap>(&other)->wrapped);
//     return *this;
//   }
//   virtual WeightImplBase &PowerEq(size_t n) {
//     wrapped = wrapped.attr("power")(n);
//     return *this;
//   }
//   virtual ~FSTWeightWrap() {}
// };

// class WeightBase {
// public:
//   virtual py::object create()=0;
//   virtual py::object add(WeightBase *other)=0;
// };

// class PyWeightBase : public WeightBase {
// public:

// };

bool add_arc(MutableFstClass &self, int64 from, int64 to,
             int64 input_label, int64 output_label, py::object weight) {
  // TODO: check if the weight is the correct python instance
  if(weight.is_none()) {
    throw fsterror("weight can not be None");
  }
  FSTWeight w1(weight);
  WeightClass w2(w1);
  ArcClass a(input_label, output_label, w2, to);
  // FSTWeight w(weight);
  return self.AddArc(from, a);
}

void set_final(MutableFstClass &self, int64 state, py::object weight) {
   if(weight.is_none()) {
    throw fsterror("weight can not be None");
  }
  FSTWeight w1(weight);
  WeightClass w2(w1);
  self.SetFinal(state, w2);
}

py::object final_weight(MutableFstClass &self, int64 state) {
  return self.GetMutableFst<Arc>()->Final(state).impl;
}

PYBIND11_MODULE(openfst_wrapper_backend, m) {
  m.doc() = "Backing wrapper for OpenFST";

  py::class_<MutableFstClass>(m, "FSTBase")
    .def(py::init<>(&create_fst))
#define d(name) .def(#name, &MutableFstClass:: name)
    .def("AddArc", &add_arc, py::keep_alive<1, 6>()) // keep the weight alive when added
    //d(AddArc)
    d(AddState)
    // there are more than one method with this name but different type signatures
    .def("DeleteArcs", [](MutableFstClass* m, int64 v) { if(m) m->DeleteArcs(v); })
    .def("DeleteStates", [](MutableFstClass *m) { if(m) m->DeleteStates(); })

    d(NumStates)
    d(ReserveArcs)
    d(ReserveStates)
    ///d(SetInputSymbols)
    //d(SetOutputSymbols)

    //d(SetFinal)
    .def("SetFinal", &set_final)
    d(SetStart)


    d(Final)
    d(NumArcs)
    d(Start)

    .def("FinalWeight", &final_weight)

    ;
#undef d

}
