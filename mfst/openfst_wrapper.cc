#include <memory>
#include <string>
#include <sstream>

#undef NDEBUG
#include <pybind11/pybind11.h>

#include <fst/fst.h>
#include <fst/mutable-fst.h>
#include <fst/arc.h>
#include <fst/script/fst-class.h>
#include <fst/script/arc-class.h>
#include <fst/script/register.h>

#include <assert.h>


using namespace std;
using namespace fst;
using namespace fst::script;
namespace py = pybind11;

namespace fst { namespace script {
    void __mfl_hack_VectorFSTRegister(string name, IORegistration<VectorFstClass>::Entry* entry);
  } }


class fsterror : public exception {
private:
  const char* error;
public:
  fsterror(const char *e): error(e) {}
  virtual const char* what() { return error; }
};

class FSTWeight {
  FSTWeight(int32 g) : impl(), flags(g) {} // temp to debug the creation
public:

  enum {
    isZero = 0x1,
    isOne = 0x2,
    isNoWeight = 0x4,
    isSet = 0x8
  };

  // the python object that we are wrapping
  int16_t flags;
  //int16_t one_sum_count = 0;

  //int32_t created = 0;
  py::object impl;

  FSTWeight() : impl() {
    //py::handle h(nullptr);
    //impl = py::object(h), false;
    flags = isNoWeight;
  }

  FSTWeight(py::object i) : impl(i) {
    if(impl.is_none())
      throw fsterror("Can not have a None edge weight");
    //impl.inc_ref();
    flags = isSet;
    //created = 0xdeadbeef;
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
  static const uint64 Properties () { return kSemiring | kCommutative; }

  bool Member() const {
    py::object r = impl.attr("_member")();
    return r.cast<bool>();
  }

  // do nothing?
  FSTWeight Reverse() { return *this; }

  FSTWeight Quantize(float delta = kDelta) const {
    return FSTWeight(impl.attr("_quantize")(delta));
  }

  std::istream &Read(std::istream &strm) const { throw fsterror("not implemented"); }
  std::ostream &Write(std::ostream &strm) const { throw fsterror("not implemented"); }

  size_t Hash() const {
    py::object r = impl.attr("__hash__")();
    return r.cast<size_t>();
  }

  FSTWeight& operator=(const FSTWeight& other) {
    // py::handle h(other.impl);
    // h.inc_ref();
    // impl.dec_ref();
    impl = other.impl;
    flags = other.flags;
  }

  virtual ~FSTWeight() {
    cout << "delete weight\n";
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

std::ostream &operator<<(std::ostream &os, const FSTWeight &w) {
  //return os << py::str(w.impl.attr(u8"__str__")());
  return os << "the python type\n";
}

std::istream &operator>>(std::istream &is, const FSTWeight &w) {
  throw fsterror("python weights do not support loading");
}


using PyArc = ArcTpl<FSTWeight>;

static FstRegister < PyArc > our_weights;

struct register_pyweight {
  register_pyweight() {
    __mfl_hack_VectorFSTRegister(PyArc::Type(),
    new IORegistration<VectorFstClass>::Entry(
                                          VectorFstClass::Read<PyArc>,
                                          VectorFstClass::Create<PyArc>,
                                          VectorFstClass::Convert<PyArc>
                                              ));
  }
};
static register_pyweight _register_pyweight;

MutableFstClass* create_fst() {

  // register the arc type class as can't get this to work normally
  // we might be getting loaded before the other shared object, which I guess is causing us to be unable
  // to properly register or something????

  // REGISTER_FST_WEIGHT(FSTWeight);
  // REGISTER_FST_CLASSES(PyArc);


  MutableFstClass* r = new VectorFstClass(PyArc::Type());
  return r;

}

bool check_is_weight(py::object &weight) {
  return
    hasattr(weight, "__add__") &&
    hasattr(weight, "__mul__") &&
    hasattr(weight, "__div__") &&
    hasattr(weight, "__pow__") &&
    hasattr(weight, "__hash__") &&
    hasattr(weight, "__eq__") &&
    hasattr(weight, "__str__") &&
    hasattr(weight, "_member") &&
    hasattr(weight, "_quantize");
  // zero & one ??
}

bool add_arc(MutableFstClass &self, int64 from, int64 to,
             int64 input_label, int64 output_label, py::object weight) {
  // TODO: check if the weight is the correct python instance

  if(weight.is_none()) {
    throw fsterror("weight can not be None");
  }
  if(!check_is_weight(weight)) {
    throw fsterror("weight is missing required method");
  }
  // weight.inc_ref();
  // weight.inc_ref();
  // weight.inc_ref();
  // weight.inc_ref();
  // weight.inc_ref();
  // weight.inc_ref();


  FSTWeight w1(weight);
  WeightClass w2(w1);
  ArcClass a(input_label, output_label, w2, to);
  // FSTWeight w(weight);
  cout << "before add\n";
  bool ret = self.AddArc(from, a);
  cout << "after add\n";
  return ret;
}

void set_final(MutableFstClass &self, int64 state, py::object weight) {
  if(weight.is_none()) {
    throw fsterror("weight can not be None");
  }
  if(!check_is_weight(weight)) {
    throw fsterror("weight is missing required method");
  }
  // weight.inc_ref();
  // weight.inc_ref();
  // weight.inc_ref();
  // weight.inc_ref();
  // weight.inc_ref();

  //cout << weight;
  FSTWeight w1(weight);
  WeightClass w2(w1);
  self.SetFinal(state, w2);
}

py::object final_weight(MutableFstClass &self, int64 state) {
  //MutableFstClass self = pself.cast<MutableFstClass*>();
  FSTWeight finalW = self.GetMutableFst<PyArc>()->Final(state);
  if(finalW.isBuiltIn()) {
    if(finalW.flags == FSTWeight::isZero) {
      return py::cast(0);
    } else if(finalW.flags == FSTWeight::isOne) {
      return py::cast(1);
    } else {
      // invalid
      return py::cast("FST INVALID");
    }
  }

  assert(finalW.flags == FSTWeight::isSet);

  py::object r =  finalW.impl; // this should still be holding onto the handle
  //assert(finalW.created == 0xdeadbeef);

  // r.inc_ref();
  // r.inc_ref();
  // r.inc_ref();
  // r.inc_ref();
  // r.inc_ref();
  // r.inc_ref();
  // r.inc_ref();
  // r.inc_ref();


  // py::handle s = r.attr("__str__");
  // string ss = s().cast<string>();
  // cout << ss;
  return r;
}

PYBIND11_MODULE(openfst_wrapper_backend, m) {
  m.doc() = "Backing wrapper for OpenFST";

  py::class_<MutableFstClass>(m, "FSTBase")
    .def(py::init<>(&create_fst))
#define d(name) .def("_" #name, &MutableFstClass:: name)
    .def("_AddArc", &add_arc) // keep the weight alive when added
    //d(AddArc)
    d(AddState)
    // there are more than one method with this name but different type signatures
    .def("_DeleteArcs", [](MutableFstClass* m, int64 v) { if(m) m->DeleteArcs(v); })
    .def("_DeleteStates", [](MutableFstClass *m) { if(m) m->DeleteStates(); })

    d(NumStates)
    d(ReserveArcs)
    d(ReserveStates)
    ///d(SetInputSymbols)
    //d(SetOutputSymbols)

    //d(SetFinal)
    .def("_SetFinal", &set_final)
    d(SetStart)

    d(NumArcs)
    d(Start)

    .def("_FinalWeight", &final_weight)

    ;
#undef d

}
