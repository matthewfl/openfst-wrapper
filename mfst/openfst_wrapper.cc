#include <memory>
#include <string>
#include <sstream>

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

class fsterror : public exception {
private:
  const char* error;
public:
  fsterror(const char *e): error(e) {}
  virtual const char* what() { return error; }
};

class FSTWeight {
  FSTWeight(int32 g) : impl(nullptr), created(g) {} // temp to debug the creation
public:

  // the python object that we are wrapping
  int32_t created = 0;
  py::handle impl;

  FSTWeight() : impl(nullptr) {}

  FSTWeight(py::object i) : impl(i) {
    if(impl.is_none())
      throw fsterror("Can not have a None edge weight");
    impl.inc_ref();
    created = 0xdeadbeef;
  }

  static const FSTWeight& Zero() {
    static const FSTWeight zero = FSTWeight(123);
    return zero;
  }
  static const FSTWeight& One() {
    static const FSTWeight one = FSTWeight(456);
    return one;
  }
  static const FSTWeight& NoWeight() {
    static const FSTWeight no_weight = FSTWeight(789);
    return no_weight;
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
    py::object r = impl.attr("member")();
    return r.cast<bool>();
  }

  // do nothing?
  FSTWeight Reverse() { return *this; }

  FSTWeight Quantize(float delta = kDelta) const {
    return FSTWeight(impl.attr("quantize")(delta));
  }

  std::istream &Read(std::istream &strm) const { throw fsterror("not implemented"); }
  std::ostream &Write(std::ostream &strm) const { throw fsterror("not implemented"); }

  size_t Hash() const {
    py::object r = impl.attr("__hash__")();
    return r.cast<size_t>();
  }

  virtual ~FSTWeight() {
    impl.dec_ref();
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

MutableFstClass* create_fst() {

  // register the arc type class as can't get this to work normally
  // we might be getting loaded before the other shared object, which I guess is causing us to be unable
  // to properly register or something????

  // REGISTER_FST_WEIGHT(FSTWeight);
  // REGISTER_FST_CLASSES(PyArc);

  static auto *io_register =
    IORegistration<VectorFstClass>::Register::GetRegister();
  io_register->SetEntry(PyArc::Type(),
                        IORegistration<VectorFstClass>::Entry(
                                                              VectorFstClass::Read<PyArc>,
                                                              VectorFstClass::Create<PyArc>,
                                                              VectorFstClass::Convert<PyArc>));


  MutableFstClass* r = new VectorFstClass(PyArc::Type());
  return r;

}

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
  //cout << weight;
  FSTWeight w1(weight);
  WeightClass w2(w1);
  self.SetFinal(state, w2);
}

void final_weight(MutableFstClass &self, int64 state) {
  FSTWeight finalW = self.GetMutableFst<PyArc>()->Final(state);
  py::handle r =  finalW.impl;
  string s = r.attr("__str__")().cast<string>();
  cout << s;
  //return r;
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
