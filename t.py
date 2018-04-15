print("here")

import mfst

num_weights = 0

class MyWeight(mfst.AbstractSemiringWeight):

    def __init__(self, v=0):
        global num_weights
        super().__init__()
        print('create', v)
        self._value = v
        num_weights += 1

    def __add__(self, other):
        print('hi from add')
        return MyWeight(self._value + other._value)

    def __mul__(self, other):
        print('hi from mul')
        return MyWeight(self._value * other._value)

    def __eq__(self, other):
        print('hi from equal')
        return self._value == other._value

    def __hash__(self, other):
        print('hi from hash')
        return hash(self._value)

    def approx_eq(self, other, delta):
        print('hi from approx eq')
        return self == other

    def __del__(self):
        global num_weights
        print('del', type(self), self._value)
        num_weights -= 1

    def __str__(self):
        print('calling string')#, type(self), dir(self))
        return 'MyWeight({})'.format(self._value)

MyWeight.zero = MyWeight(0)
MyWeight.one = MyWeight(1)


gg = mfst.FST(MyWeight)

for i in range(9):
    gg.add_state()


for i in range(5):
    gg.add_arc(i, i+1, weight=MyWeight(i))

gg.initial_state = 0
gg.set_final_weight(3, MyWeight(1))

print("before final")

vv = gg.get_final_weight(3)
print(type(vv), dir(vv))
print(str(vv))

print(gg)

print("after")


# try the isomorphics to itself
assert gg.isomorphic(gg)

for a in gg.get_arcs(3):
    print(a)
a = None

vv = None
gg = None

assert num_weights == 2 # check that everything got deleted (except the zero and 1 static)


gg = mfst.FST()
sf = gg.create_from_string('test')
print(sf)
assert sf.get_unique_output_string() == 'test'

assert sf._repr_html_() is not None
print(sf._repr_html_())


def build_hc_fst(fst, n=3):
    start = fst.add_state()
    fst.initial_state = start
    h = [fst.add_state() for i in range(n)]
    c = [fst.add_state() for i in range(n)]
    fst.add_arc(start, h[0], input_label='H')
    fst.add_arc(start, c[0], input_label='C')
    for i in range(n - 1):
        fst.add_arc(h[i], h[i+1], input_label='H')
        fst.add_arc(c[i], c[i+1], input_label='C', weight=fst.semiring(2))
        fst.add_arc(h[i], c[i+1], input_label='C')
        fst.add_arc(c[i], h[i+1], input_label='H')
    final = fst.add_state()
    fst.set_final_weight(final, 1)
    fst.add_arc(c[-1], final)
    fst.add_arc(h[-1], final)

class pathSemiring2(mfst.PythonValueSemiringWeight):
    semiring_properties = 'path'

pathSemiring2.zero = pathSemiring2(0)
pathSemiring2.one = pathSemiring2(1)
pathSemiring2.one.zzz = 1
pathSemiring2.zero.zzz = 1



hc = mfst.FST(pathSemiring2)
build_hc_fst(hc)
print(hc.determinize())

hc.random_path()

fst = mfst.FST()
f1 = fst.create_from_string('hello')
f2 = fst.create_from_string('world')

g = f1.union(f2)
print(g.full_str())
g._repr_html_()

fst = mfst.FST(acceptor=True)
f1 = fst.create_from_string('hello')
f2 = fst.create_from_string('world')
g = f1.intersect(f2)
print(g.full_str())
g._repr_html_()

fst.random_path()
f1.num_arcs(0)

assert hc.shortest_path().num_states > 0


# check that we can use the other semirings that are defined in

from mfst import semirings

for s in dir(semirings):
    if s.startswith('_') or s == 'AbstractSemiringWeight' or s == 'BooleanSemiringWeight':
        continue
    ss = getattr(semirings, s)

    fst = mfst.FST(ss)
    build_hc_fst(fst)
    fst.determinize()  # just do some operation to check that the semiring seems ok

mfst.FST(semirings.BooleanSemiringWeight).create_from_string('test123').push()


print('done')
