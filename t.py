print("here")

import mfst

num_weights = 0

class MyWeight(mfst.WeightBase):

    def __init__(self, v=0):
        global num_weights
        super().__init__()
        print('create', v)
        self._value = v
        num_weights += 1

    def __add__(self, other):
        print('hi from add')
        return super().add(other)

    def __mul__(self, other):
        print('hi from mul')
        return super().multiply(other)

    def __eq__(self, other):
        print('hi from equal')
        return super(self).equal(other)

    def _approx_eq(self, other, delta):
        print('hi from approx eq')
        return self == other

    def __del__(self):
        global num_weights
        print('del', type(self), self._value)
        num_weights -= 1

    def __str__(self):
        print('calling string', type(self), dir(self))
        return str(self._value)

gg = mfst.FST(MyWeight)

for i in range(10):
    gg.add_state()


for i in range(5):
    gg.add_arc(i, i+1, weight=MyWeight(i))

gg.start_state = 0
gg.set_final_weight(3, MyWeight(1))

print("before final")

vv = gg.get_final_weight(3)
print(type(vv), dir(vv))
print(str(vv))

print("after")


# try the isomorphics to itself
gg.isomorphic(gg)


vv = None
gg = None

print(num_weights)
