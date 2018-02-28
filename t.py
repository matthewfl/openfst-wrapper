print("here")

import mfst

gg = mfst.FST()

for i in range(10):
    gg.AddState()

class MyWeight(mfst.Weight):

    def __init__(self, v=0):
        super().__init__()
        print('create', v)
        self._value = v

    def add(self, other):
        print('hi from add')
        return super().add(other)

    def multiply(self, other):
        print('hi from mul')
        return super().multiply(other)

    def equal(self, other):
        print('hi from equal')
        return super(self).equal(other)

    def __del__(self):
        print('del', type(self), self._value)

    def __str__(self):
        print('calling string', type(self), dir(self))
        return str(self._value)

for i in range(5):
    gg.AddArc(i,i+5,-1,-1, MyWeight(i))

gg.SetStart(0)
gg.SetFinal(3, MyWeight())

print("before final")

vv = gg.FinalWeight(3)
print(type(vv), dir(vv))
print(str(vv))

print("after")
