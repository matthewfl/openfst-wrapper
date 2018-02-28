print("here")

import mfst

gg = mfst.FST()

for i in range(10):
    gg.AddState()

class MyWeight(mfst.Weight):

    def add(self, other):
        print('hi from add')
        return super().add(other)

    def multiply(self, other):
        print('hi from mul')
        return super().multiply(other)

    def __del__(self):
        print('del')

for i in range(5):
    gg.AddArc(i,i+5,-1,-1, mfst.Weight())

gg.SetStart(0)
gg.SetFinal(3, mfst.Weight())

print("before final")

gg.FinalWeight(3)
#print(gg.FinalWeight(3))

print("after")
