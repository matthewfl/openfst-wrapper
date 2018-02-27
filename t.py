print("here")

import mfst

gg = mfst.FST()

for i in range(10):
    gg.AddState()

class MyWeight(mfst.Weight):

    def add(self, other):
        print('hi from add')
        return super().add(other)

for i in range(5):
    gg.AddArc(i,i+5,-1,-1, mfst.Weight())

gg.SetStart(0)
gg.SetFinal(3, mfst.Weight())

print(gg.FinalWeight(3))
