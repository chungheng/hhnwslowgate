import hhn
from hhn import *
import numpy as np


from matplotlib import pyplot as p


dt = 3e-5
t  = np.arange(0,1,dt)
I  = np.zeros_like(t)
I[500:] = 1
s  = np.ones_like(t)
for i in xrange(1,len(t)):
    s[i] = hhn._update_sg(s[i-1],I[i]-I[i-1],dt)


p.figure()
p.plot(t,s)
#p.savefig("test.png")
p.show()
