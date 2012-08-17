import hhn
from hhn import *
import numpy as np


from matplotlib import pyplot as p


dt = 3e-5
t  = np.arange(0,1,dt)
I  = np.zeros_like(t)
I[500] = 1
sc = np.zeros(3)
s  = np.ones_like(t)
for i in xrange(1,len(t)):
    hhn._update_slowChannel(sc,dt,(I[i]-I[i-1])/dt)
    s[i] = sc[0]

p.figure()
p.plot(t,s)
#p.savefig("test.png")
p.show()