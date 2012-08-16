from hhn import hodgkin_huxley
import numpy as np
#import matplotlib as mp
from matplotlib import pyplot as p


dt = 3e-5
t  = np.arange(0,1,dt)
I  = np.ones_like(t)*15
    
V, spk = hodgkin_huxley(t,I)

p.figure()
p.plot(t,V)
#p.savefig("test.png")
p.show()