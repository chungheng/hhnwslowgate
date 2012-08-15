from hhn import hodgkin_huxley
import numpy as np
import matplotlib as mp
import matplotlib.pyplot as p


dt = 1e-5
t  = np.arange(0,1,dt)
I  = np.ones_like(t)*1
    
V = hodgkin_huxley(t,I)
p.figure()
p.plot(t,V)
p.savefig("test.png")
