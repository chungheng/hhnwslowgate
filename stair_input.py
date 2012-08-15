# Duplicate the result 
import numpy as np
from hhn import hodgkin_huxley
from matplotlib import pyplot as p

dt = 1e-5


initial_period   = 8.
transient_period = 0.3
transient_length = round(transient_period/dt)
interval_period  = 2.

hamming_window = np.hanning(transient_length*2)[:transient_length]
level = np.array([10.,40.,70.,100.,70.,40.,10.])


t = np.arange(0,initial_period,dt)
I = np.ones_like(t)*level[0]

for y,x in zip(level[:-1],level[1:]):
    seg_t = t[-1]+dt+np.arange(0,interval_period,dt)
    seg_I = np.ones_like(seg_t)*x
    if x > y:
        seg_I[:transient_length] = y + (x-y)*hamming_window
    else:
        seg_I[:transient_length] = x + (y-x)*(1.-hamming_window)
        
    t = np.concatenate((t,seg_t))
    I = np.concatenate((I,seg_I))
    

V = hodgkin_huxley(t,I)


fig = p.figure()
ax = fig.add_subplot(2,1,1,
                     title="Staircase Waveform",
                     xlim = (6,20),ylim=(0,120))
ax.plot(t,I)
ax = fig.add_subplot(2,1,2,
                     title="Membrane Potential",
                     xlim = (6,20))
ax.plot(t,V)
p.show()
