# Duplicate the result 
import numpy as np
from hhn import hodgkin_huxley, compute_psth, hodgkin_huxley_slowGate
from matplotlib import pyplot as p

dt = 3e-5

# Prepare staircase input waveform
# ================================
initial_period   = 2.
transient_period = 0.3
transient_length = round(transient_period/dt)
interval_period  = 2.

hamming_window = np.hanning(transient_length*2)[:transient_length]
level = np.array([10.,25.,40.,55.,40.,25.,10.])


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


# Run simulation
# ==============
trial = 1
noise = 1e-4    
V, spk, sg = hodgkin_huxley_slowGate(t,I,noise)
t_psth, psth = compute_psth(t,spk)

for i in xrange(1,trial):
    V, spk, sg = hodgkin_huxley_slowGate(t,I,noise)
    t_psth, temp = compute_psth(t,spk)
    psth += temp
psth /= float(trial)

fig = p.figure(figsize=(7,4))
ax = fig.add_subplot(4,1,1,
                     title="Staircase Waveform",
                     xticklabels=[],
                     xlim = (0,7),ylim=(0,120))
ax.plot(t,I)
ax = fig.add_subplot(4,1,2,
                     title="PSTH",
                     xlabel="time, sec",
                     xlim = (0,7))
ax.plot(t_psth,psth)
ax = fig.add_subplot(4,1,3,
                     
                     title="s",
                     xlim = (0,7))
ax.plot(t,sg)
ax = fig.add_subplot(4,1,4,
                     xlabel="time, sec",
                     title="s",
                     xlim = (0,7))
ax.plot(t,V)
p.show()
