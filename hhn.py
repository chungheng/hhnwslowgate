import sys
import numpy as np
from progressbar import *
from numpy import random
import pdb
s_a = 10
s_b = 200
a_r = 5.
a_d = 6.


n_a = lambda v: (10.-v)/(100.*(np.exp((10.-v)/10.)-1.))
n_b = lambda v: 0.125*np.exp(-v/80.)
m_a = lambda v: (25.-v)/(10.*(np.exp((25.-v)/10.)-1.))
m_b = lambda v: 4.*np.exp(-v/18.)
h_a = lambda v: 0.07*np.exp(-v/20.)
h_b = lambda v: 1./(np.exp((30.-v)/10.)+1.)
_update_sg = lambda sg,diff_c,dt: (1.-s_a*dt)*sg + s_a*dt + s_b*diff_c 


def compute_psth(t,s,interval=0.025,binSize=0.1):
    """
    Compute the psth of spike sequence S with time course t
    
    interval: float
        sample interval
    bin: float
        bin size
        
    """
    dt = t[1]-t[0]
    t_ds  = np.arange(t[0],t[-1]+interval,interval)
    s_idx = np.nonzero(s)[0]
    freq  = np.zeros_like(t_ds)
    off = int(round(binSize/interval))
    for spk in s_idx:
        idx = int(dt*spk/interval)
        freq[ max(0,idx-off+1):idx+1 ] += 1
    freq /= binSize    
    return t_ds, freq

def hodgkin_huxley_slowChannel(t,I_ext,noise=None):
    """
    Modified Hodgkin-Huxley Neuron with Potassian, Sodium, leaky channel, and
    slow channel.
    
    s" + (a_r+a_d)s' + a_r*a_d*s = a_r * a_d * dc
     
    Original written by Lev Givon and Robert Turetsky in Matlab.
    """
    
    t = np.array(t)*1000
        
    V    = np.zeros_like(t)
    V[0] = -10.
    spk  = np.zeros_like(t)
    
    E   = np.array([-12., 115., 10.613])
    g   = np.array([ 36., 120.,  0.300])
    nmh = np.array([  0.,   0.,  1.000])
    
    I = np.zeros((3,len(t)))
    gnmh = np.zeros(3)

    s  = np.zeros_like(t)
    sc = np.zeros(3)
    if sys.flags.debug:
        pbar = ProgressBar(maxval=len(t))
        pbar.start()
    for i in xrange(1,len(t)):
        if sys.flags.debug: pbar.update(i) 
        dt = t[i]-t[i-1]
        
        _update_nmh(nmh,V[i-1],dt)
        nmh += 0.0 if noise == None else random.rand(3)*noise**2
        
        _update_slowChannel(sc,dt,(I_ext[i]-I_ext[i-1])/dt)
        s[i] = sc[0]
        
        gnmh[0] = g[0]*nmh[0]**4
        gnmh[1] = g[1]*nmh[1]**3*nmh[2]
        gnmh[2] = g[2]

        # Update the ionic currents and membrane voltage:
        I[:,i] = gnmh*(V[i-1]-E)
        V[i]   = V[i-1] + dt*(-np.sum(I[:,i])+I_ext[i]+15*(s[i]+2))
        
    if sys.flags.debug: pbar.finish()
    for i in xrange(2,len(t)):
        spk[i-1] = _spike_detect(V[i-2],V[i-1],V[i])
    return V, spk, s
    

def hodgkin_huxley_slowGate(t,I_ext,noise=None):
    """
    Standard Hodgkin-Huxley Neuron with Potassian, Sodium, and leaky channel. 
    Original written by Lev Givon and Robert Turetsky in Matlab.
    """
    
    t = np.array(t)*1000
        
    V = np.zeros_like(t)
    s = np.zeros_like(t)
    V[0] = -10.
    
    E   = np.array([-12., 115., 10.613])
    g   = np.array([ 36., 120.,  0.300])
    nmh = np.array([  0.,   0.,  1.000])
    
    I = np.zeros((3,len(t)))
    gnmh = np.zeros(3)

    sg = np.zeros_like(t)
    sg[0] = 1.
    if sys.flags.debug:
        pbar = ProgressBar(maxval=len(t))
        pbar.start()
    for i in xrange(1,len(t)):
        if sys.flags.debug: pbar.update(i) 
        dt = t[i]-t[i-1]
        
        _update_nmh(nmh,V[i-1],dt)
        nmh += 0.0 if noise == None else random.rand(3)*noise**2
        
        sg[i] = _update_sg(sg[i-1],I_ext[i]-I_ext[i-1],dt)
        
        gnmh[0] = g[0]*nmh[0]**4
        gnmh[1] = g[1]*nmh[1]**3*nmh[2]*sg[i]
        gnmh[2] = g[2]

        # Update the ionic currents and membrane voltage:
        I[:,i] = gnmh*(V[i-1]-E)
        V[i]   = V[i-1] + dt*(I_ext[i]-np.sum(I[:,i]))
        
    if sys.flags.debug: pbar.finish()
    for i in xrange(2,len(t)):
        s[i-1] = _spike_detect(V[i-2],V[i-1],V[i])
    return V, s, sg

def hodgkin_huxley(t,I_ext,noise=None):
    """
    Standard Hodgkin-Huxley Neuron with Potassian, Sodium, and leaky channel. 
    Original written by Lev Givon and Robert Turetsky in Matlab.
    """
    
    t = np.array(t)*1000
        
    V = np.zeros_like(t)
    s = np.zeros_like(t)
    V[0] = -10.
    
    E   = np.array([-12., 115., 10.613])
    g   = np.array([ 36., 120.,  0.300])
    nmh = np.array([  0.,   0.,  1.000])
    
    I = np.zeros((3,len(t)))
    gnmh = np.zeros(3)

    if sys.flags.debug:
        pbar = ProgressBar(maxval=len(t))
        pbar.start()
    for i in xrange(1,len(t)):
        if sys.flags.debug: pbar.update(i) 
        dt = t[i]-t[i-1]
        
        _update_nmh(nmh,V[i-1],dt)
        nmh += 0.0 if noise == None else random.rand(3)*noise**2
    
        gnmh[0] = g[0]*nmh[0]**4
        gnmh[1] = g[1]*nmh[1]**3*nmh[2]
        gnmh[2] = g[2]

        # Update the ionic currents and membrane voltage:
        I[:,i] = gnmh*(V[i-1]-E)
        V[i]   = V[i-1] + dt*(I_ext[i]-np.sum(I[:,i]))
        
    if sys.flags.debug: pbar.finish()
    for i in xrange(2,len(t)):
        s[i-1] = _spike_detect(V[i-2],V[i-1],V[i])
    return V, s

def _update_slowChannel(sc,dt,dc):
    sc_new = np.zeros_like(sc)
    sc_new[0] = sc[0] + dt*sc[1]
    sc_new[1] = sc[1] + dt*sc[2] + a_r*a_d*dc
    sc_new[2] = -a_r*a_d*sc[0] - (a_r+a_d)*sc[1]
    sc[:] = sc_new

def _spike_detect(v_pre,v_cur,v_post):
    if v_pre <= v_cur and v_cur >= v_post:
        return 1
    else:
        return 0

def _update_nmh(x,v,dt):
    a = np.zeros(3)
    b = np.zeros(3)
    a[0] = n_a(v)
    a[1] = m_a(v)
    a[2] = h_a(v)
    
    b[0] = n_b(v)
    b[1] = m_b(v)
    b[2] = h_b(v)
    tau = 1. / (a+b)
    x0 = a*tau;
    x[:] =  (1.-dt/tau)*x + dt/tau*x0