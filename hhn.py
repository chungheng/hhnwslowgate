import numpy as np
from progressbar import *
from numpy import random

n_a = lambda v: (10.-v)/(100.*(np.exp((10.-v)/10.)-1.))
n_b = lambda v: 0.125*np.exp(-v/80.)
m_a = lambda v: (25.-v)/(10.*(np.exp((25.-v)/10.)-1.))
m_b = lambda v: 4.*np.exp(-v/18.)
h_a = lambda v: 0.07*np.exp(-v/20.)
h_b = lambda v: 1./(np.exp((30.-v)/10.)+1.)

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

def _spike_detect(v_pre,v_cur,v_post):
    if v_pre <= v_cur and v_cur >= v_post:
        return 1
    else:
        return 0

def hodgkin_huxley(t,I_ext,noise=None):
    """
    Standard Hodgkin-Huxley Neuron with Potassian, Sodium, and leaky channel. 
    Original written by Lev Givon and Robert Turetsky in Matlab.
    """
    
    t = np.array(t)*1000
    dt = t[1]-t[0]

    E = np.array([-12., 115., 10.613])
    g = np.array([ 36., 120.,  0.300])
    x = np.array([  0.,   0.,  1.000])
    
    V = np.zeros_like(t)
    s = np.zeros_like(t)
    V[0] = -10.
    
    I = np.zeros((3,len(t)))
    
    a = np.zeros(3)
    b = np.zeros(3)
    
    gnmh = np.zeros(3)
    pbar = ProgressBar(maxval=len(t))
    pbar.start()
    for i in xrange(1,len(t)):
        #pbar.update(i)
        a[0] = n_a(V[i-1])*(1+0.05*random.rand())
        a[1] = m_a(V[i-1])*(1+0.05*random.rand())
        a[2] = h_a(V[i-1])*(1+0.05*random.rand())
    
        b[0] = n_b(V[i-1])
        b[1] = m_b(V[i-1])
        b[2] = h_b(V[i-1])
    
        tau = 1. / (a+b)
        x0 = a*tau;
    
        x = (1.-dt/tau)*x + dt/tau*x0
    
        gnmh[0] = g[0]*x[0]**4;
        gnmh[1] = g[1]*x[1]**3*x[2];
        gnmh[2] = g[2]

        # Update the ionic currents and membrane voltage:
        I[:,i] = gnmh*(V[i-1]-E*(1+0.05*random.rand(3)))
        V[i]   = V[i-1] + dt*(0.05*random.rand()*I_ext[i]-np.sum(I[:,i]))
    pbar.finish()
    for i in xrange(2,len(t)):
        s[i-1] = _spike_detect(V[i-2],V[i-1],V[i])
    return V, s
