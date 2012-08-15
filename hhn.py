import numpy as np

def hodgkin_huxley(t,I_ext):
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
    V[0] = -10.
    
    I = np.zeros((3,len(t)))
    
    a = np.zeros(3)
    b = np.zeros(3)
    
    gnmh = np.zeros(3)
    
    for i in xrange(1,len(t)):
        a[0] = (10.-V[i-1])/(100.*(np.exp((10.-V[i-1])/10.)-1.))
        a[1] = (25.-V[i-1])/(10.*(np.exp((25.-V[i-1])/10.)-1.))
        a[2] = 0.07*np.exp(-V[i-1]/20.)
    
        b[0] = 0.125*np.exp(-V[i-1]/80.)
        b[1] = 4.*np.exp(-V[i-1]/18.);
        b[2] = 1./(np.exp((30.-V[i-1])/10.)+1.);
    
        tau = 1. / (a+b)
        x0 = a*tau;
    
        x = (1.-dt/tau)*x + dt/tau*x0
    
        gnmh[0] = g[0]*x[0]**4;
        gnmh[1] = g[1]*x[1]**3*x[2];
        gnmh[2] = g[2]

        # Update the ionic currents and membrane voltage:
        I[:,i] = gnmh*(V[i-1]-E)
        V[i]   = V[i-1] + dt*(I_ext[i]-np.sum(I[:,i]))
    return V
