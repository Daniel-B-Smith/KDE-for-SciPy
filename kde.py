"""
An implementation of the kde bandwidth selection method outlined in:

Z. I. Botev, J. F. Grotowski, and D. P. Kroese. Kernel density
estimation via diffusion. The Annals of Statistics, 38(5):2916-2957, 2010.

Based on the implementation in Matlab by Zdravko Botev.

Daniel Smith, PhD
Updated 1-23-2013
"""

from __future__ import division

import numpy as np
import scipy.optimize
import scipy.fftpack

def kde(data, N=2**14, MIN=None, MAX=None):

    # Parameters to set up the mesh on which to calculate
    N = 2**np.ceil(np.log2(N))
    if MIN is None or MAX is None:
        minimum = np.min(data)
        maximum = np.max(data)
        Range = maximum - minimum
        MIN = minimum - Range/10 if MIN is None else MIN
        MAX = maximum + Range/10 if MAX is None else MAX

    # Range of the data
    R = MAX-MIN

    # Histogram the data to get a crude first approximation of the density
    M = len(data)
    DataHist, bins = np.histogram(data,bins=N,range=(MIN,MAX))
    DataHist = DataHist/M
    DCTData = scipy.fftpack.dct(DataHist, norm='ortho')

    I = (np.arange(N-1)+1)**2
    SqDCTData = (DCTData[1:]/2)**2

    # The fixed point calculation finds the bandwidth = t_star
    t_star = scipy.optimize.brentq(fixed_point,0,0.1,args=(M,I,SqDCTData))

    # Smooth the DCTransformed data using t_star
    SmDCTData = DCTData*np.exp(-np.arange(N)**2*np.pi**2*t_star/2)
    # Inverse DCT to get density
    density = scipy.fftpack.idct(SmDCTData, norm='ortho')*N/R
    mesh = [(bins[i]+bins[i+1])/2 for i in xrange(N)]
    bandwidth = np.sqrt(t_star)*R
    
    return bandwidth, mesh, density

def fixed_point(t, M, I, a2):
    l=7
    I = np.float128(I)
    M = np.float128(M)
    a2 = np.float128(a2)
    f = 2*np.pi**(2*l)*np.sum(I**7*a2*np.exp(-I*np.pi**2*t))
    for s in range(l, 1, -1):
        K0 = np.prod(xrange(1, 2*s, 2))/np.sqrt(2*np.pi)
        const = (1 + (1/2)**(s + 1/2))/3
        time=(2*const*K0/M/f)**(2/(3+2*s))
        f=2*np.pi**(2*s)*np.sum(I**s*a2*np.exp(-I*np.pi**2*time))
    return t-(2*M*np.sqrt(np.pi)*f)**(-2/5)
