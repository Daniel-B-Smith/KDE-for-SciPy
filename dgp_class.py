"""
This file creates the data generating functions necessary to reproduce Table 1
from:

Z. I. Botev, J. F. Grotowski, and D. P. Kroese. Kernel density
estimation via diffusion. The Annals of Statistics, 38(5):2916-2957, 2010.

Daniel B. Smith, PhD
1-29-2013
"""

from __future__ import division

import numpy as np
from random import shuffle

rand = np.random

_samp_doc = """Generates random numbers according to a {dist} distribution

Parameters:
----------
size: d1, ..., dn : `n` ints, optional
      The dimensions of the returned array, should be all positive.

{eq}
"""

_pdf_doc = """Calculated probability distribution function according to a 
{dist} distribution.

{eq}
"""

_class_doc = """{class_} class to generate data generating function objects.
    Each object has two methods:

    dgp.sample(size=1): generates array of samples according to shape defined 
                        in size.
    dgp.pdf(mesh):      calculates pdf on the given mesh

    Probability distribution function, N(mu, sigma^2) normal:
      {eq}
    """

# Adapted from scipy:
_NORM_PDF_C = np.sqrt(2*np.pi)
def _norm(x, params):
    x = np.asarray(x)
    return np.exp(-(x-params[0])**2/2.0/params[1]**2) / _NORM_PDF_C / params[1]

def _generate(inputs, counts):
    """
    Generates random samples from a sum of normals based on inputs, counts

    Parameters
    ----------
    inputs : list of (mean, standard deviation) tuples
    counts : list of number of samples to draw from each normal defined in
             inputs
    """
    out = []
    for iC, count in enumerate(counts):
        out.extend(inputs[iC][1]*rand.randn(count)+inputs[iC][0])
    shuffle(out)
    return out

class dgp(object):
    __doc__ = _class_doc.format(class_='Default', eq='N/A')
    def __str__(self):
        # Print first line of documentation
        return self.__doc__.split('\n')[0]
    def sample(self, size=1):
        nsamp = np.prod(size)
        out = self._sample(nsamp)
        return np.resize(out, size)
    def pdf(self, mesh):
        return self._pdf(mesh)
    def _pdf(self, mesh):
        return sum((self._rates[k]*_norm(mesh, input_) for k, input_ in 
                    enumerate(self._inputs)))
    def _sample(self, nsamp):
        counts = rand.multinomial(nsamp, self._rates, size=1)[0]
        return _generate(self._inputs, counts)
    def mesh(self, N=None):
        """
        Generates default mesh
        """
        if N is None:
            N = 2**14
        maxes = [input_[0]+4*input_[1] for input_ in self._inputs]
        mins = [input_[0]-4*input_[1] for input_ in self._inputs]
        min_ = min(mins)
        max_ = max(maxes)
        return np.linspace(min_, max_, num=N)
    _inputs = []
    _rates = []

class Claw(dgp):
    def __init__(self):
        dgp.__init__(self)
        eq = ' 1/2*N(0,1) + sum_{k=0}^4 1/10*N(k/2-1, (1/10)^2)'
        self.sample.__func__.__doc__ = _samp_doc.format(dist='claw', eq=eq)
        self.pdf.__func__.__doc__ = _pdf_doc.format(dist='claw', eq=eq)
        self.__doc__ = _class_doc.format(class_='Claw', eq=eq)
    _inputs = [(0, 1)]
    for k in xrange(5):
        _inputs.append((k/2-1, 1/10))
    _rates = [1/2] + [1/10]*5

class StronglySkewed(dgp):
    def __init__(self):
        dgp.__init__(self)
        eq = 'sum_{k=0}^7 1/8*N(3*((2/3)^k-1), (2/3)^(2k))'
        self.sample.__func__.__doc__ = _samp_doc.format(dist='strongly skewed', 
                                                        eq=eq)
        self.pdf.__func__.__doc__ = _pdf_doc.format(dist='strongly skewed', 
                                                    eq=eq)
        self.__doc__ = _class_doc.format(class_='Strongly Skewed', eq=eq)
        for k in xrange(8):
            self._inputs.append((3*((2/3)**k-1), (2/3)**k))
    _rates = [1/8]*8

class KurtoticUnimodal(dgp):
    def __init__(self):
        dgp.__init__(self)
        eq = '2/3*N(0,1) + 1/3*N(0,(1/10)^2)'        
        dist = 'kurtotic unimodal'
        self.sample.__func__.__doc__ = _samp_doc.format(dist=dist,  eq=eq)
        self.pdf.__func__.__doc__ = _pdf_doc.format(dist=dist, eq=eq)
        self.__doc__ = _class_doc.format(class_='Kurtotic Unimodal', eq=eq)
    _inputs = [(0, 1), (0, 1/10)]
    _rates = [2/3, 1/3]

class DoubleClaw(dgp):
    def __init__(self):
        dgp.__init__(self)
        eq = ('49/100*N(-1, (2/3)^2) + 49/100*N(1, (2/3)^2) + \n' + 
              '    sum_{k=0}^6 1/350*N((k-3)/2, (1/100)^2)')
        self.sample.__func__.__doc__ = _samp_doc.format(dist='double claw', 
                                                        eq=eq)
        self.pdf.__func__.__doc__ = _pdf_doc.format(dist='double claw', eq=eq)
        self.__doc__ = _class_doc.format(class_='Double Claw', eq=eq)
    _inputs = [(-1, 2/3), (1, 2/3)]
    for k in xrange(7):
        _inputs.append(((k-3)/2, 1/100))
    _rates = [49/100]*2+[1/350]*7

class DiscreteComb(dgp):
    def __init__(self):
        dgp.__init__(self)
        eq = ('2/7*sum_{k=0}^2 N((12*k-15/7), (2/7)^2) + \n' +
              '    1/21*sum_{k=8}^10 N(2*k/7, (1/21)^2)')
        self.sample.__func__.__doc__ = _samp_doc.format(dist='discrete comb', 
                                                        eq=eq)
        self.pdf.__func__.__doc__ = _pdf_doc.format(dist='discrete comb', 
                                                    eq=eq)
        self.__doc__ = _class_doc.format(class_='Discrete Comb', eq=eq)
    _inputs = []
    for k in xrange(3):
        _inputs.append(((12*k-15)/7, 2/7))
    for k in xrange(8, 11):
        _inputs.append((2*k/7, 1/21))
    _rates = [2/7]*3 + [1/21]*3

class AsymDoubleClaw(dgp):
    def __init__(self):
        dgp.__init__(self)
        eq = ('46/100*sum_{k=0}^1 N(2*k-1, (2/3)^2) + 1/300*sum_{k=1}^3 ' + 
              'N(-k/2, (1/100)^2)\n    sum_{k=1}^3 N(k/2, (7/100)^2)')
        dist='asymmetric double claw'
        self.sample.__func__.__doc__ = _samp_doc.format(dist=dist, eq=eq)
        self.pdf.__func__.__doc__ = _pdf_doc.format(dist=dist, eq=eq)
        self.__doc__ = _class_doc.format(class_='Asymmetric Double Claw', eq=eq)
    _inputs = []
    for k in xrange(2):
        _inputs.append((2*k-1, 2/3))
    for k in xrange(1, 4):
        _inputs.append((-k/2, 1/100))
    for k in xrange(1, 4):
        _inputs.append((k/2, 7/100))
    _rates = [46/100]*2 + [1/300]*3 + [7/300]*3    

class Outlier(dgp):
    def __init__(self):
        dgp.__init__(self)
        eq = '1/10*N(0, 1)+9/10*N(0, (1/10)^2)'
        self.sample.__func__.__doc__ = _samp_doc.format(dist='outlier', eq=eq)
        self.pdf.__func__.__doc__ = _pdf_doc.format(dist='outlier', eq=eq)
        self.__doc__ = _class_doc.format(class_='Outlier', eq=eq)
    _inputs = [(0, 1), (0, 1/10)]
    _rates = [1/10, 9/10]

class SeparatedBimodal(dgp):
    def __init__(self):
        dgp.__init__(self)
        eq = '1/2*N(-12, (1/2)^2) + 1/2*N(12, (1/2)^2)'
        dist = 'separated bimodal'
        self.sample.__func__.__doc__ = _samp_doc.format(dist=dist, eq=eq)
        self.pdf.__func__.__doc__ = _pdf_doc.format(dist=dist, eq=eq)
        self.__doc__ = _class_doc.format(class_='Separated Bimodal', eq=eq)
    _inputs  = [(-12, 1/2), (12, 1/2)]
    _rates = [1/2]*2

class SkewBimodal(dgp):
    def __init__(self):
        dgp.__init__(self)
        eq = '3/4*N(0, 1) + 1/4*N(3/2, (1/3)^2)'
        dist = 'skewed bimodal'
        self.sample.__func__.__doc__ = _samp_doc.format(dist=dist, eq=eq)
        self.pdf.__func__.__doc__ = _pdf_doc.format(dist=dist, eq=eq)
        self.__doc__ = _class_doc.format(class_='Skewed Bimodal', eq=eq)
    _inputs = [(0,1), (3/2, 1/3)]
    _rates = [3/4, 1/4]

class Bimodal(dgp):
    def __init__(self):
        dgp.__init__(self)
        eq = '1/2*N(0, (1/10)^2) + 1/2*N(5, 1)'
        self.sample.__func__.__doc__ = _samp_doc.format(dist='bimodal', eq=eq)
        self.pdf.__func__.__doc__ = _pdf_doc.format(dist='bimodal', eq=eq)
        self.__doc__ = _class_doc.format(class_='Bimodal', eq=eq)
    _inputs = [(0, 1/10), (5, 1)]
    _rates = [1/2]*2

class LogNormal(dgp):
    def __init__(self):
        dgp.__init__(self)
        eq = 'Wrapper for Numpy\'s log normal random generator'
        dist = 'log normal'
        self.pdf.__func__.__doc__ = _pdf_doc.format(dist=dist, eq=eq)
        self.__doc__ = _class_doc.format(class_='Log Normal', eq=eq)
    sample = rand.lognormal
    def _pdf(self, mesh):
        mesh = np.asarray(mesh)
        if (mesh<=0).any():
            raise ValueError('mesh must be >0')
        return np.exp(-np.log(mesh)**2/2)/mesh/_NORM_PDF_C
    def mesh(self, N=None):
        if N is None:
            N = 2**14
        mesh, step = np.linspace(0, 10, num=N, retstep=True)
        mesh += step
        return mesh
        
class AsymClaw(dgp):
    def __init__(self):
        dgp.__init__(self)
        eq = '1/2*N(0, 1) + sum_{k=-2}^2 2**(1-k)/31*N(k+1/2, (2**-k/10)^2)'
        dist = 'asymmetric claw'
        self.sample.__func__.__doc__ = _samp_doc.format(dist=dist, eq=eq)
        self.pdf.__func__.__doc__ = _pdf_doc.format(dist=dist, eq=eq)
        self.__doc__ = _class_doc.format(class_='Asymmetric Claw', eq=eq)
    _inputs = [(0, 1)]
    _rates = [1/2]
    for k in xrange(-2, 3):
        _inputs.append((k+1/2, 2**(-k)/10))
        _rates.append(2**(1-k)/31)

class Trimodal(dgp):
    def __init__(self):
        dgp.__init__(self)
        eq = '1/3*sum_{k=0}^2 N(80*k, (k+1)^4)'
        self.sample.__func__.__doc__ = _samp_doc.format(dist='trimodal', eq=eq)
        self.pdf.__func__.__doc__ = _pdf_doc.format(dist='trimodal', eq=eq)
        self.__doc__ = _class_doc.format(class_='Trimodal', eq=eq)
    _inputs = []
    for k in xrange(3):
        _inputs.append((80*k, (k+1)**2))
    _rates = [1/3]*3

class FiveModes(dgp):
    def __init__(self):
        dgp.__init__(self)
        eq = '1/5*sum_{k=0}^4 N(80*k, (k+1)^2)'
        dist = 'five modal'
        self.sample.__func__.__doc__ = _samp_doc.format(dist=dist, eq=eq)
        self.pdf.__func__.__doc__ = _pdf_doc.format(dist=dist, eq=eq)
        self.__doc__ = _class_doc.format(class_='Five Modal', eq=eq)
    _inputs = []
    for k in xrange(5):
        _inputs.append((80*k, k+1))
    _rates = [1/5]*5

class TenModes(dgp):
    def __init__(self):
        dgp.__init__(self)
        eq = '1/10*sum_{k=0}^9 N(100*k, (k+1)^2)'
        dist = 'ten modal'
        self.sample.__func__.__doc__ = _samp_doc.format(dist=dist, eq=eq)
        self.pdf.__func__.__doc__ = _pdf_doc.format(dist=dist, eq=eq)
        self.__doc__ = _class_doc.format(class_='Ten Modal', eq=eq)
    _inputs = []
    for k in xrange(10):
        _inputs.append((100*k, k+1))
    _rates = [1/10]*10

class SmoothComb(dgp):
    def __init__(self):
        dgp.__init__(self)
        eq = 'sum_{k=0}^5 2**(5-k)/63*N((65-96*2**-k)/21, (32/63*2**(-2*k))^2)'
        dist = 'smooth comb'
        self.sample.__func__.__doc__ = _samp_doc.format(dist=dist, eq=eq)
        self.pdf.__func__.__doc__ = _pdf_doc.format(dist=dist, eq=eq)
        self.__doc__ = _class_doc.format(class_='Ten Modal', eq=eq)
    _inputs = []
    _rates = []
    for k in xrange(6):
        _inputs.append(((65-96*2**-k)/21, 32/63*2**(-k)))
        _rates.append(2**(5-k)/63)
