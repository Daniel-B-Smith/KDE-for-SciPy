"""
This file creates the data generating functions necessary to reproduce Table 1
from:

Z. I. Botev, J. F. Grotowski, and D. P. Kroese. Kernel density
estimation via diffusion. The Annals of Statistics, 38(5):2916-2957, 2010.

Daniel B. Smith, PhD
1-29-2013
"""

from __future__ import division

import numpy.random as rand
from random import shuffle

_doc = """Generates according to a {dist} distribution

Parameters:
----------
nsamp: int
       Number of samples to generate

{eq}
"""

def nsamp_opt(func):
    def inner(nsamp=None):
        if nsamp is None:
            nsamp = 1
        return func(nsamp)
    return inner

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

@nsamp_opt
def claw(nsamp):
    inputs = [(0, 1)]
    for k in xrange(5):
        inputs.append((k/2-1, 1/10))
    counts = rand.multinomial(nsamp, [1/2]+[1/10]*5, size=1)[0]
    return _generate(inputs, counts)
eq = ' 1/2*N(0,1) + sum_{k=0}^4 1/10*N(k/2-1, (1/10)^2)'
claw.__doc__ = _doc.format(dist='claw', eq=eq)

@nsamp_opt
def strongly_skewed(nsamp):
    inputs = []
    for k in xrange(8):
        inputs.append((3*((2/3)**k-1), (2/3)**k))
    counts = rand.multinomial(nsamp, [1/8]*8, size=1)[0]
    return _generate(inputs, counts)
eq = 'sum_{k=0}^7 1/8*N(3*((2/3)^k-1), (2/3)^(2k))'
strongly_skewed.__doc__ = _doc.format(dist='strongly skewed', eq=eq)

@nsamp_opt
def kurtotic_unimodal(nsamp):
    inputs = [(0, 1), (0, 1/10)]
    counts = rand.multinomial(nsamp, [2/3, 1/3], size=1)[0]
    return _generate(inputs, counts)
kurtotic_unimodal.__doc__ = _doc.format(dist='kurtotic unimodal',
                                        eq='2/3*N(0,1) + 1/3*N(0,(1/10)^2)')

@nsamp_opt
def double_claw(nsamp):
    inputs = [(-1, 2/3), (1, 2/3)]
    for k in xrange(7):
        inputs.append(((k-3)/2, 1/100))
    counts = rand.multinomial(nsamp, [49/100]*2+[1/350]*7, size=1)[0]
    return _generate(inputs, counts)
eq = ('49/100*N(-1, (2/3)^2) + 49/100*N(1, (2/3)^2) + \n' + 
      '    sum_{k=0}^6 1/350*N((k-3)/2, (1/100)^2)')
double_claw.__doc__ = _doc.format(dist='double claw', eq=eq)
                                  
@nsamp_opt
def discrete_comb(nsamp):
    inputs = []
    for k in xrange(3):
        inputs.append(((2*k-15)/7, 2/7))
    for k in xrange(8, 11):
        inputs.append((2*k/7, 1/21))
    counts = rand.multinomial(nsamp, [2/7]*3+[1/21]*3, size=1)[0]
    return _generate(inputs, counts)
eq = ('2/7*sum_{k=0}^2 N((12*k-15/7), (2/7)^2) + \n' +
      '    1/21*sum_{k=8}^10 N(2*k/7, (1/21)^2)')
discrete_comb.__doc__ = _doc.format(dist='discrete comb', eq=eq)

@nsamp_opt
def asym_double_claw(nsamp):
    inputs = []
    for k in xrange(2):
        inputs.append((2*k-1, 2/3))
    for k in xrange(1, 4):
        inputs.append((-k/2, 1/100))
    for k in xrange(1, 4):
        inputs.append((k/2, 7/100))
    counts = rand.multinomial(nsamp, [46/100]*2+[1/300]*3+[7/300]*3, size=1)[0]
    return _generate(inputs, counts)
eq = ('46/100*sum_{k=0}^1 N(2*k-1, (2/3)^2) + 1/300*sum_{k=1}^3 ' + 
      'N(-k/2, (1/100)^2)\n    sum_{k=1}^3 N(k/2, (7/100)^2)')
asym_double_claw.__doc__ = _doc.format(dist='asymmetric double claw', eq=eq)

@nsamp_opt
def outlier(nsamp):
    inputs = [(0, 1), (0, 1/10)]
    counts = rand.multinomial(nsamp, [1/10, 9/10], size=1)[0]
    return _generate(inputs, counts)
outlier.__doc__ = _doc.format(dist='outlier', 
                              eq='1/10*N(0, 1)+9/10*N(0, (1/10)^2)')

@nsamp_opt
def sep_bimodal(nsamp):
    inputs  = [(-12, 1/2), (12, 1/2)]
    counts = rand.multinomial(nsamp, [1/2, 1/2], size=1)[0]
    return _generate(inputs, counts)
eq = '1/2*N(-12, (1/2)^2) + 1/2*N(12, (1/2)^2)'
sep_bimodal.__doc__ = _doc.format(dist='separated bimodal', eq=eq)


@nsamp_opt
def skew_bimodal(nsamp):
    inputs = [(0,1), (3/2, 1/3)]
    counts = rand.multinomial(nsamp, [3/4, 1/4], size=1)[0]
    return _generate(inputs, counts)
eq = '3/4*N(0, 1) + 1/4*N(3/2, (1/3)^2)'
skew_bimodal.__doc__ = _doc.format(dist='skewed bimodal', eq=eq)

@nsamp_opt
def bimodal(nsamp):
    inputs = [(0, 1/10), (5, 1)]
    counts = rand.multinomial(nsamp, [1/2, 1/2], size=1)[0]
    return _generate(inputs, counts)
eq = '1/2*N(0, (1/10)^2) + 1/2*N(5, 1)'
bimodal.__doc__ = _doc.format(dist='bimodal', eq=eq)

@nsamp_opt
def log_normal(nsamp):
    return rand.lognormal(size=nsamp)
eq = 'Wrapper for Numpy\'s log normal random generator'
log_normal.__doc__ = _doc.format(dist='log normal', eq=eq)

@nsamp_opt
def asym_claw(nsamp):
    inputs = [(0, 1)]
    rates = [1/2]
    for k in xrange(-2, 3):
        inputs.append((k+1/2, 2**(-k)/10))
        rates.append(2**(1-k)/31)
    counts = rand.multinomial(nsamp, rates, size=1)[0]
    return _generate(inputs, counts)
eq = '1/2*N(0, 1) + sum_{k=-2}^2 2**(1-k)/31*N(k+1/2, (2**-k/10)^2)'
asym_claw.__doc__ = _doc.format(dist='asymmetric claw', eq=eq)

@nsamp_opt
def trimodal(nsamp):
    inputs = []
    for k in xrange(3):
        inputs.append((80*k, (k+1)^2))
    counts = rand.multinomial(nsamp, [1/3]*3, size=1)[0]
    return _generate(inputs, counts)
trimodal.__doc__ = _doc.format(dist='trimodal',
                               eq='1/3*sum_{k=0}^2 N(80*k, (k+1)^4)')

@nsamp_opt
def five_modes(nsamp):
    inputs = []
    for k in xrange(5):
        inputs.append((80*k, (k+1)^2))
    counts = rand.multinomial(nsamp, [1/5]*5, size=1)[0]
    return _generate(inputs, counts)
five_modes.__doc__ = _doc.format(dist='five modal',
                                 eq='1/5*sum_{k=0}^4 N(80*k, (k+1)^4)')

@nsamp_opt
def ten_modes(nsamp):
    inputs = []
    for k in xrange(10):
        inputs.append((100*k, (k+1)^2))
    counts = rand.multinomial(nsamp, [1/10]*10, size=1)[0]
    return _generate(inputs, counts)
ten_modes.__doc__ = _doc.format(dist='ten modal',
                                 eq='1/10*sum_{k=0}^9 N(100*k, (k+1)^4)')

@nsamp_opt
def smooth_comb(nsamp):
    inputs = []
    rates = []
    for k in xrange(6):
        inputs.append(((65-96)*2**-k/21, 32/63*2**(-2*k)))
        rates.append(2**(5-k)/63)
    counts = rand.multinomial(nsamp, rates, size=1)[0]
    return _generate(inputs, counts)
eq = ('sum_{k=0}^5 2**(5-k)/63*N((65-96)*2**-k/21, (32/63*2**(-2*k))^2)'
smooth_comb.__doc__ = _doc.format(dist='smooth comb', eq=eq)
