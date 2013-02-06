#!/usr/bin/env python -tt

from __future__ import division

import sys
import inspect
import numpy as np
import matplotlib.pyplot as plt
import kde
import dgp_class as dgp

def main(Nsamp=None, Nmesh=None):
    """
    Generates plots for the 16 test cases for both the analytical pdf and the
    kernel density estimate.

    Parameters
    ----------
    Nsamp: int
           Number of samples used for the kde
    Nmesh: int
           Number of points used for the mesh
    """
    if Nsamp is None:
        Nsamp = 10000
    classes = [(cls, name) for name, cls in inspect.getmembers(dgp)
               if inspect.isclass(cls) and not cls in (dgp.dgp, dgp.LogNormal)]
    for cls, name in classes:
        print 'Generating graph for', name
        model = cls()
        x = model.sample(size=Nsamp)
        t, mesh, kdense = kde.kde(x, N=Nmesh)
        f = model.pdf(mesh)
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title(name, size=36)

        plt.plot(mesh, kdense)
        plt.plot(mesh, f)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontsize(24)

        fig.set_figheight(10)
        fig.set_figwidth(12)
        fig.savefig(name+'.pdf')
        plt.close()
        

    return None

if __name__ == "__main__":
    answer = raw_input('Warning: This script generates 16 pdfs\n' +
                       'Do you wish to continue? (Y/N)')
    if answer in ('Y', 'y', 'yes', 'Yes', 'YES'):
        main()
        sys.exit(0)
    else:
        sys.exit(1)

