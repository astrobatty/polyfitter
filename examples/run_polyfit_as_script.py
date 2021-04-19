#!/usr/bin/env python
# coding: utf-8

"""
This example shows how to run Polyfitter as a script.
Usually this code can be executed as:

python run_polyfit_as_script.py
"""

from polyfitter import Polyfitter
try:
    import matplotlib.pyplot as plt
except ImportError:
    # For Mac users if loading QtAgg fails
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
import numpy as np

# Main function is mandatory to avoid iterative importing!
if __name__ == '__main__':

    # Parameters from OGLE database
    ID = 'OGLE-BLG-ECL-040474'
    P=1.8995918
    t0=7000.90650

    # Load light curve from OGLE database
    # This is in magnitude scale
    path_to_ogle = 'http://ogledb.astrouw.edu.pl/~ogle/OCVS/data/I/'+ID[-2:]+'/'+ID+'.dat'
    lc = np.loadtxt(path_to_ogle).T

    # For clarity
    time = lc[0]
    mag  = lc[1]
    err  = lc[2]

    # Create Polyfitter instance by setting the brightness scale of your data
    # Set "mag" or "flux" scale
    pf = Polyfitter(scale='mag')

    # Run polynomial chain fitting
    t0new, phase, polyfit, messages = pf.get_polyfit(time,mag,err,P,t0)

    # Plot and save phase curve and polyfit
    plt.errorbar((time-t0new)/P%1,mag,err,fmt='k.')
    plt.errorbar((time-t0new)/P%1-1,mag,err,fmt='k.')
    plt.plot(phase,polyfit,c='r',zorder=10)
    plt.plot(phase+1,polyfit,c='r',zorder=10)
    plt.xlabel('Phase')
    plt.ylabel('Magnitude')
    plt.xlim(-0.5,1)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(ID + '.pdf')
    plt.close()

    # Get morphology classification
    morp_array = pf.c
    print('Morphology type =' , morp_array[0] )

    # Check original and new epochs
    print('Original epoch: {} -> new epoch: {}'.format(t0,t0new) )
