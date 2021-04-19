import pytest

from polyfitter import Polyfitter
import numpy as np

from numpy.testing import assert_array_almost_equal

def test_get_morph():
    """Can we get the proper morphology type?"""
    ID = 'OGLE-BLG-ECL-040474'
    P=1.8995918
    t0=7000.90650

    path_to_ogle = 'http://ogledb.astrouw.edu.pl/~ogle/OCVS/data/I/'+ID[-2:]+'/'+ID+'.dat'
    lc = np.loadtxt(path_to_ogle).T

    time = lc[0]
    mag  = lc[1]
    err  = lc[2]

    pf = Polyfitter(scale='mag')
    t0new,phase,polyfit,messages = pf.get_polyfit(time,mag,err,P,t0)

    assert_array_almost_equal( pf.c , np.array([0.42073795]) )

    assert_array_almost_equal( pf.get_c( np.vstack((polyfit,polyfit)) ) , np.array([0.42073795,0.42073795]) )
