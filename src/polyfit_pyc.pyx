# Purpose: Cython wrapper for the polyfit module
# Author: Attila Bódi
# Version: 0.1  2021MAR18

import cython
import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc,realloc, free

# declare the interface to the C code
cdef extern char* polyfitter(
    double* inphase,
    double* influx,
    double* inerror,
    int nobs,
    int order,
    int ITERS,
    double STEP_SIZE,
    int vertices,
    int nKNOTS,
    double* knots,
    int FIND_KNOTS,
    int FIND_STEP,
    int chainlength,
    int applypshift,
    double* outphase,
    double* outpolyfit
    )

@cython.boundscheck(False)
@cython.wraparound(False)
def polyfit(    np.ndarray[np.double_t, ndim=1, mode="c"] phase not None,
                np.ndarray[np.double_t, ndim=1, mode="c"] flux not None,
                np.ndarray[np.double_t, ndim=1, mode="c"] error not None,
                int order=2,
                int iters=10000,
                double stepsize=0.01,
                np.ndarray[np.double_t, ndim=1, mode="c"] knots=None,
                int vertices=200,
                int findknots=False,
                int findstep=False,
                int chainlength=8,
                int applypshift=False
    ):
    """
    ``Polyfit`` fits polynomial chain to phase curve

    Parameters
    ----------
    phase: array
        Phase values from -0.5 to 0.5
    flux: array
        Flux/mag values
    error: array
        Flux/mag errors
    order: int, default: 2
        fitting polynomial order
    iters: int, default: 10000
        number of iterations
    stepsize: float, default: 0.01
        step for random knot displacement
    knots: array
        explicit array of knots
    vertices: int, default: 200
        number of vertices in the computed fit
    findknots: bool, default: False
        attempt to find knots automatically
    findstep: bool, default: False
        attempt to find step automatically
    chainlength: int, default: 8
        minimum chain length for automatic knot search
    applypshift: bool, default: False
        shift phase so that the polyfit minimum is at phase 0
    """

    cdef int nobs = phase.size
    cdef np.ndarray[np.double_t, ndim=1, mode="c"] outphase   = np.zeros((vertices),dtype=np.floating)
    cdef np.ndarray[np.double_t, ndim=1, mode="c"] outpolyfit = np.zeros((vertices),dtype=np.floating)
    cdef char *messages
    cdef list stripedmessages
    cdef bytes item

    # --- Variables in C code ---
    cdef int    nKNOTS = 4
    cdef double* knotvalues = NULL

    # --- Update knots if passed ---
    if knots is not None and not findknots:
        nKNOTS = len(knots)
        knotvalues = <double*>realloc(knotvalues, nKNOTS*sizeof(double))  # use the realloc function from C's stdlib library
        for i in range(nKNOTS):
            knotvalues[i] = knots[i]

    # --- Call C funtion ---
    messages = polyfitter(
        &phase[0],
        &flux[0],
        &error[0],
        nobs,
        order,
        iters,
        stepsize,
        vertices,
        nKNOTS,
        &knotvalues[0],
        findknots,
        findstep,
        chainlength,
        applypshift,
        &outphase[0],
        &outpolyfit[0]
    )

    free(knotvalues);

    stripedmessages = [ str(item, 'utf-8') for item in messages.splitlines() ]
    free(messages)

    return outphase, outpolyfit,stripedmessages
