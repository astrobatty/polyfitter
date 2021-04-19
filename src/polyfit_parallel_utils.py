#!/usr/bin/env python
# coding: utf-8

import polyfitcore # MUST BE AT BEGINING OF FILE!
import numpy as np
import copy
from sklearn.cluster import DBSCAN
from scipy.signal import medfilt
from scipy.interpolate import interp1d
from functools import partial
import multiprocessing
try:
    multiprocessing.set_start_method('spawn')
except RuntimeError:
    pass

import pickle
from pathlib import Path
from . import PACKAGEDIR

import os
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Process
from multiprocessing import Queue
from typing import Callable
from typing import Iterable
from typing import Dict
from typing import Any

class ProcessKillingExecutor:
    """
    Source: https://codereview.stackexchange.com/questions/142828/python-executer-that-kills-processes-after-a-timeout

    The ProcessKillingExecutor works like an `Executor
    <https://docs.python.org/dev/library/concurrent.futures.html#executor-objects>`_
    in that it uses a bunch of processes to execute calls to a function with
    different arguments asynchronously.

    But other than the `ProcessPoolExecutor
    <https://docs.python.org/dev/library/concurrent.futures.html#concurrent.futures.ProcessPoolExecutor>`_,
    the ProcessKillingExecutor forks a new Process for each function call that
    terminates after the function returns or if a timeout occurs.

    This means that contrary to the Executors and similar classes provided by
    the Python Standard Library, you can rely on the fact that a process will
    get killed if a timeout occurs and that absolutely no side can occur
    between function calls.

    Note that descendant processes of each process will not be terminated –
    they will simply become orphaned.
    """

    def __init__(self, max_workers: int=None):
        self.processes = max_workers or os.cpu_count()

    def map(self,
            func: Callable,
            iterable: Iterable,
            timeout: float=None,
            callback_timeout: Callable=None,
            daemon: bool = True
            ) -> Iterable:
        """
        :param func: the function to execute
        :param iterable: an iterable of function arguments
        :param timeout: after this time, the process executing the function
                will be killed if it did not finish
        :param callback_timeout: this function will be called, if the task
                times out. It gets the same arguments as the original function
        :param daemon: define the child process as daemon
        """
        executor = ThreadPoolExecutor(max_workers=self.processes)
        params = ({'func': func, 'fn_args': p_args, "p_kwargs": {},
                   'timeout': timeout, 'callback_timeout': callback_timeout,
                   'daemon': daemon} for p_args in iterable)
        return executor.map(self._submit_unpack_kwargs, params)

    def _submit_unpack_kwargs(self, params):
        """ unpack the kwargs and call submit """

        return self.submit(**params)

    def submit(self,
               func: Callable,
               fn_args: Any,
               p_kwargs: Dict,
               timeout: float,
               callback_timeout: Callable[[Any], Any],
               daemon: bool):
        """
        Submits a callable to be executed with the given arguments.
        Schedules the callable to be executed as func(*args, **kwargs) in a new
         process.
        :param func: the function to execute
        :param fn_args: the arguments to pass to the function. Can be one argument
                or a tuple of multiple args.
        :param p_kwargs: the kwargs to pass to the function
        :param timeout: after this time, the process executing the function
                will be killed if it did not finish
        :param callback_timeout: this function will be called with the same
                arguments, if the task times out.
        :param daemon: run the child process as daemon
        :return: the result of the function, or None if the process failed or
                timed out
        """
        p_args = fn_args if isinstance(fn_args, tuple) else (fn_args,)
        queue = Queue()
        p = Process(target=self._process_run,
                    args=(queue, func, fn_args,), kwargs=p_kwargs)

        if daemon:
            p.deamon = True

        p.start()
        p.join(timeout=timeout)
        if not queue.empty():
            return queue.get()
        if callback_timeout:
            callback_timeout(*p_args, **p_kwargs)
        if p.is_alive():
            p.terminate()
            p.join()

    @staticmethod
    def _process_run(queue: Queue, func: Callable[[Any], Any]=None,
                     *args, **kwargs):
        """
        Executes the specified function as func(*args, **kwargs).
        The result will be stored in the shared dictionary
        :param func: the function to execute
        :param queue: a Queue
        """
        queue.put(func(*args, **kwargs))


def fun_timeout(n):
    print('timeout:', n)

def preprocess_polyfit(t,y,z,t0,P):
    phase = (t-t0)/P - ((t-t0)/P).astype(int)
    phase[ phase>0.5 ] -= 1
    phase[ phase<-0.5 ] += 1

    order = np.argsort(phase)
    t = phase[order]
    #y = -y[order] + 2*np.median(y)
    y = y[order]
    z = z[order]

    # --- rough sigma clip ---
    um = (y < np.mean(y) + 4*np.std(y)) & (y > np.mean(y) - 22*np.std(y))
    t = t[um]
    y = y[um]
    z = z[um]

    # --- medfilter + sigma clip | outlier points  (bright points) ---
    # extend 0-1 phase to -1–+1 to handle edges
    t_extended = np.concatenate((t-1,t,t+1))
    y_extended = np.concatenate((y,y,y))
    um = y_extended>np.median(y)
    yfilt = medfilt(y_extended,kernel_size=3)
    um2 = (y_extended-yfilt) > np.median(y_extended-yfilt) + 5*np.std(y_extended-yfilt)
    um = np.logical_and(um,um2)
    um = np.invert(um)
    um = um[(t_extended>=t.min()) & (t_extended<=t.max())] # Get back normal phase range
    t = t[um]
    y = y[um]
    z = z[um]

    # --- medfilter + sigma clip | inlier points (faint points) ---
    # extend 0-1 phase to -1–+1 to handle edges
    t_extended = np.concatenate((t-1,t,t+1))
    y_extended = np.concatenate((y,y,y))
    um = y_extended<np.median(y) - 8*np.std(y)
    yfilt = medfilt(y_extended,kernel_size=3)
    um2 = (y_extended-yfilt) < np.median(y_extended-yfilt) - 10*np.std(y_extended-yfilt)
    um = np.logical_and(um,um2)
    um = np.invert(um)
    um = um[(t_extended>=t.min()) & (t_extended<=t.max())] # Get back normal phase range
    if np.sum(um)>0 and y.min() < y[um].min():
        # Cut faint point if there are no other fainter points left
        t = t[um]
        y = y[um]
        z = z[um]

    # --- find outliers w/ DBSCAN ---
    if P<5:
        # stricter constraints for short period EBs
        yfactor = 0.3     # Scale y values to get ellipsoid
        dbsigma = 4       # For DBSCAN eps
    else:
        yfactor = 0.5
        dbsigma = 7
    # extend phase range to handle points at edges
    tdb = np.concatenate((t-1,t,t+1))
    ydb = np.concatenate((y,y,y))
    ydb = (ydb-ydb.min())/ydb.ptp() *yfactor

    #--- find outliers w/ DBSCAN ---
    diffy = np.abs(np.diff(ydb))
    eps = np.mean(diffy)+dbsigma*np.std(diffy)
    db  = DBSCAN(eps=eps, min_samples=2,n_jobs=-1).fit(np.c_[tdb,ydb])

    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    core_samples_mask = core_samples_mask[len(t):2*len(t)]
    mask = core_samples_mask

    """# Plot identified outliers
    yscaled = (y-y.min())/y.ptp() *yfactor
    zscaled *= yfactor

    fig,axs = plt.subplots(1,1)
    for pts in zip(t[~core_samples_mask],yscaled[~core_samples_mask]):
        circle = plt.Circle((pts[0], pts[1]), eps, color='gray', facecolor=None,alpha=0.3,fill=False)
        axs.add_patch(circle)
        circle = plt.Circle((pts[0]+1, pts[1]), eps, color='gray', facecolor=None,alpha=0.3,fill=False)
        axs.add_patch(circle)
    plt.errorbar(t,yscaled,yerr=zscaled,fmt='.')
    plt.errorbar(t+1,yscaled,yerr=zscaled,fmt='.',c='C0')
    plt.plot(t[~core_samples_mask],yscaled[~core_samples_mask],'ro')
    plt.plot(t[~core_samples_mask]+1,yscaled[~core_samples_mask],'ro')
    plt.show()
    plt.close()
    """

    if np.sum(mask) < len(t) * 0.5:
        # if DBSCAN mark all points as outlier
        return t,y,z
    else:
        return t[mask],y[mask],z[mask]

def get_initial_knots(out):
    knotat = [index for index,o in enumerate(out) if '#    knot\t\t' in o][0]

    initial_knots = []
    while True:
        knotat = knotat+1
        try:
            initial_knots.append( str(float(out[knotat].split('\t')[0].split('#')[-1])) )
        except (IndexError,ValueError):
            break

    return np.array(initial_knots).astype(np.float64)

def check_if_sharp_upward_peak(out,phase,polyfit,debug=False):
    # Check if we don't have a sharp upward peak
    # If so, do fit again with other initial knot values
    origknots = get_initial_knots(out)
    # The first element must be added to the end to measure all distances
    origknots = np.concatenate((origknots,origknots[:1]+1))
    origknotsdiff = np.diff(origknots)
    if np.any(origknotsdiff<0.05):
        if debug: print('there is a sharp peak !')
        # Check if there is a sharp peak
        peakat = np.where(origknotsdiff<0.05)[0][0]
        phase_duplicated = np.concatenate((phase,phase+1))
        polyfit_duplicated = np.concatenate((polyfit,polyfit))
        # +/- 1/1000 phase neeeded to make sure the peak is covered (solves resolution issuses)
        um = ((phase_duplicated >= origknots[peakat]-0.0011) & (phase_duplicated <= origknots[peakat+1]+0.0011))
        if np.all( np.sort(np.unique(origknotsdiff)[:2]) <0.05):
            # if we have two sharp minima, i.e. an algol, do not fix inital values!
            if debug: print('an algol, do not fix inital values!')
            do_fit_again_fixed_knots = False
        elif len(polyfit_duplicated[um])>0 and np.max(polyfit_duplicated[um])>=polyfit_duplicated[um][0] and np.max(polyfit_duplicated[um])>=polyfit_duplicated[um][-1]:
            # Check if sharp peak is upwards
            if debug: print('peak is upwards!')
            do_fit_again_fixed_knots = True
        else:
            do_fit_again_fixed_knots = False
    else:
        do_fit_again_fixed_knots = False

    return do_fit_again_fixed_knots

def get_BIC(chi2,n,k):
    return chi2 + k*np.log(n)

def fit_polyfit_and_correct_epoch(params,lc,t0,P,subtract,
                                        chain_length,debug,
                                        vertices,maxiters):
    """
    This function prepare the phase curve and fit Polyfit.
    Then checks if primary minimum is at phase 0, if not
    it shifts the light curve and performs the fit again
    and again until the correct epoch is not found.
    """
    find_knots = params[0]
    order = params[1]
    initial_knots = params[2]

    t0new = copy.copy(t0)

    x,y,z = lc[0].copy(),lc[1].copy(),lc[2].copy()
    x,y,z = preprocess_polyfit(x,y,z,t0,P)

    if debug: print('Fitting polyfit...')
    phase,polyfit,out = polyfitcore.polyfit(x,y,z,
                                  findknots=find_knots,
                                  chainlength=chain_length,
                                  order=order,
                                  knots=initial_knots,
                                  findstep=True,
                                  iters=maxiters,
                                  vertices=vertices)



    try:
        # Get phase of first and second minima
        first_minimum = phase[np.argmin(polyfit)]
        um = np.logical_or( (phase < -0.45 ) , ( 0.45 < phase) )
        second_minimum = phase[um][np.argmin(polyfit[um])]

        # If T0 is wrong redo polyfit w/ new T0
        isSecondMinOK = (polyfit[um][np.argmin(polyfit[um])] < np.mean(polyfit))
        if (-0.49 < first_minimum < -0.01 or 0.01 < first_minimum < 0.49) and isSecondMinOK:
            if (-0.49 < second_minimum < 0.49):
                # Check if second min is smaller than mean -> check if second min yields good T0
                t0 += phase[np.argmin(polyfit)] * P
                t0new = np.copy(t0) + subtract

                x,y,z = lc[0].copy(),lc[1].copy(),lc[2].copy()
                x,y,z = preprocess_polyfit(x,y,z,t0,P)

                if debug: print('Fitting polyfit w/ new t0...')
                phase,polyfit,out = polyfitcore.polyfit(x,y,z,
                                              findknots=find_knots,
                                              chainlength=chain_length,
                                              order=order,
                                              knots=initial_knots,
                                              findstep=True,
                                              iters=maxiters,
                                              vertices=vertices)


        elif -0.49 < first_minimum < -0.01 or 0.01 < first_minimum < 0.49:
            # Check if second min is small -> check only first minimum
            t0 += phase[np.argmin(polyfit)] * P
            t0new = np.copy(t0) + subtract

            x,y,z = lc[0].copy(),lc[1].copy(),lc[2].copy()
            x,y,z = preprocess_polyfit(x,y,z,t0,P)

            if debug: print('Fitting polyfit where second min is small...')
            phase,polyfit,out = polyfitcore.polyfit(x,y,z,
                                          findknots=find_knots,
                                          chainlength=chain_length,
                                          order=order,
                                          knots=initial_knots,
                                          findstep=True,
                                          iters=maxiters,
                                          vertices=vertices)



        # Get phase of first and second minima
        um = (-0.05 < phase) & (phase < 0.05)
        first_minimum = phase[um][np.argmin(polyfit[um])]
        first_minimum_mag = np.min(polyfit[um])
        um = np.logical_or( (phase < -0.45 ) , ( 0.45 < phase) )
        second_minimum = phase[um][np.argmin(polyfit[um])]
        second_minimum_mag = np.min(polyfit[um])

        # Check if first minimum is really the primary!
        if second_minimum_mag < first_minimum_mag:
            t0 += P/2
            t0new = np.copy(t0) + subtract

            x,y,z = lc[0].copy(),lc[1].copy(),lc[2].copy()
            x,y,z = preprocess_polyfit(x,y,z,t0,P)

            if debug: print('Fitting polyfit to check primary minimum...')
            phase,polyfit,out = polyfitcore.polyfit(x,y,z,
                                          findknots=find_knots,
                                          chainlength=chain_length,
                                          order=order,
                                          knots=initial_knots,
                                          findstep=True,
                                          iters=maxiters,
                                          vertices=vertices)


        # Final check to make sure smallest minimum is the primary
        # Get phase of first minimum
        smallest_minimum = phase[np.argmin(polyfit)]

        # Check if first minimum is really the primary!
        if smallest_minimum < -0.01 or 0.01 < smallest_minimum:
            t0shift = phase[np.argmin(polyfit)]
            # shift polyfit to set smallest minimum as primary
            phase = phase-t0shift
            if np.sum(phase>0.5)>0:
                offset = np.min(phase[phase>0.5])-0.5
                phase[phase>0.5] -= 1+offset
            if np.sum(phase<-0.5)>0:
                offset = np.max(phase[phase<-0.5])+0.5
                phase[phase<-0.5] += 1-offset
            keyorder = np.argsort(phase)
            phase = phase[keyorder]
            polyfit = polyfit[keyorder]
            # shift light curve to set smallest minimum as primary
            x = x-t0shift
            if np.sum(x>0.5)>0:
                offset = np.min(x[x>0.5])-0.5
                x[x>0.5] -= 1+offset
            if np.sum(x<-0.5)>0:
                offset = np.max(x[x<-0.5])+0.5
                x[x<-0.5] += 1-offset
            keyorder = np.argsort(x)
            x = x[keyorder]
            y = y[keyorder]
            z = z[keyorder]

            t0 += t0shift* P
            t0new = np.copy(t0) + subtract

    except ValueError:
        if debug: print("Something went wrong")

        return [np.nan]*9

    # Get chain length from PolyFit output
    #try:
    #    chain_length_final =  int([o for o in out.decode().split('\n') if 'total number of chains found:' in o][0].split()[-1])
    #except IndexError:
    #    chain_length_final = chain_length

    # Get order length from PolyFit output
    #try:
    #    order_final =  int([o for o in out if '# Fitting polynomial order:' in o][0].split()[-1])
    #except IndexError:
    #    order_final = order

    # Interpolate PolyFit to known phase values to get chi sq
    try:
        func = interp1d(phase, polyfit, kind='cubic', assume_sorted=True,fill_value='extrapolate')
    except ValueError:
        if debug: print("Something went wrong")

        return [np.nan]*9

    # Calculate chi2 and Bayesian Information Criterion to decide which is the best model
    reduced_chi2 = (np.power( func(x)-y, 2 )/(z**2) ).sum()
    reduced_chi2 /= (x.shape[0] - (order+1) )
    chi2 = (np.power( func(x)-y, 2 )/z ).sum()
    BIC = get_BIC(chi2,len(x),order)

    return phase,polyfit,out, t0new, BIC,chi2,reduced_chi2,order,find_knots

def compare_results_and_choose_fit(all_phase,all_polyfit,
                                    all_out,all_t0new,
                                    all_BIC,all_chi2,all_reduced_chi2,all_order,
                                    all_find_knots,
                                    debug=False,lc=None,P=None):
    """
    After all polyfit is done this function compares the results
    and choose the best fit based on Chi2 and BIC.
    """

    # --- Initialize values ---
    bestBIC = np.inf
    best_chi2 = np.inf
    best_reduced_chi2 = np.inf
    do_fit_again_final_knots = False
    do_fit_again_fixed_knots = False
    one_only_sharp_peak = False


    for ii in range(len(all_BIC)):
        phase       = all_phase[ii]
        polyfit     = all_polyfit[ii]
        out         = all_out[ii]
        t0new       = all_t0new[ii]
        BIC         = all_BIC[ii]
        chi2        = all_chi2[ii]
        reduced_chi2 = all_reduced_chi2[ii]
        order       = all_order[ii]
        find_knots  = all_find_knots[ii]

        if  np.all(np.isnan(phase)):
            continue

        if BIC < bestBIC-1 and order==4 and chi2 < 1.5*best_chi2 and reduced_chi2<50:
            bestBIC = BIC
            best_chi2 = chi2
            best_reduced_chi2 = reduced_chi2
            if debug: print("\n---> This is the best!")
            if find_knots:
                # If knots are fixed, we'll get the some result again
                do_fit_again_final_knots = True
            else:
                do_fit_again_final_knots = False
            final_knots = get_initial_knots(out)
            # Check if we don't have a sharp upward peak
            # If so, do fit again with other initial knot values
            do_fit_again_fixed_knots = check_if_sharp_upward_peak(out,phase,polyfit,debug=debug)
            final_phase, final_polyfit = phase.copy(), polyfit.copy()
            final_t0new = copy.copy(t0new)
            final_out = copy.copy(out)
            final_order = copy.copy(order)
            final_find_knots = copy.copy(find_knots)

            # Flag if there is only one very sharp peak
            # then slightly greater BIC and chi2 can be good
            origknots = get_initial_knots(out)
            origknots = np.concatenate((origknots,origknots+1))[:5]
            origknotsdiff = np.diff(origknots)
            if np.sum( origknotsdiff < 0.01 ) == 1:
                one_only_sharp_peak = True
            else:
                one_only_sharp_peak = False




        elif BIC < bestBIC and order==2 and chi2 < 1.5*best_chi2:
            bestBIC = BIC
            best_chi2 = chi2
            best_reduced_chi2 = reduced_chi2
            if debug: print("\n---> This is the best!")
            if find_knots:
                # If knots are fixed, we'll get the some result again
                do_fit_again_final_knots = True
            else:
                do_fit_again_final_knots = False
            final_knots = get_initial_knots(out)
            # Check if we don't have a sharp upward peak
            # If so, do fit again with other initial knot values
            do_fit_again_fixed_knots = check_if_sharp_upward_peak(out,phase,polyfit,debug=debug)
            final_phase, final_polyfit = phase.copy(), polyfit.copy()
            final_t0new = copy.copy(t0new)
            final_out = copy.copy(out)
            final_order = copy.copy(order)
            final_find_knots = copy.copy(find_knots)

            # Flag if there is only one very sharp peak
            # then slightly greater BIC and chi2 can be good
            origknots = get_initial_knots(out)
            origknots = np.concatenate((origknots,origknots+1))[:5]
            origknotsdiff = np.diff(origknots)
            if np.sum( origknotsdiff < 0.01 ) == 1:
                one_only_sharp_peak = True
            else:
                one_only_sharp_peak = False




        elif BIC < bestBIC+2 and chi2 < best_chi2*1.95/3:
            if not (order ==4 and reduced_chi2>=50):
                # Chi2 is much smaller and BIC is almost the same (not to mention in wikipedia)
                bestBIC = BIC
                best_chi2 = chi2
                best_reduced_chi2 = reduced_chi2
                if debug: print("\n---> This is the best!")
                if find_knots:
                    # If knots are fixed, we'll get the some result again
                    do_fit_again_final_knots = True
                else:
                    do_fit_again_final_knots = False
                final_knots = get_initial_knots(out)
                # Check if we don't have a sharp upward peak
                # If so, do fit again with other initial knot values
                do_fit_again_fixed_knots = check_if_sharp_upward_peak(out,phase,polyfit,debug=debug)
                final_phase, final_polyfit = phase.copy(), polyfit.copy()
                final_t0new = copy.copy(t0new)
                final_out = copy.copy(out)
                final_order = copy.copy(order)
                final_find_knots = copy.copy(find_knots)

                # Flag if there is only one very sharp peak
                # then slightly greater BIC and chi2 can be good
                origknots = get_initial_knots(out)
                origknots = np.concatenate((origknots,origknots+1))[:5]
                origknotsdiff = np.diff(origknots)
                if np.sum( origknotsdiff < 0.01 ) == 1:
                    one_only_sharp_peak = True
                else:
                    one_only_sharp_peak = False


            """ !!!! 2 -> 5 lett a BIC"""
        elif one_only_sharp_peak and BIC < bestBIC+5 and chi2 < best_chi2*1.1:
            # Chi2 is slightly larger and BIC is almost the same
            # but the sharp peak may be elliminated this way
            bestBIC = BIC
            best_chi2 = chi2
            best_reduced_chi2 = reduced_chi2
            if debug: print("\n---> This is the best!")
            if find_knots:
                # If knots are fixed, we'll get the some result again
                do_fit_again_final_knots = True
            else:
                do_fit_again_final_knots = False
            final_knots = get_initial_knots(out)
            # Check if we don't have a sharp upward peak
            # If so, do fit again with other initial knot values
            do_fit_again_fixed_knots = check_if_sharp_upward_peak(out,phase,polyfit,debug=debug)
            final_phase, final_polyfit = phase.copy(), polyfit.copy()
            final_t0new = copy.copy(t0new)
            final_out = copy.copy(out)
            final_order = copy.copy(order)
            final_find_knots = copy.copy(find_knots)

            # Flag if there is only one very sharp peak
            # then slightly greater BIC and chi2 can be good
            origknots = get_initial_knots(out)
            origknots = np.concatenate((origknots,origknots+1))[:5]
            origknotsdiff = np.diff(origknots)
            if np.sum( origknotsdiff < 0.01 ) == 1:
                one_only_sharp_peak = True
            else:
                one_only_sharp_peak = False

        if debug:
            #func = interp1d(phase, polyfit, kind='cubic', assume_sorted=True,fill_value='extrapolate')
            x,y,z = preprocess_polyfit(lc[0],lc[1],lc[2],t0new,P)

            print( '\npeak?',check_if_sharp_upward_peak(out,phase,polyfit,debug=debug) )

            #SSR = (np.power( func(x)-y, 2 ) ).sum()
            #print('BICnew=', len(x) * np.log( SSR/len(x)) + order*np.log( len(x) ) )
            print( 'Chi2=',chi2,'Order=',order)
            print( 'Reduced Chi2=',reduced_chi2)
            print( 'BIC=',BIC)
            origknots = get_initial_knots(out)
            print( 'origknots=',origknots)
            origknots = np.concatenate((origknots,origknots+1))[:5]
            origknotsdiff = np.diff(origknots)
            print('one_only_sharp_peak=',np.sum( origknotsdiff < 0.01 ) == 1)

            plt.figure(figsize=(15,2))
            plt.plot(x,y,'.')
            plt.plot(x-1,y,'C0.')
            plt.plot(x+1,y,'C0.')
            #plt.gca().invert_yaxis()
            plt.plot(phase,polyfit,lw=3)
            plt.plot(phase-1,polyfit,'C1',lw=3)
            plt.plot(phase+1,polyfit,'C1',lw=3)
            plt.xlim(-1,1)
            plt.show()
            plt.close()


        '''
        # Check if there is another minimum we missed
        origknots = np.array(get_initial_knots(out)).astype(np.float32)
        origknots = np.concatenate((origknots,origknots+1))
        origknotsdiff = np.diff(origknots)
        residual = y-func(x)
        umres = np.where(residual < np.mean(residual)-5*np.std(residual))[0]
        if len(umres)>0 and not do_fit_again_fixed_knots and np.any(origknotsdiff<0.05) and np.any( np.diff( np.where( np.diff(umres)==1 )[0] )==1 ):
            print('another minimum')
            umoutlier = np.where( np.diff(umres)==1 )[0]
            umres = umres[umoutlier]
            umoutlier = np.where(np.diff(umoutlier)==1)[0]
            umres = umres[umoutlier]
            print(get_initial_knots(out))
            plt.plot(x,residual,'.')
            plt.plot(x[umres],residual[umres],'.')
            plt.axhline(np.mean(residual)-5*np.std(residual))
            plt.show()
            plt.close()

            primary_minimum =       phase[np.argmin(polyfit)]
            new_secondary_minimum = x[umres][np.argmin(residual[umres])]
            if np.abs(primary_minimum-new_secondary_minimum) > 0.01:
                # if we found another minimum, not the same
                final_knots = np.sort(np.array([primary_minimum-0.005,primary_minimum+0.005,new_secondary_minimum-0.005,new_secondary_minimum+0.005]))
                final_knots = [str(final_knots[0]),str(final_knots[1]),str(final_knots[2]),str(final_knots[3])]
                do_fit_again_final_knots = True
        '''

    try:
        return (final_phase, final_polyfit, final_t0new,
                final_out,final_knots,final_find_knots,
                bestBIC, best_chi2, best_reduced_chi2,final_order,
                do_fit_again_final_knots, do_fit_again_fixed_knots,
                one_only_sharp_peak)
    except UnboundLocalError:
        return [np.nan]*13


def find_best_polyfit(lc,t0,P,test_knots=None,
                            debug=False,
                            timeout=1000,
                            verbose=0,
                            vertices=1000,
                            maxiters=4000):
    subtract = t0//P*P
    t0 -= subtract


    if test_knots is None:
        # add test knots for very sharp eclipses
        test_knots = [None, np.array([-0.49, -0.01, 0.01, 0.49]) ]
    else:
        test_knots = [test_knots,  np.array([-0.49, -0.01, 0.01, 0.49])]

    # Collect results from all polyfits
    all_phase = []
    all_polyfit = []
    all_out = []
    all_t0new = []
    all_BIC = []
    all_chi2 = []
    all_reduced_chi2 = []
    all_order = []
    all_find_knots = []

    chain_length = 2

    params = []
    for initial_knots in test_knots:
        for find_knots in [False,True]:
            for order in [2,4]:
                params.append([find_knots,order,initial_knots])

    if verbose==1: print('Fitting polyfits...')
    freezedfitting = partial(fit_polyfit_and_correct_epoch, lc=lc,t0=t0,P=P,
                                                    subtract=subtract,
                                                    chain_length=chain_length,
                                                    debug=debug,
                                                    vertices=vertices,
                                                    maxiters=maxiters)

    '''
    ncores = multiprocessing.cpu_count()
    with multiprocessing.Pool(processes=ncores) as p:
        for result in p.imap( freezedfitting, params ):
            all_phase.append(result[0])
            all_polyfit.append(result[1])
            all_out.append(result[2])
            all_t0new.append(result[3])
            all_BIC.append(result[4])
            all_chi2.append(result[5])
            all_reduced_chi2.append(result[6])
            all_order.append(result[7])
            all_find_knots.append(result[8])
    '''

    ncores = multiprocessing.cpu_count()
    executor = ProcessKillingExecutor(max_workers=ncores)
    generator = executor.map(freezedfitting, params, timeout=timeout,
                             callback_timeout=fun_timeout)
    for result in generator:
        if result is not None:
            all_phase.append(result[0])
            all_polyfit.append(result[1])
            all_out.append(result[2])
            all_t0new.append(result[3])
            all_BIC.append(result[4])
            all_chi2.append(result[5])
            all_reduced_chi2.append(result[6])
            all_order.append(result[7])
            all_find_knots.append(result[8])

    '''
    for par  in params:
        result = fit_polyfit_and_correct_epoch(par, lc=lc,t0=t0,P=P,subtract=subtract,chain_length=chain_length,debug=debug,
        vertices=vertices,
        maxiters=maxiters)
        all_phase.append(result[0])
        all_polyfit.append(result[1])
        all_out.append(result[2])
        all_t0new.append(result[3])
        all_BIC.append(result[4])
        all_chi2.append(result[5])
        all_reduced_chi2.append(result[6])
        all_order.append(result[7])
        all_find_knots.append(result[8])
    '''

    # --- Choose best model based on Chi2 and BIC ---
    if verbose==1: print('Comparing polyfit models...')
    (final_phase, final_polyfit, final_t0new,
    final_out,final_knots,final_find_knots,
    bestBIC, best_chi2, best_reduced_chi2, final_order,
    do_fit_again_final_knots, do_fit_again_fixed_knots,
    one_only_sharp_peak) = compare_results_and_choose_fit(all_phase,all_polyfit,all_out,all_t0new,
                                    all_BIC,all_chi2,all_reduced_chi2,all_order,all_find_knots,
                                    debug=debug,lc=lc,P=P)

    if one_only_sharp_peak:
        # best fit has one_only_sharp_peak check fit again
        if verbose==1: print('Best fit has one only sharp peak -> checking fit again')

        initial_knots = None
        find_knots = False
        order = 2
        chain_length = 2

        # Initilaize results with previously best fit parameters
        all_phase = [final_phase]
        all_polyfit = [final_polyfit]
        all_out = [final_out]
        all_t0new = [final_t0new]
        all_BIC = [bestBIC]
        all_chi2 = [best_chi2]
        all_reduced_chi2 = [best_reduced_chi2]
        all_order = [final_order]
        all_find_knots = [final_find_knots]

        (phase,polyfit,out,
        t0new, BIC,chi2,reduced_chi2,
        order,_) = fit_polyfit_and_correct_epoch([find_knots,order,
                                                initial_knots],
                                                lc,t0,P,subtract,
                                                chain_length,debug,
                                                vertices=vertices,
                                                maxiters=maxiters)

        all_phase.append(phase)
        all_polyfit.append(polyfit)
        all_out.append(out)
        all_t0new.append(t0new)
        all_BIC.append(BIC)
        all_chi2.append(chi2)
        all_reduced_chi2.append(reduced_chi2)
        all_order.append(order)
        all_find_knots.append(find_knots)

        if  np.all(np.isnan(phase)):
            # Just return
            try:
                if do_fit_again_final_knots:
                    return subtract,final_t0new,final_phase,final_polyfit,do_fit_again_final_knots,do_fit_again_fixed_knots,final_knots,final_out
                else:
                    return subtract,final_t0new,final_phase,final_polyfit,do_fit_again_final_knots,do_fit_again_fixed_knots,[],final_out
            except UnboundLocalError as e:
                # All fits went wrong, probably because phase space is sparsely covered
                return np.nan,np.nan,np.nan,np.nan,False,False,[],np.nan


        # --- Choose best model based on Chi2 and BIC ---
        (final_phase, final_polyfit, final_t0new,
        final_out,final_knots,final_find_knots,
        bestBIC, best_chi2, best_reduced_chi2, final_order,
        do_fit_again_final_knots, do_fit_again_fixed_knots,
        one_only_sharp_peak) = compare_results_and_choose_fit(all_phase,all_polyfit,all_out,all_t0new,
                                        all_BIC,all_chi2,all_reduced_chi2,all_order,all_find_knots,
                                        debug=debug,lc=lc,P=P)


    try:
        if do_fit_again_final_knots:
            return subtract,final_t0new,final_phase,final_polyfit,do_fit_again_final_knots,do_fit_again_fixed_knots,final_knots,final_out
        else:
            return subtract,final_t0new,final_phase,final_polyfit,do_fit_again_final_knots,do_fit_again_fixed_knots,[],final_out
    except UnboundLocalError as e:
        # All fits went wrong, probably because phase space is sparsely covered
        return np.nan,np.nan,np.nan,np.nan,False,False,[],np.nan

def min_bin_cal(per):
    P = per/365.25
    R = 0.00465047 # AU
    M = 2 #S mass
    G = 39.478 # AU^3 * yr^(-2) * M_sun^(-1)

    a = ((G*M*P**2)/ (4*np.pi**2))**(1/3)
    alpha = np.arctan(R / a) * 2
    dur = alpha * 2 * P / (2 * np.pi)
    min_bin_num = 10.0/dur*P
    if min_bin_num <50 :
        min_bin_num = 50
    elif min_bin_num >2000:
        min_bin_num = 2000

    return min_bin_num

def check_if_fit_is_above_data(polyfit,lc,t0,P):
    subtract = t0//P*P
    t0 -= subtract
    x,y,z = lc[0].copy(),lc[1].copy(),lc[2].copy()
    x,y,z = preprocess_polyfit(x,y,z,t0,P)

    badpts = polyfit > y.max()
    if np.sum(badpts) > 0:
        polyfit[badpts] = np.max(polyfit[~badpts])

    return polyfit

class Polyfitter:
    def __init__(self,scale=None,debug=False):
        """
        Class to perform polynomial chain fitting and
        classification based on light curve morphology.

        Parameters
        ----------
        scale : "mag" or "flux"
            The scale of the input data that will be used with this instance.
        debug : bool, default: False
            If `True` each fit will be displayed with auxiliary messages.
        """
        if scale is None:
            raise KeyError('Please specify scale="flux" or "mag"')
        else:
            self.scale = scale

        # Turn on verbosity for debugging
        self.debug = debug

        if debug:
            import matplotlib.pyplot as plt

        # --- Load LLE to 3D transformer ---
        transformerfile = Path(PACKAGEDIR, "transformers", "LLE_transformer.pickle")
        with open(transformerfile,'rb') as f:
            self.LLE = pickle.load(f)

        # --- Load LLE to 2D transformer ---
        transformerfile = Path(PACKAGEDIR, "transformers", "LLE_3Dto2D_transformer.pickle")
        with open(transformerfile,'rb') as f:
            self.LLE2 = pickle.load(f)

        # --- Load thetaScaler (2D to c) ---
        transformerfile = Path(PACKAGEDIR, "transformers", "thetaScaler.pickle")
        with open(transformerfile,'rb') as f:
            self.thetaScaler = pickle.load(f)

    def get_polyfit(self,time,flux,error,period,epoch,
                    verbose=1,
                    vertices=1000,
                    maxiters=4000,
                    timeout=100):
        """
        ``get_polyfit`` performs polynomial chain fitting by running
        several fits parallel, then choosing the best one based on
        Bayesian Information Criterion.

        This code is built upon the original `polyfit` by
        Prsa et al. (2008), ApJ 687, 542.

        Parameters
        ----------
        time : array of floats
            Time values.
        flux : array of floats
            Flux or magnitude values for every time point.
        error : array of floats
            Uncertainty on each flux or magnitude data point.
        period: float
            The orbital period of the binary star.
        epoch : float
            The epoch of the binary star.
        verbose: int, 0 or 1, default: 1
            If `0` the fits will be done silently.
        vertices: int, default: 1000
            Number of equidistant vertices in the computed fit.
        maxiters: int, default: 4000
            Maximum number of iterations.
        timeout: seconds, default: 100
            The time in seconds after a fit will be terminated.

        Returns
        -------
        t0new : float
            The new epoch based on the minimum of the best polyfit.
        phase : array of floats
            Phase values of the best polyfit.
        polyfit : array of floats
            Flux/magnitude values of the best polyfit.
        messages : list
            The messages that the original `polyfit` would create during fitting.
        """

        # ----------- Prepare data ---------
        if self.scale=='mag':
            flux = -flux+ 2*np.median(flux)
            lc = np.ascontiguousarray( np.c_[time,flux,error].T )
        elif self.scale=='flux':
            lc    = np.ascontiguousarray( np.c_[time,flux,error].T )

        # ----------- First Polyfit ---------
        (subtract,t0new,phase,polyfit,
        do_fit_again_final_knots,
        do_fit_again_fixed_knots,
        final_knots, final_out) = find_best_polyfit(lc,epoch,period,
                                                        test_knots=None,
                                                        debug=self.debug,
                                                        verbose=verbose,
                                                        vertices=vertices,
                                                        maxiters=maxiters,
                                                        timeout=timeout)

        # ----------- Second Polyfit with fixed knots ---------
        test_knots = np.array([-0.3, -0.2, 0.2, 0.3])

        if do_fit_again_fixed_knots:
            if self.debug: print('Doing fit again w/ fixed knots...')

            (subtract,t0new,phase,polyfit,
            do_fit_again_final_knots,
            do_fit_again_fixed_knots,
            final_knots, final_out) = find_best_polyfit(lc,epoch,period,
                                                        test_knots=test_knots,
                                                        debug=self.debug,
                                                        verbose=verbose,
                                                        vertices=vertices,
                                                        maxiters=maxiters,
                                                        timeout=timeout)


        # ----------- Third Polyfit with fixed final knots ---------
        if do_fit_again_final_knots:
            if self.debug: print('Doing the fit again w/ final knots..')

            (subtract,t0new,phase,polyfit,
            do_fit_again_final_knots,
            do_fit_again_fixed_knots,
            _, final_out) = find_best_polyfit(lc,epoch,period,
                                                test_knots=final_knots,
                                                debug=self.debug,
                                                verbose=verbose,
                                                vertices=vertices,
                                                maxiters=maxiters,
                                                timeout=timeout)

        # --- Return nan if fit failed ---
        if np.isnan(subtract) or np.all(np.isnan(polyfit)):
            return np.nan,np.nan,np.nan,[]

        # --- Final Polyfit check ---
        polyfit = check_if_fit_is_above_data(polyfit,lc,t0new,period)

        # --- Transform Polyfit to match data ---
        if self.scale=='mag':
            polyfit = -polyfit+ 2*np.median(flux)

        self.polyfit = polyfit

        return t0new+subtract,phase,polyfit,final_out

    def _convert_polyfit_to_morph(self):
        # --- Convert all curves to [0,1] range ---
        X = np.atleast_2d(self.polyfit)
        X = ( (X.transpose() - np.min(X,axis=1) )/np.ptp(X,axis=1) ).transpose()

        if self.debug: print('LLE transform...')
        X_lle = self.LLE.transform(X)

        if self.debug: print('LLE2 transform...')
        X_lle2 = self.LLE2.transform(X_lle)

        # ---- 2D LLE offset -------
        xval = X_lle2[:,0] *-1
        yval = X_lle2[:,1] - 0.03

        if self.debug: print('thetaX transform...')
        # Calculate r, theta polar coordinates
        theta = np.arctan( yval/xval ) / (2*np.pi) * 360
        # Transform theta to 0-180 deg
        theta[ theta<0 ] += 180
        # To get c=0 for W Uma and c=1 for algol
        theta = 180 - theta
        # Scale theta to 0-1 interval -> c parameter
        thetaX = self.thetaScaler.transform( theta.reshape(-1,1) )
        thetaX = thetaX.reshape(-1)

        # ---- Polynomial transformation -------
        theta2c = np.poly1d([ 1.12966346e+02, -4.55409111e+02,  7.52059680e+02, -6.44493820e+02,
                              2.98655212e+02, -7.12443759e+01,  8.45086131e+00,  3.51164494e-02])
        thetaX = theta2c(thetaX)

        return thetaX

    @property
    def c(self):
        """
        Returns the morphology parameter.
        """
        # ----------- Prepare polyfit ---------
        if self.scale=='mag':
            self.polyfit = -self.polyfit
        elif self.scale=='flux':
            self.polyfit = 2.5*np.log10(self.polyfit) + 25.

        morph = self._convert_polyfit_to_morph()

        return morph

    def get_c(self,polyfits):
        """
        Returns the morphology parameter of a given (set of) polyfit(s).

        Parameters
        ----------
        polyfits : ndarray of polyfits
            The previously preformed ``polyfit`` with 1000 equidistant points.
            If it is only one ``polyfit``, then it can be a simple 1D array.
            If it is more than one ``polyfit``, then the shape of the array
            must be (n,1000), where n is the number of ``polyfits``.

        Returns
        -------
        morph : array of float(s)
            The morphology parameter for each given ``polyfit``.
        """
        # ----------- Prepare polyfit ---------
        if self.scale=='mag':
            self.polyfit = -polyfits
        elif self.scale=='flux':
            self.polyfit = 2.5*np.log10(polyfits) + 25.

        morph = self._convert_polyfit_to_morph()

        return morph
