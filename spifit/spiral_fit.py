#! /usr/bin/env python

"""
Module required for spiral fitting
"""

#import itertools as itt
import math
from multiprocessing import cpu_count #Pool, 
import numpy as np
from numpy.polynomial.polynomial import polyval
import pdb
from scipy.optimize import minimize
from vip_hci.conf import time_ini, timing
from vip_hci.conf.utils_conf import pool_map, iterable #eval_func_tuple as EFT
from vip_hci.specfit import find_nearest
import vip_hci
from .bootstrap_uncertainties import bootstrap_unc

__all__ = ['spiral_minimize',
           'chisquare',
           'spiral_eq_eval',
           'spiral_eq_definition',
           'spirals_muto_hc_linsearch',
           'spirals_muto_hc_beta_linsearch',
           'spirals_muto_rc_thc_hc_linsearch',
           'spirals_muto_rc_thc_hc_beta_linsearch',
           'spirals_muto_simult_linsearch',
           'spirals_muto_simult_linsearch_gamma',
           'find_params_unc',
           'measure_pitch_angle'
           ]


def spiral_minimize(p, spi_trace, errors, method='Nelder-Mead', fit_eq='gen_archi', 
                    options={'xtol':1e-1, 'maxiter':4000, 'maxfev':8000}, 
                    verbose=False, bounds=None, fac_hr=100, **kwargs):
    """
    Determines the best fit parameters of a spiral, depending on the model assumed (general Archimedean spiral, logarithmic spiral)
    
    Parameters
    ----------
    
    p : np.array
        Estimate of the spiral parameters, e.g. in the following format for general Archimede spirals:
        np.array([A_spi,n_spi, r0_spi])
    spi_trace: numpy.array
        Spiral trace in polar coordinates, e.g. in the following format for general Archimede spirals:
        np.array([theta_spi_1,r_spi])
    errors: np.array
        Error on r for each point of the spiral (on theta for Muto12 fit), e.g. in the following format for general Archimede spirals:
        np.array([err_pt1_spi,err_pt2_spi, err_pt3_spi])
    method: str, optional
        Solver to be used from scipy.minimize. Default: 'Nelder-Mead'.
        See description at: 
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
    fit_eq: string, {'gen_archi', 'log', 'lin_archi', 'poly', 'muto12', 
                     'muto12_4params', 'muto12_3params', 'muto12_2params',
                     'muto12_2params_hc_beta', 'muto12_1param'}
        Type of equation to which the coordinate is evaluated.
    options: dict, optional
        The scipy.optimize.minimize options.
    verbose : boolean, optional
        If True, informations are displayed in the shell.
        
    Returns
    -------
    out : np.array
        Parameters of the best fit to Archimedean model, for each spiral.
        
    """
    
    # Check the input parameters are in the right format
    if fit_eq not in {'gen_archi', 'log', 'lin_archi', 'muto12', 
                      'muto12_4params','muto12_3params', 'muto12_2params',
                      'muto12_2params_hc_beta', 'muto12_1param', 'poly'}:
        raise ValueError('Pick a valid spiral equation type.')
    
    nparams = p.shape[0]
    if fit_eq == 'gen_archi' and nparams != 3:
        raise ValueError('There should be exactly 3 parameters to fit with a general Archimedean spiral equation.')
    elif fit_eq == 'log' and nparams != 2:
        raise ValueError('There should be exactly 2 parameters to fit with a logarithmic spiral equation.')
    elif fit_eq == 'lin_archi' and nparams != 2:
        raise ValueError('There should be exactly 3 parameters to fit with a linear Archimedean spiral equation.')
    elif fit_eq == 'muto12' and nparams != 5:
        raise ValueError('There should be exactly 5 parameters to fit with Muto+2012 spiral equation.')
    elif fit_eq == 'muto12_4params' and nparams != 4:
        raise ValueError('There should be exactly 4 parameters to fit with Muto+2012 spiral equation (3 params).')   
    elif fit_eq == 'muto12_3params' and nparams != 3:
        raise ValueError('There should be exactly 3 parameters to fit with Muto+2012 spiral equation (3 params).')   
    elif fit_eq == 'muto12_2params' and nparams != 2:
        raise ValueError('There should be exactly 2 parameters to fit with Muto+2012 spiral equation (3 params).')   
    elif fit_eq == 'muto12_1param' and nparams != 1:
        raise ValueError('There should be exactly 2 parameters to fit with Muto+2012 spiral equation (1 param).')   
        
    if verbose:
        print('')
        print('{} minimization is running...'.format(options.get('method',method)))
    
    clockwise = kwargs.get('clockwise',-1) 
    alpha = kwargs.get('alpha',None)   
    beta = kwargs.get('beta',None)  
    h_c = kwargs.get('h_c',None)     
    r_c = kwargs.get('r_c',None)   
    th_c = kwargs.get('th_c',None) 
    
    res = minimize(chisquare, p, args=(spi_trace, errors, fit_eq, False, 
                                       clockwise, alpha, beta, h_c, r_c, th_c,
                                       fac_hr), 
                   method=options.pop('method',method), bounds=bounds,
                   options=options)

    best_fit_params = res.x

    if verbose:
        msg = 'Success: {}, nit: {}, nfev: {}, chi2r: {}'
        print(msg.format(res.success,res.nit,res.nfev,res.fun))
        print('message: {}'.format(res.message))
        print(res.x)

    return best_fit_params
    
    
def spirals_muto_simult_linsearch(rc_test, thc_test, hc_test, spi_trace, errors, 
                                  alpha, beta, verbose=False, nproc=2, same_weight=False,
                                  full_output=False,**kwargs):
    """
    Determines the best fit parameters of several spirals in a same system with Muto's equation (N_spirals*2 + 1 free params).
    h_c is fixed to the same value for all spirals.
    
    Parameters
    ----------
    
    rc_test : list of 1D numpy.array
        Range of the "rc" spiral parameter for the linear search, for each spiral:
        2 spirals example: [np.array([rc_to_be_tested_spi1]), np.array([rc_to_be_tested_spi2])]
    thc_test: list of 1D numpy.array
        Range of the "theta_c" spiral parameter for the linear search, for each spiral:
        2 spirals example: [np.array([thc_to_be_tested_spi1]), np.array([thc_to_be_tested_spi2])]
    hc_test: 1D numpy.array
        Range of the "h_c" spiral parameter for the linear search (same for all spirals)
    spi_trace: list of 2D numpy.array
        Spiral trace in polar coordinates, e.g. in the following format:
        [np.array([[[theta_spi1],[r_spi1]]]),np.array([[[theta_spi2],[r_spi2]]])]
    errors: list of 1D numpy.array
        Error on theta for each point of the spiral(s), e.g.:
        [np.array([err_pt1_spi1,err_pt2_spi1, ..., err_ptN_spi1),np.array([err_pt1_spi2,err_pt2_spi2, ..., err_ptN_spi2)]
    alpha, beta: floats
        Muto's equation alpha and beta parameters
    verbose : boolean, optional
        If True, informations are displayed in the shell.
    nproc: int
        Number of processors to be used. Default is to use multiprocessing.
        0/None -> nproc/2 (computed by cpu_count() function)
        1  -> single processing
        >1 -> multiprocessing
    same_weight: bool, opt
        Whether the mean chi_square calculated over the different spirals should 
        not be weighted (same_weight=True) or weighted by the number of points 
        considered in each spiral trace (same_weight=False).
        
    Returns
    -------
    out : np.array
        Parameters of the best fit to Muto's equation, for each spiral.
        
    """
    
    # Check the input parameters are in the right format
    
    start_time = time_ini()    
    
    if type(rc_test) is list and type(thc_test) is list and type(spi_trace) is list and type(errors) is list:
        nspi = len(rc_test)
        if len(thc_test) != nspi or len(spi_trace) != nspi or len(errors) != nspi:
            raise ValueError('The length of all the lists should be the same: {:} .'.format(nspi))
        else:
            nrc_tests = []
            nthc_tests = []
            for nn in range(nspi):
                nrc_tests.append(rc_test[nn].shape[0])
                nthc_tests.append(thc_test[nn].shape[0])
            nhc_tests = hc_test.shape[0]
    else:
       raise ValueError('rc_test, thc_test, spi_trace and errors shuold all be lists of the same length (nspi).')
        
    if verbose:
        print('')
        print('Linear search is running...')
    
    clockwise = kwargs.get('clockwise',-1)

    
    if not nproc:
        nproc = int(cpu_count()/2.)
    
    chi_sq_arr_all = []    
    
    if nproc == 1:
        for nn in range(nspi):
            chi_sq_arr = np.zeros([nhc_tests,nrc_tests[nn],nthc_tests[nn]])
            if verbose:
                print("***** Tests for spiral {:}/{:} *****".format(nn, nspi))
                timing(start_time)
            for hh in range(nhc_tests):
                h_c = hc_test[hh]
                if verbose:
                    print("----- Testing h_c = {:} (case {:}/{:}) -----".format(h_c, hh+1, nhc_tests))
                    timing(start_time)
                for rr in range(nrc_tests[nn]):
                    r_c = rc_test[nn][rr]
                    if verbose:
                        print(" Testing r_c = {:} (case {:}/{:}) ".format(rc_test[nn][rr], rr+1, nrc_tests[nn]))
                        timing(start_time)
                    for tt in range(nthc_tests[nn]):
                        p = np.array([thc_test[nn][tt],r_c,h_c])
                        chi_sq_arr[hh,rr,tt] = chisquare(p, spi_trace[nn], errors[nn], 'muto12_3params', True, clockwise, alpha, beta)
            chi_sq_arr_all.append(chi_sq_arr)
    elif nproc > 1:     
        for nn in range(nspi):
            # create list of input test params:
            if verbose:
                print(" *** Creating list of test input parameters for spiral {:}/{:}... ***".format(nn+1,nspi))
            p = []
            for hh in range(nhc_tests):
                for rr in range(nrc_tests[nn]):
                    for tt in range(nthc_tests[nn]):
                        p.append(np.array([thc_test[nn][tt],rc_test[nn][rr],hc_test[hh]]))
            # Do multiprocessing evaluation
            if verbose:
                print("*** Evaluating all test parameters for spiral {:}/{:}... ***".format(nn+1,nspi))
            #pool = Pool(processes=int(nproc))  
            res = pool_map(nproc, chisquare, iterable(p), spi_trace[nn],
                           errors[nn], 'muto12_3params', True,clockwise,
                           alpha, beta)
            res = np.array(res)
            #pool.close()
            chi_sq_arr_all.append(res.reshape([nhc_tests,nrc_tests[nn],nthc_tests[nn]]))
    else:
        raise ValueError('nproc should be a positive integer')     

    # GET THE OPTIMAL CHI SQUARE FOR EACH h_c and each SPIRAL
    opt_chi_sq_1 = np.zeros([nhc_tests,nspi])
    opt_params_1 = np.zeros([nhc_tests,nspi,2])
    for nn in range(nspi):
        for hh in range(nhc_tests):
            opt_chi_sq_1[hh,nn] = np.amin(chi_sq_arr_all[nn][hh,:,:])
            idx_rc_min, idx_thc_min = np.unravel_index(np.argmin(chi_sq_arr_all[nn][hh,:,:]),(nrc_tests[nn],nthc_tests[nn]))
            opt_params_1[hh,nn,0] = rc_test[nn][idx_rc_min]
            opt_params_1[hh,nn,1] = thc_test[nn][idx_thc_min]

    # THEN GET THE WEIGHTED/MEAN CHI SQUARE OVER ALL SPIRALS, for each h_c
    opt_chi_sq_2 = np.zeros([nhc_tests])
    if same_weight:
        opt_chi_sq_2 = np.mean(opt_chi_sq_1, axis=1)
    else:
        npt_spi = np.zeros(nspi)
        npt_total = 0
        for nn in range(nspi):
            npt_spi[nn] = spi_trace[nn].shape[1]
            npt_total += npt_spi[nn]
        for nn in range(nspi):
            opt_chi_sq_2 = opt_chi_sq_2 + (opt_chi_sq_1[:,nn]*npt_spi[nn]/npt_total)

    # Finally, find the optimal h_c for which the weighted/mean chi_sq is min
    opt_chi_sq_3 = np.amin(opt_chi_sq_2)
    idx_opt_hc = np.argmin(opt_chi_sq_2)
    opt_params_3 = opt_params_1[idx_opt_hc,:,:] # final optimal 
    if verbose:
        print("Optimal chi_sq: {:.4f}".format(opt_chi_sq_3))
        print("Optimal separate chi_sq of each spiral: ", opt_chi_sq_1[idx_opt_hc])
        print("Optimal h_c: {:.3f}".format(hc_test[idx_opt_hc]))
        if idx_opt_hc == 0 or idx_opt_hc == nhc_tests-1:
            print("!!! WARNING: the optimal h_c was found at the limit of the tested range.")
        for nn in range(nspi):
            print("Optimal (r_c,th_c) parameters for spiral {:}: ({:.2f},{:.2f})".format(nn+1,opt_params_3[nn,0],opt_params_3[nn,1]))
            if opt_params_3[nn,0] == rc_test[nn][0] or opt_params_3[nn,0] == rc_test[nn][-1]:
                print("!!! WARNING: the optimal r_c was found at the limit of the tested range for spiral {:}.".format(nn+1))
            if opt_params_3[nn,1] == thc_test[nn][0] or opt_params_3[nn,1] == thc_test[nn][-1]:
                print("!!! WARNING: the optimal th_c was found at the limit of the tested range for spiral {:}.".format(nn+1))
                
    if verbose:
        print("FINAL ELAPSED TIME:")  
        timing(start_time)

    if full_output:
        return opt_chi_sq_3, hc_test[idx_opt_hc], opt_params_3, opt_chi_sq_1, opt_params_1
    else:
        return opt_chi_sq_3, hc_test[idx_opt_hc], opt_params_3
  

def spirals_muto_simult_linsearch_gamma(rc_test, thc_test, hc100_test, gamma_test, dist, spi_trace, errors, 
                                        alpha, beta, verbose=False, nproc=2, same_weight=False,
                                        full_output=False,pix_to_dist_factor=1.,**kwargs):
    """
    Determines the best fit parameters of several spirals in a same system with Muto's equation (N_spirals*2 + 1 free params).
    hc100 is fixed to the same value for all spirals, with hc = hc100 * (r/100au)^gamma
    
    Parameters
    ----------
    
    rc_test : list of 1D numpy.array
        Range of the "rc" spiral parameter for the linear search, for each spiral:
        2 spirals example: [np.array([rc_to_be_tested_spi1]), np.array([rc_to_be_tested_spi2])]
    thc_test: list of 1D numpy.array
        Range of the "theta_c" spiral parameter for the linear search, for each spiral:
        2 spirals example: [np.array([thc_to_be_tested_spi1]), np.array([thc_to_be_tested_spi2])]
    hc_test: 1D numpy.array
        Range of the "h_c" spiral parameter for the linear search (same for all spirals)
    spi_trace: list of 2D numpy.array
        Spiral trace in polar coordinates, e.g. in the following format:
        [np.array([[[theta_spi1],[r_spi1]]]),np.array([[[theta_spi2],[r_spi2]]])]
    errors: list of 1D numpy.array
        Error on theta for each point of the spiral(s), e.g.:
        [np.array([err_pt1_spi1,err_pt2_spi1, ..., err_ptN_spi1),np.array([err_pt1_spi2,err_pt2_spi2, ..., err_ptN_spi2)]
    alpha, beta: floats
        Muto's equation alpha and beta parameters
    verbose : boolean, optional
        If True, informations are displayed in the shell.
    nproc: int
        Number of processors to be used. Default is to use multiprocessing.
        0/None -> nproc/2 (computed by cpu_count() function)
        1  -> single processing
        >1 -> multiprocessing
    same_weight: bool, opt
        Whether the mean chi_square calculated over the different spirals should 
        not be weighted (same_weight=True) or weighted by the number of points 
        considered in each spiral trace (same_weight=False).
        
    Returns
    -------
    out : np.array
        Parameters of the best fit to Muto's equation, for each spiral.
        
    """
    
    # Check the input parameters are in the right format
    
    start_time = time_ini()    
    
    if type(rc_test) is list and type(thc_test) is list and type(spi_trace) is list and type(errors) is list:
        nspi = len(rc_test)
        if len(thc_test) != nspi or len(spi_trace) != nspi or len(errors) != nspi:
            raise ValueError('The length of all the lists should be the same: {:} .'.format(nspi))
        else:
            nrc_tests = []
            nthc_tests = []
            for nn in range(nspi):
                nrc_tests.append(rc_test[nn].shape[0])
                nthc_tests.append(thc_test[nn].shape[0])
            nhc100_tests = hc100_test.shape[0]
            ngamma_tests = gamma_test.shape[0]
    else:
       raise ValueError('rc_test, thc_test, spi_trace and errors should all be lists of the same length (nspi).')
    
    
    if verbose:
        print('')
        print('Linear search is running...')
    
    clockwise = kwargs.get('clockwise',-1)

    if not nproc:
        nproc = int(cpu_count()/2.)
    
    chi_sq_arr_all = []    
    
    # Special case if both hc100_test and gamma_test have only one element (i.e. they are fixed)
    if nhc100_tests == 1 and ngamma_tests == 1:
        eq = 'muto12_2params'
    else:
        eq = 'muto12_3params'
    
    if nproc == 1:
        for nn in range(nspi):
            chi_sq_arr = np.zeros([nhc100_tests,ngamma_tests,nrc_tests[nn],nthc_tests[nn]])
            if verbose:
                print("***** Tests for spiral {:}/{:} *****".format(nn, nspi))
                timing(start_time)
            for hh in range(nhc100_tests):
                hc_100 = hc100_test[hh]
                if verbose:
                    print("----- Testing hc_100 = {:} (case {:}/{:}) -----".format(hc_100, hh+1, nhc100_tests))
                    timing(start_time)
                for gg in range(ngamma_tests):
                    gamma = gamma_test[gg]
                    if verbose:
                        print("----- Testing gamma = {:} (case {:}/{:}) -----".format(gamma, gg+1, ngamma_tests))
                        timing(start_time)
                    for rr in range(nrc_tests[nn]):
                        r_c = rc_test[nn][rr]
                        h_c = hc_100*(r_c/(100.*(dist/200.)/pix_to_dist_factor))**gamma # assumes value of hc_100 is given for 100au for a dist of 200pc, while new dist estimate might change it
                        if verbose:
                            print(" Testing r_c = {:} (case {:}/{:}) ".format(rc_test[nn][rr], rr+1, nrc_tests[nn]))
                            timing(start_time)
                        for tt in range(nthc_tests[nn]):
                            if eq == 'muto12_3params':
                                p = np.array([thc_test[nn][tt],r_c,h_c])
                            else:
                                p = np.array([thc_test[nn][tt],r_c])
                            chi_sq_arr[hh,gg,rr,tt] = chisquare(p, spi_trace[nn], errors[nn], eq, True, clockwise, alpha, beta, h_c)
                    chi_sq_arr_all.append(chi_sq_arr)
    elif nproc > 1:     
        for nn in range(nspi):
            # create list of input test params:
            if verbose:
                print(" *** Creating list of test input parameters for spiral {:}/{:}... ***".format(nn+1,nspi))
            p = []
            for hh in range(nhc100_tests):
                hc_100 = hc100_test[hh]
                for gg in range(ngamma_tests):
                    gamma = gamma_test[gg]
                    for rr in range(nrc_tests[nn]):
                        r_c = rc_test[nn][rr]
                        h_c = hc_100*(r_c/(100.*(dist/200.)/pix_to_dist_factor))**gamma
                        for tt in range(nthc_tests[nn]):
                            if eq == 'muto12_3params':
                                p.append(np.array([thc_test[nn][tt],rc_test[nn][rr],h_c]))
                            else:
                                p.append(np.array([thc_test[nn][tt],rc_test[nn][rr]]))
            # Do multiprocessing evaluation
            if verbose:
                print("*** Evaluating all test parameters for spiral {:}/{:}... ***".format(nn+1,nspi))
            #pool = Pool(processes=int(nproc))  
            res = pool_map(nproc, chisquare, iterable(p), spi_trace[nn],
                           errors[nn], eq, True, clockwise, alpha, beta, h_c)
            res = np.array(res)
            #pool.close()
            chi_sq_arr_all.append(res.reshape([nhc100_tests,ngamma_tests,nrc_tests[nn],nthc_tests[nn]]))
    else:
        raise ValueError('nproc should be a positive integer')     


    # GET THE OPTIMAL CHI SQUARE FOR EACH h_c and each SPIRAL
    opt_chi_sq_1 = np.zeros([nhc100_tests,ngamma_tests,nspi])
    opt_params_1 = np.zeros([nhc100_tests,ngamma_tests,nspi,2])
    for nn in range(nspi):
        for hh in range(nhc100_tests):
            for gg in range(ngamma_tests):
                opt_chi_sq_1[hh,gg,nn] = np.amin(chi_sq_arr_all[nn][hh,gg,:,:])
                idx_rc_min, idx_thc_min = np.unravel_index(np.argmin(chi_sq_arr_all[nn][hh,gg,:,:]),(nrc_tests[nn],nthc_tests[nn]))
                opt_params_1[hh,gg,nn,0] = rc_test[nn][idx_rc_min]
                opt_params_1[hh,gg,nn,1] = thc_test[nn][idx_thc_min]

    # THEN GET THE WEIGHTED/MEAN CHI SQUARE OVER ALL SPIRALS, for each h_c
    opt_chi_sq_2 = np.zeros([nhc100_tests,ngamma_tests])
    if same_weight:
        opt_chi_sq_2 = np.mean(opt_chi_sq_1, axis=2)
    else:
        npt_spi = np.zeros(nspi)
        npt_total = 0
        for nn in range(nspi):
            npt_spi[nn] = spi_trace[nn].shape[1]
            npt_total += npt_spi[nn]
        for nn in range(nspi):
            opt_chi_sq_2 = opt_chi_sq_2 + (opt_chi_sq_1[:,:,nn]*npt_spi[nn]/npt_total)

    # Finally, find the optimal h_c for which the weighted/mean chi_sq is min
    opt_chi_sq_3 = np.amin(opt_chi_sq_2)
    idx_opt_hc100,idx_opt_gamma = np.unravel_index(np.argmin(opt_chi_sq_2),(nhc100_tests,ngamma_tests))
    opt_params_3 = opt_params_1[idx_opt_hc100,idx_opt_gamma,:,:] # final optimal 
    if verbose:
        print("Optimal chi_sq: {:.4f}".format(opt_chi_sq_3))
        print("Optimal separate chi_sq of each spiral: ", opt_chi_sq_1[idx_opt_hc100,idx_opt_gamma])
        print("Optimal h_c: {:.3f}".format(hc100_test[idx_opt_hc100]))
        print("Optimal gamma: {:.3f}".format(gamma_test[idx_opt_gamma]))
        if idx_opt_hc100 == 0 or idx_opt_hc100 == nhc100_tests-1:
            print("!!! WARNING: the optimal h_c was found at the limit of the tested range.")
        if idx_opt_gamma == 0 or idx_opt_gamma == ngamma_tests-1:
            print("!!! WARNING: the optimal gamma was found at the limit of the tested range.")
        for nn in range(nspi):
            print("Optimal (r_c,th_c) parameters for spiral {:}: ({:.2f},{:.2f})".format(nn+1,opt_params_3[nn,0],opt_params_3[nn,1]))
            if opt_params_3[nn,0] == rc_test[nn][0] or opt_params_3[nn,0] == rc_test[nn][-1]:
                print("!!! WARNING: the optimal r_c was found at the limit of the tested range for spiral {:}.".format(nn+1))
            if opt_params_3[nn,1] == thc_test[nn][0] or opt_params_3[nn,1] == thc_test[nn][-1]:
                print("!!! WARNING: the optimal th_c was found at the limit of the tested range for spiral {:}.".format(nn+1))
                
    if verbose:
        print("FINAL ELAPSED TIME:")  
        timing(start_time)

    if full_output:
        return opt_chi_sq_3, hc100_test[idx_opt_hc100], gamma_test[idx_opt_gamma], opt_params_3, opt_chi_sq_1, opt_params_1
    else:
        return opt_chi_sq_3, hc100_test[idx_opt_hc100], gamma_test[idx_opt_gamma], opt_params_3
        

def spirals_muto_hc_linsearch(hc_test, spi_trace, errors, alpha, beta, r_c, th_c, 
                              verbose=False, nproc=2, **kwargs):
    """
    Determines the best fit parameters of several individual spirals with a linear search applied to
    Muto's 1 free parameter (h_c) equation.
    
    Parameters
    ----------
    
    hc_test: 1D numpy.array
        Range of the "h_c" spiral parameter for the linear search (same for all spirals)
    spi_trace: 2D numpy.array
        Spiral trace in polar coordinates, e.g. in the following format:
        np.array([theta_spi,r_spi])
    errors: 1D numpy.array or list
        Error on theta for each point of the spiral(s), e.g.:
        np.array([err_pt1_spi1,err_pt2_spi1, ..., err_ptN_spi1])
    alpha, beta, r_c: floats
        Muto's equation alpha, beta and r_c parameters
    th_c: list
        Muto's equation th_c parameter (should be given as a list or 1d array as 
        the basis is different for each spiral)
    verbose : boolean, optional
        If True, informations are displayed in the shell.
    nproc: int
        Number of processors to be used. Default is to use multiprocessing.
        0/None -> nproc/2 (computed by cpu_count() function)
        1  -> single processing
        >1 -> multiprocessing
    same_weight: bool, opt
        Whether the mean chi_square calculated over the different spirals should 
        not be weighted (same_weight=True) or weighted by the number of points 
        considered in each spiral trace (same_weight=False).
        
    Returns
    -------
    out : np.array
        Parameters of the best fit to Muto's equation, for each spiral.
        
    """
    
    # Check the input parameters are in the right format
    
    start_time = time_ini()    
    
    if type(spi_trace) is np.ndarray and (type(errors) is list or type(errors) is np.ndarray):
        npts = spi_trace.shape[1]
        if len(errors) != npts:
            raise ValueError('The length of the errors list should be the same: {:} .'.format(npts))
        elif spi_trace.shape[0] != 2:
            raise ValueError('spiral trace should contain theta and r values in 2 columns')
        else:
            nhc_tests = hc_test.shape[0]
    else:
       raise ValueError('spi_trace or errors do not have the right format.')
        
    if verbose:
        print('Linear search is running...')
    
    clockwise = kwargs.get('clockwise',-1)

    
    if not nproc:
        nproc = int(cpu_count()/2.)
    
    h_c = None # has to be defined to avoid a bug in chisquare    
    
    if nproc == 1:
        chi_sq_arr = np.zeros(nhc_tests)
        if verbose:
            timing(start_time)
        for hh in range(nhc_tests):
            h_c = hc_test[hh]
            if verbose:
                print("----- Testing h_c = {:} (case {:}/{:}) -----".format(h_c, hh+1, nhc_tests))
                timing(start_time)
                p = np.array([h_c])
                chi_sq_arr[hh] = chisquare(p, spi_trace, errors, 'muto12_1param', True, clockwise, alpha, beta, h_c, r_c, th_c)
    elif nproc > 1:     
        # create list of input test params:
        if verbose:
            print(" *** Creating list of test input parameters for spiral... ***")
        p = []
        for hh in range(nhc_tests):
            p.append(np.array([hc_test[hh]]))
        # Do multiprocessing evaluation
        if verbose:
            print("*** Evaluating all test parameters for spiral ... ***")
        #pool = Pool(processes=int(nproc))  
        res = pool_map(nproc, chisquare, iterable(p), spi_trace, errors, 
                       'muto12_1param', True, clockwise, alpha, beta, h_c, r_c,
                        th_c)
        chi_sq_arr = np.array(res)
        #pool.close()
    else:
        raise ValueError('nproc should be a positive integer')     

    # GET THE OPTIMAL CHI SQUARE FOR EACH h_c and each SPIRAL
    opt_chi_sq_3 = np.amin(chi_sq_arr)
    idx_hc_min = np.argmin(chi_sq_arr)
    opt_params_3 = hc_test[idx_hc_min]
    if idx_hc_min == 0 or idx_hc_min == nhc_tests-1:
        print("!!! WARNING: the optimal h_c was found at the limit of the tested range.")

    # Finally, find the optimal h_c for which the weighted/mean chi_sq is min
    if verbose:
        print("Optimal chi_square value: ", opt_chi_sq_3)
        print("Optimal h_c value: ", opt_params_3)
                
    if verbose:
        print("FINAL ELAPSED TIME:")
        timing(start_time)

    return opt_chi_sq_3, opt_params_3
    

def spirals_muto_hc_beta_linsearch(hc_test, beta_test, spi_trace, errors, alpha, r_c, th_c, 
                                   verbose=False, debug=False, **kwargs):
    """
    Determines the best fit parameters of an individual spiral with a linear search applied to
    Muto's 2 free parameters (h_c and beta) equation.
    
    Parameters
    ----------
    
    hc_test: 1D numpy.array
        Range of the "h_c" spiral parameter for the linear search
    beta_test: 1D numpy.array
        Range of the "beta" spiral parameter for the linear search
    spi_trace: list of 2D numpy.array
        Spiral trace in polar coordinates, e.g. in the following format:
        np.array([[[theta_spi1],[r_spi1]]])
    errors: list of 1D numpy.array
        Error on theta for each point of the spiral(s), e.g.:
        np.array([err_pt1_spi1,err_pt2_spi1, ..., err_ptN_spi1)
    alpha, beta, r_c: floats
        Muto's equation alpha, beta and r_c parameters
    th_c: list
        Muto's equation th_c parameter (should be given as a list or 1d array as 
        the basis is different for each spiral)
    verbose : boolean, optional
        If True, informations are displayed in the shell.
    nproc: int
        Number of processors to be used. Default is to use multiprocessing.
        0/None -> nproc/2 (computed by cpu_count() function)
        1  -> single processing
        >1 -> multiprocessing
    same_weight: bool, opt
        Whether the mean chi_square calculated over the different spirals should 
        not be weighted (same_weight=True) or weighted by the number of points 
        considered in each spiral trace (same_weight=False).
        
    Returns
    -------
    out : np.array
        Parameters of the best fit to Muto's equation, for each spiral.
        
    """
    
    # Check the input parameters are in the right format
    
    start_time = time_ini()    
    
    if type(spi_trace) is np.ndarray and (type(errors) is list or type(errors) is np.ndarray):
        npts = spi_trace.shape[1]
        if len(errors) != npts:
            raise ValueError('The length of the errors list should be the same: {:} .'.format(npts))
        elif spi_trace.shape[0] != 2:
            raise ValueError('spiral trace should contain theta and r values in 2 columns')
        else:
            nhc_tests = hc_test.shape[0]
            nbeta_tests = beta_test.shape[0]
    else:
       raise ValueError('spi_trace or errors do not have the right format.')
        
    if verbose:
        print('')
        print('Linear search is running...')
    
    clockwise = kwargs.get('clockwise',-1)
    
    h_c = None # has to be defined to avoid a bug in chisquare    
    
    chi_sq_arr = np.zeros([nhc_tests,nbeta_tests])
    if verbose:
        timing(start_time)
    for hh in range(nhc_tests):
        h_c = hc_test[hh]
        if verbose:
            print("----- Testing h_c = {:} (case {:}/{:}) -----".format(h_c, hh+1, nhc_tests))
            timing(start_time)
        for bb in range(nbeta_tests):
            p = np.array([h_c])
            chi_sq_arr[hh,bb] = chisquare(p, spi_trace, errors, 'muto12_1param', True, clockwise, alpha, bb, h_c, r_c, th_c)

    # GET THE OPTIMAL CHI SQUARE FOR EACH h_c and each SPIRAL
    opt_params_3 = np.zeros(2)
    opt_chi_sq_3 = np.amin(chi_sq_arr)
    idx_hc_min,idx_beta_min = np.unravel_index(np.argmin(chi_sq_arr),(nhc_tests,nbeta_tests))
    opt_params_3[0] = hc_test[idx_hc_min]
    opt_params_3[1] = beta_test[idx_beta_min]
    if idx_hc_min == 0 or idx_hc_min == nhc_tests-1:
        print("!!! WARNING: the optimal h_c was found at the limit of the tested range.")
    if idx_beta_min == 0 or idx_beta_min == nbeta_tests-1:
        print("!!! WARNING: the optimal beta was found at the limit of the tested range.")


    # Finally, find the optimal h_c for which the weighted/mean chi_sq is min
    if verbose:
        print("Optimal chi_square values: ", opt_chi_sq_3)
        print( "Optimal h_c values: ", opt_params_3[:,0])
        print("Optimal beta values: ", opt_params_3[:,1])
                
    if verbose:
        print("FINAL ELAPSED TIME:")    
        timing(start_time)
        
    if debug:
        pdb.set_trace()

    return opt_chi_sq_3, opt_params_3
  

def spirals_muto_rc_thc_hc_linsearch(thc_test, rc_test, hc_test, spi_trace, errors, alpha, 
                                     beta, verbose=False, nproc=2, **kwargs):
    """
    Determines the best fit parameters of several individual spirals with a linear search applied to
    Muto's 3 free parameters (r_c,th_c,h_c) equation.
    
    Parameters
    ----------
    
    thc_test, rc_test, hc_test: 1D numpy.arrays
        Range of the "h_c" spiral parameter for the linear search (same for all spirals)
    spi_trace: 2D numpy.array
        Spiral trace in polar coordinates, e.g. in the following format:
        np.array([theta_spi,r_spi])
    errors: 1D numpy.array or list
        Error on theta for each point of the spiral(s), e.g.:
        np.array([err_pt1_spi1,err_pt2_spi1, ..., err_ptN_spi1])
    alpha: float
        Muto's equation alpha parameter
    th_c: list
        Muto's equation th_c parameter (should be given as a list or 1d array as 
        the basis is different for each spiral)
    verbose : boolean, optional
        If True, informations are displayed in the shell.
    nproc: int
        Number of processors to be used. Default is to use multiprocessing.
        0/None -> nproc/2 (computed by cpu_count() function)
        1  -> single processing
        >1 -> multiprocessing
    same_weight: bool, opt
        Whether the mean chi_square calculated over the different spirals should 
        not be weighted (same_weight=True) or weighted by the number of points 
        considered in each spiral trace (same_weight=False).
        
    Returns
    -------
    out : np.array
        Parameters of the best fit to Muto's equation, for each spiral.
        
    """
    
    # Check the input parameters are in the right format
    
    start_time = time_ini()    
    
    if type(spi_trace) is np.ndarray and (type(errors) is list or type(errors) is np.ndarray):
        npts = spi_trace.shape[1]
        if len(errors) != npts:
            raise ValueError('The length of the errors list should be the same: {:} .'.format(npts))
        elif spi_trace.shape[0] != 2:
            raise ValueError('spiral trace should contain theta and r values in 2 columns')
        else:
            nthc_tests = thc_test.shape[0]
            nrc_tests = rc_test.shape[0]
            nhc_tests = hc_test.shape[0]
    else:
       raise ValueError('spi_trace or errors do not have the right format.')
        
    if verbose:
        print('')
        print('Linear search is running...')
    
    clockwise = kwargs.get('clockwise',-1)

    
    if not nproc:
        nproc = int(cpu_count()/2.) 
    
    if nproc == 1:
        chi_sq_arr = np.zeros([nthc_tests,nrc_tests,nhc_tests])
        if verbose:
            timing(start_time)
        for tt in range(nthc_tests):
            th_c = thc_test[tt]
            if verbose:
                print("----- Testing th_c = {:} (case {:}/{:}) -----".format(th_c, tt+1, nthc_tests))
                timing(start_time)
            for rr in range(nrc_tests):
                r_c = rc_test[rr]
                for hh in range(nhc_tests):
                    h_c = hc_test[hh]
                    p = np.array([th_c,r_c,h_c])
                    chi_sq_arr[tt,rr,hh] = chisquare(p, spi_trace, errors, 'muto12_3params', True, clockwise, alpha, beta)
                    if chi_sq_arr[tt,rr,hh] is np.nan:
                        pdb.set_trace()
    elif nproc > 1:     
        # create list of input test params:
        if verbose:
            print(" *** Creating list of test input parameters for spiral... ***")
        p = []
        for tt in range(nthc_tests):
            for rr in range(nrc_tests):
                for hh in range(nhc_tests):       
                    p.append(np.array([thc_test[tt],rc_test[rr],hc_test[hh]]))
        # Do multiprocessing evaluation
        if verbose:
            print("*** Evaluating all test parameters for spiral ... ***")
        #pool = Pool(processes=int(nproc))  
        res = pool_map(nproc, chisquare, iterable(p), spi_trace, errors, 
                       'muto12_3params', True, clockwise, alpha, beta)
        chi_sq_arr = np.array(res).reshape((nthc_tests,nrc_tests,nhc_tests))
        #pool.close()
    else:
        raise ValueError('nproc should be a positive integer')     

    # GET THE OPTIMAL CHI SQUARE FOR EACH parameter
    opt_chi_sq_3 = np.nanmin(chi_sq_arr)
    idx_thc_min,idx_rc_min,idx_hc_min = np.unravel_index(np.nanargmin(chi_sq_arr),(nthc_tests,nrc_tests,nhc_tests))
    opt_params_3 = np.array([thc_test[idx_thc_min],rc_test[idx_rc_min],hc_test[idx_hc_min]])
    if idx_thc_min == 0 or idx_thc_min == nthc_tests-1:
        print("!!! WARNING: the optimal th_c was found at the limit of the tested range.")
    if idx_rc_min == 0 or idx_rc_min == nrc_tests-1:
        print("!!! WARNING: the optimal r_c was found at the limit of the tested range.")
    if idx_hc_min == 0 or idx_hc_min == nhc_tests-1:
        print("!!! WARNING: the optimal h_c was found at the limit of the tested range.")

    # Finally, find the optimal h_c for which the weighted/mean chi_sq is min
    if verbose:
        print("Optimal chi_square value: ", opt_chi_sq_3)
        print("Optimal th_c value: ", opt_params_3[0])
        print("Optimal r_c value: ", opt_params_3[1])
        print("Optimal h_c value: ", opt_params_3[2])
               
        print("FINAL ELAPSED TIME:")  
        timing(start_time)

    return opt_chi_sq_3, opt_params_3

  

def spirals_muto_rc_thc_hc_beta_linsearch(thc_test,rc_test,hc_test, beta_test,spi_trace, errors, alpha, 
                                          verbose=False, nproc=2, **kwargs):
    """
    Determines the best fit parameters of several individual spirals with a linear search applied to
    Muto's 4 free parameters (r_c,th_c,h_c,beta) equation.
    
    Parameters
    ----------
    
    hc_test: 1D numpy.array
        Range of the "h_c" spiral parameter for the linear search (same for all spirals)
    spi_trace: 2D numpy.array
        Spiral trace in polar coordinates, e.g. in the following format:
        np.array([theta_spi,r_spi])
    errors: 1D numpy.array or list
        Error on theta for each point of the spiral(s), e.g.:
        np.array([err_pt1_spi1,err_pt2_spi1, ..., err_ptN_spi1])
    alpha: float
        Muto's equation alpha parameter
    th_c: list
        Muto's equation th_c parameter (should be given as a list or 1d array as 
        the basis is different for each spiral)
    verbose : boolean, optional
        If True, informations are displayed in the shell.
    nproc: int
        Number of processors to be used. Default is to use multiprocessing.
        0/None -> nproc/2 (computed by cpu_count() function)
        1  -> single processing
        >1 -> multiprocessing
    same_weight: bool, opt
        Whether the mean chi_square calculated over the different spirals should 
        not be weighted (same_weight=True) or weighted by the number of points 
        considered in each spiral trace (same_weight=False).
        
    Returns
    -------
    out : np.array
        Parameters of the best fit to Muto's equation, for each spiral.
        
    """
    
    # Check the input parameters are in the right format
    
    start_time = time_ini()    
    
    if type(spi_trace) is np.ndarray and (type(errors) is list or type(errors) is np.ndarray):
        npts = spi_trace.shape[1]
        if len(errors) != npts:
            raise ValueError('The length of the errors list should be the same: {:} .'.format(npts))
        elif spi_trace.shape[0] != 2:
            raise ValueError('spiral trace should contain theta and r values in 2 columns')
        else:
            nthc_tests = thc_test.shape[0]
            nrc_tests = rc_test.shape[0]
            nhc_tests = hc_test.shape[0]
            nbeta_tests = beta_test.shape[0]
    else:
       raise ValueError('spi_trace or errors do not have the right format.')
        
    if verbose:
        print('')
        print('Linear search is running...')
    
    clockwise = kwargs.get('clockwise',-1)

    
    if not nproc:
        nproc = int(cpu_count()/2.) 
    
    if nproc == 1:
        chi_sq_arr = np.zeros([nthc_tests,nrc_tests,nhc_tests,nbeta_tests])
        if verbose:
            timing(start_time)
        for tt in range(nthc_tests):
            th_c = thc_test[tt]
            if verbose:
                print("----- Testing th_c = {:} (case {:}/{:}) -----".format(th_c, tt+1, nthc_tests))
                timing(start_time)
            for rr in range(nrc_tests):
                r_c = rc_test[rr]
                for hh in range(nhc_tests):
                    h_c = hc_test[hh]
                    for bb in range(nbeta_tests):
                        beta = beta_test[bb]
                        p = np.array([th_c,r_c,h_c,beta])
                        chi_sq_arr[tt,rr,hh,bb] = chisquare(p, spi_trace, errors, 'muto12_4params', True, clockwise, alpha)
                        if chi_sq_arr[tt,rr,hh,bb] is np.nan:
                            pdb.set_trace()
    elif nproc > 1:
        # create list of input test params:
        if verbose:
            print(" *** Creating list of test input parameters for spiral... ***")
        p = []
        for tt in range(nthc_tests):
            for rr in range(nrc_tests):
                for hh in range(nhc_tests):
                    for bb in range(nbeta_tests):        
                        p.append(np.array([thc_test[tt],rc_test[rr],hc_test[hh],beta_test[bb]]))
        # Do multiprocessing evaluation
        if verbose:
            print("*** Evaluating all test parameters for spiral ... ***")
        #pool = Pool(processes=int(nproc))  
        res = pool_map(nproc, chisquare, iterable(p), spi_trace, errors, 
                       'muto12_4params', True, clockwise, alpha)
        chi_sq_arr = np.array(res).reshape((nthc_tests,nrc_tests,nhc_tests,nbeta_tests))
        #pool.close()
    else:
        raise ValueError('nproc should be a positive integer')     

    # GET THE OPTIMAL CHI SQUARE FOR EACH parameter
    opt_chi_sq_3 = np.nanmin(chi_sq_arr)
    idx_thc_min,idx_rc_min,idx_hc_min,idx_beta_min = np.unravel_index(np.nanargmin(chi_sq_arr),(nthc_tests,nrc_tests,nhc_tests,nbeta_tests))
    opt_params_3 = np.array([thc_test[idx_thc_min],rc_test[idx_rc_min],hc_test[idx_hc_min],beta_test[idx_beta_min]])
    if idx_thc_min == 0 or idx_thc_min == nthc_tests-1:
        print("!!! WARNING: the optimal th_c was found at the limit of the tested range.")
    if idx_rc_min == 0 or idx_rc_min == nrc_tests-1:
        print("!!! WARNING: the optimal r_c was found at the limit of the tested range.")
    if idx_hc_min == 0 or idx_hc_min == nhc_tests-1:
        print("!!! WARNING: the optimal h_c was found at the limit of the tested range.")
    if idx_beta_min == 0 or idx_beta_min == nbeta_tests-1:
        print("!!! WARNING: the optimal beta was found at the limit of the tested range.")

    # Finally, find the optimal h_c for which the weighted/mean chi_sq is min
    if verbose:
        print("Optimal chi_square value: ", opt_chi_sq_3)
        print("Optimal th_c value: ", opt_params_3[0])
        print("Optimal r_c value: ", opt_params_3[1])
        print("Optimal h_c value: ", opt_params_3[2])
        print("Optimal beta value: ", opt_params_3[3])
               
        print("FINAL ELAPSED TIME:")  
        timing(start_time)

    return opt_chi_sq_3, opt_params_3
  
    

def find_params_unc(spi_coords, errors, best_params, fit_eq, clockwise, fwhm, 
                    res_file, step_unc=0.01, verbose=False, outpath='',
                    bootstrap=False, n_boots=10000, n_proc=2, method='nelder-mead', 
                    options=None, bounds=None, **kwargs):
    """
    Calculate the uncertainties on the best fit parameters to a spiral eq.
    For that, the method samples independent mesurements within the trace.
    It draws as many sets of independent measurements and estimate the uncertainty
    on each set of indepndent measurements.
    The final uncertainty is then the mean of these uncertainties.
    
    Example: 
        If for measurements to be considered independent (separation > fwhm),
        they need to be separated by 10deg, but the original measuremnts are
        made every 1deg. There will be 10 sets of independent measurements,
        starting at 0, 1,...,9 deg respectively and separated by 10deg.
        
    Parameters
    ----------
    spi_coords: 2d numpy array
        Spiral trace in polar coordinates: np.array([theta_spi,r_spi]).
    errors: 1d numpy array, opt
        Corresponding errors on r_spi. Required to compute pitch uncertainty.
        If not provided, assumes same uncertainty on each point.
    clockwise: bool
        Whether the spiral is clockwise or counter-clockwise
    fwhm, float
        FWHM in the obs. The algo samples the provided trace so that each point
        considered is independent from its neighbour (separated by fwhm).
    step_unc: flt, opt
        Step (in terms of fraction of best fit parameter value) used to search 
        for the uncertainty.
    p_ini : numpy array of 2 elements, opt
        Initial estimate for the logarithmic spiral parameters, in the format:
        np.array([A,B]) for r = A*exp(B*theta). Note: usually simplex finds its 
        way whichever the initial estimate.
    res_file: str, opt
        Full path and name of a text file to write results.
    verbose: bool. opt
        Whether to print lower and upper uncertainty on each spiral section.
        
    Returns
    -------
    phi: float
        The pitch angle.
    unc_phi: float
        (returned if unc set to True) Uncertainty on pitch angle.
    
    """

    alpha = kwargs.get('alpha',None)   
    beta = kwargs.get('beta',None)  
    #h_c = kwargs.get('h_c',None)
    r_c = kwargs.get('r_c',None)   
    th_c = kwargs.get('th_c',None) 

    if fit_eq == 'log':
        best_pitch = np.rad2deg(math.atan(best_params[1]))


    if 'muto12' in fit_eq or bootstrap:
        # do bootstrapping
        uncertainties = bootstrap_unc(spiral_minimize, best_params, spi_coords, 
                                      errors, fit_eq, res_file, n_boots=n_boots, 
                                      n_proc=n_proc, plot=True, verbose=verbose, 
                                      outpath=outpath, clockwise=clockwise, 
                                      method=method, options=options, 
                                      bounds=bounds, **kwargs) # kwargs: e.g. n_proc
        if res_file is not None: 
            with open(res_file, 'a+') as f:
                f.write("Uncertainties determined by bootstrapping ({:.0f} bootstraps)...\n".format(n_boots))

        
    else:
        nparams = len(best_params)
        npts = spi_coords.shape[1]
        nloop = int(npts/3) # loop an arbitrarily large number of times (we'll break if needed)
        low_unc = np.zeros([nloop,nparams])
        upp_unc = np.zeros([nloop,nparams])
    
        uncertainties = np.zeros([2,nparams])
    
        for jj in range(nloop): 
    
            # break the loop if we get to a previously included point
            if jj > 1:
                if jj == stop_it:
                    break
            
            # first resample the trace to only have independent measurements
            spi_1_coords_PA = [spi_coords[0,jj]]
            spi_1_coords_r= [spi_coords[1,jj]]
            errors_1 = [errors[jj]]
            
            counter=0
            for nn in range(jj,npts):
                x_prev = -spi_1_coords_r[counter]*np.sin(spi_1_coords_PA[counter])
                y_prev = spi_1_coords_r[counter]*np.cos(spi_1_coords_PA[counter])
                x_new = -spi_coords[1,nn]*np.sin(spi_coords[0,nn])
                y_new = spi_coords[1,nn]*np.cos(spi_coords[0,nn])        
                dist = vip_hci.var.dist(y_prev,x_prev,y_new,x_new)
                if dist > fwhm:
                    if jj == 0 and len(spi_1_coords_PA)==1: # ie. first time condition is fulfilled in the first iteration
                        stop_it = nn
                    spi_1_coords_r.append(spi_coords[1,nn])
                    spi_1_coords_PA.append(spi_coords[0,nn])
                    errors_1.append(errors[nn])
                    counter+=1
            spi_1_coords_fit = np.array([spi_1_coords_PA,spi_1_coords_r])
            errors_1 = np.array(errors_1)
            
            npts_indep = len(spi_1_coords_PA)
    
            chisq = chisquare(best_params, spi_1_coords_fit, errors_1, fit_eq=fit_eq, 
                              reduced=False, clockwise=clockwise, alpha=alpha, 
                              beta=beta, r_c=r_c, th_c=th_c)
    
            for bp in range(nparams):
                test_params = best_params.copy()
                #frac_tmp = 1.-step_uncertainty
                test_params[bp] = best_params[bp]-step_unc*best_params[bp]
                # lower uncertainty?
                counter = 0
                while True:
                    tmp_chisq = chisquare(test_params, spi_1_coords_fit, errors_1, 
                                          fit_eq=fit_eq, reduced=False, 
                                          clockwise=clockwise, alpha=alpha, 
                                          beta=beta, r_c=r_c, th_c=th_c)
                    counter+=1
                    if abs(tmp_chisq-chisq) > 1:
                        break
                    test_params[bp] = test_params[bp]-step_unc*best_params[bp]
                low_unc[jj,bp] = test_params[bp]-best_params[bp]
    
                # upper uncertainty?
                test_params = best_params.copy()
                #frac_tmp = 1.+step_uncertainty
                test_params[bp] = best_params[bp]+step_unc*best_params[bp]
                counter = 0
                while True:
                    tmp_chisq = chisquare(test_params, spi_1_coords_fit, errors_1, 
                                          fit_eq=fit_eq, reduced=False, clockwise=clockwise, 
                                          alpha=alpha, beta=beta, r_c=r_c, th_c=th_c)
                    counter+=1
                    if abs(tmp_chisq-chisq) > 1:
                        break
                    test_params[bp] = test_params[bp]+step_unc*best_params[bp]
                upp_unc[jj,bp] = test_params[bp]-best_params[bp]
        
        if verbose:
            print("For uncertainty estimates, we sampled {:.0f} independent points from the original trace ({:.0f} points)".format(npts_indep,npts))
        if res_file is not None:
            with open(res_file, 'a+') as f:
                f.write("For uncertainty estimates, we sampled {:.0f} independent points from the original trace ({:.0f} points)\n".format(npts_indep,npts))
        
        for bp in range(nparams):
            uncertainties[0,bp] = np.mean(low_unc[:stop_it,bp])
            uncertainties[1,bp] = np.mean(upp_unc[:stop_it,bp])
        
        
    if fit_eq == 'log':
        pitch_unc = np.rad2deg(np.arctan(np.abs(uncertainties[:,1])))
        print("*** FINAL PITCH ANGLE ESTIMATE (on the whole spiral): {:.1f}-{:.1f}+{:.1f} deg ***".format(best_pitch,pitch_unc[0],pitch_unc[1]))
        with open(res_file, 'a+') as f:
            f.write("*** FINAL PITCH ANGLE ESTIMATE (on the whole spiral): {:.1f}-{:.1f}+{:.1f} deg ***".format(best_pitch,pitch_unc[0],pitch_unc[1]))
                
    return uncertainties



def measure_pitch_angle(spi_coords, errors, clockwise, fwhm, step_unc=0.01,
                        p_ini=np.array([50,0.15]), unc=True, verbose=False):
    """
    Calculate the pitch angle of a spiral trace by fitting it to a logarithmic 
    spiral.
        
    Parameters
    ----------
    spi_coords: 2d numpy array
        Spiral trace in polar coordinates: np.array([theta_spi,r_spi]).
    errors: 1d numpy array, opt
        Corresponding errors on r_spi. Required to compute pitch uncertainty.
        If not provided, assumes same uncertainty on each point.
    clockwise: bool
        Whether the spiral is clockwise or counter-clockwise
    fwhm, float
        FWHM in the obs. The algo samples the provided trace so that each point
        considered is independent from its neighbour (separated by fwhm).
    step_unc: flt, opt
        Step (in terms of fraction of best fit parameter value) used to search 
        for the uncertainty.
    p_ini : numpy array of 2 elements, opt
        Initial estimate for the logarithmic spiral parameters, in the format:
        np.array([A,B]) for r = A*exp(B*theta). Note: usually simplex finds its 
        way whichever the initial estimate.
    unc: bool. opt
        Whether to return the uncertainty on pitch angle too
    verbose: bool. opt
        Whether to print lower and upper uncertainty on each spiral section.
        
    Returns
    -------
    phi: float
        The pitch angle.
    unc_phi: float
        (returned if unc set to True) Uncertainty on pitch angle.
    
    """
    
    best_params = spiral_minimize(p_ini, spi_coords, errors, fit_eq='log', 
                                 verbose=False, clockwise=clockwise)
    b = best_params[1]
    phi = np.rad2deg(math.atan(b))


    if unc:
        # we have to be careful to sample independent mesurements here in order
        # to compute meaningful uncertainties
        
        # furthermore, we want to draw all possible independent samples within 
        # the spiral trace (e.g. if 10deg separation leads to independent 
        # measurements but original measurements separated by 1deg => loop 10x)
    
        npts = spi_coords.shape[1]
        nloop = int(npts/3) # loop an arbitrarily large number of times (we'll break if needed)
        low_unc = np.zeros(nloop)
        upp_unc = np.zeros(nloop)
    
        for jj in range(nloop): 
    
            # break the loop if we get to a previously included point
            if jj > 1:
                if jj == stop_it:
                    break
            
            # first resample the trace to only have independent measurements
            spi_1_coords_PA = [spi_coords[0,jj]]
            spi_1_coords_r= [spi_coords[1,jj]]
            errors_1 = [errors[jj]]
            
            counter=0
            for nn in range(jj,npts):
                x_prev = -spi_1_coords_r[counter]*np.sin(spi_1_coords_PA[counter])
                y_prev = spi_1_coords_r[counter]*np.cos(spi_1_coords_PA[counter])
                x_new = -spi_coords[1,nn]*np.sin(spi_coords[0,nn])
                y_new = spi_coords[1,nn]*np.cos(spi_coords[0,nn])        
                dist = vip_hci.var.dist(y_prev,x_prev,y_new,x_new)
                if dist > fwhm:
                    if jj == 0 and len(spi_1_coords_PA)==1: # ie. first time condition is fulfilled in the first iteration
                        stop_it = nn
                    spi_1_coords_r.append(spi_coords[1,nn])
                    spi_1_coords_PA.append(spi_coords[0,nn])
                    errors_1.append(errors[nn])
                    counter+=1
            spi_1_coords = np.array([spi_1_coords_PA,spi_1_coords_r])
            errors_1 = np.array(errors_1)
                    
            npts_indep = len(spi_1_coords_PA)
    
        #if unc:
            # first compute chi_sq of best fit
            chisq = chisquare(best_params, spi_1_coords, errors_1, fit_eq='log', 
                              reduced=False, clockwise=clockwise)#, alpha=alpha, beta=beta, r_c=r_c, th_c=th_c)
    
            # mean r and PA
            mean_PA = np.mean(np.rad2deg(spi_1_coords[0])) # get it in deg
            mean_r = np.mean(spi_1_coords[1]) # px
    
            # find lower uncertainty
            test_params = best_params.copy()
            test_params[1] = best_params[1]-step_unc*best_params[1]
            counter = 0
            while True:
                tmp_chisq = chisquare(test_params, spi_1_coords, errors_1, fit_eq='log', 
                                      reduced=False, clockwise=clockwise)#, alpha=alpha, beta=beta, r_c=r_c, th_c=th_c)
                counter+=1
                if abs(tmp_chisq-chisq) > 1:
                    break
                test_params[1] = test_params[1]-step_unc*best_params[1]
    
            test_pitch = np.rad2deg(math.atan(test_params[1]))
            low_unc[jj] = test_pitch-phi
    
            # find upper uncertainty
            test_params = best_params.copy()
            test_params[1] = best_params[1]+step_unc*best_params[1]
            counter = 0
            while True:
                tmp_chisq = chisquare(test_params, spi_1_coords, errors_1, fit_eq='log', 
                                      reduced=False, clockwise=clockwise)#, alpha=alpha, beta=beta, r_c=r_c, th_c=th_c)
                counter+=1
                if abs(tmp_chisq-chisq) > 1:
                    break
                test_params[1] = test_params[1]+step_unc*best_params[1]
    
            test_pitch = np.rad2deg(math.atan(test_params[1]))
            upp_unc[jj] = test_pitch-phi
      
    low_unc = np.mean(low_unc[:stop_it])
    upp_unc = np.mean(upp_unc[:stop_it])
      
    if verbose:
        print("We sampled {:.0f} independent points from the original trace ({:.0f} points)".format(npts_indep,npts))
        print("Pitch angle measured near (PA={:.1f}deg,r={:.1f}px): {:.2f}-{:.2f}+{:.2f} deg".format(mean_PA,mean_r,phi,abs(low_unc),abs(upp_unc)))
    phi_unc = (abs(low_unc)+abs(upp_unc))/2
        
    return phi, phi_unc


def chisquare(modelParameters, spi_trace, errors, fit_eq='gen_archi', 
              reduced=False, clockwise=-1,alpha=None, beta=None, h_c=None, 
              r_c=None, th_c=None, fac_hr=100):
    """
    Calculate the reduced chi2 of a spiral model.
        
    Parameters
    ----------
    modelParameters : np.array
        Estimate of the spiral parameters, e.g. in the following format for general Archimede spirals:
        np.array([A_spi_1,n_spi_1, r0_spi_1])
    spi_trace: numpy.array
        Spiral trace in polar coordinates, e.g. in the following format for general Archimede spirals:
        np.array([theta_spi_1,r_spi_1])
    fit_eq: string, {'gen_archi', 'log', 'lin_archi', 'poly', 'muto12', 
                     'muto12_4params', 'muto12_3params', 'muto12_2params',
                     'muto12_2params_hc_beta', 'muto12_1param'}
        Type of equation to which the coordinate is evaluated.
        
    Returns
    -------
    out: float
    The reduced chi squared.
        
    """

    if 'muto12' in fit_eq:
        coord_0_tmp = spi_trace[1,:]
    else:
        coord_0_tmp = spi_trace[0,:]
    coord_0 = spi_trace[0,:]
    coord_1 = spi_trace[1,:]

    # Evaluate the trace of the model for each input angle or radius
    coord1_eval = spiral_eq_eval(coord_0_tmp, modelParameters, fit_eq=fit_eq, 
                                 clockwise=clockwise,alpha=alpha, beta=beta, 
                                 h_c=h_c, r_c=r_c, th_c=th_c)
                                
    # for 'muto12' fits fall back to r coord estimates
    if 'muto12' in fit_eq:
        npts = spi_trace.shape[1]
        coord_1_hr = np.linspace(spi_trace[1,0],spi_trace[1,-1],fac_hr*npts)
        coord0_eval_hr = spiral_eq_eval(coord_1_hr, modelParameters, fit_eq=fit_eq, 
                                        clockwise=clockwise,alpha=alpha, beta=beta, 
                                        h_c=h_c, r_c=r_c, th_c=th_c)
        coord1_eval = np.ones_like(coord_1)
        for ii in range(len(coord_1)):
            th_n2, idx_n2 = find_nearest(coord0_eval_hr, coord_0[ii], 
                                         output='both', constraint=None, n=2)
            if th_n2[1]-th_n2[0] == 0:
                th_tmp = th_n2.copy()
                n_ini=3
                c = 0
                while th_tmp[-1]-th_tmp[0] == 0 and c < len(coord_1)-2:
                    th_tmp, idx_tmp = find_nearest(coord0_eval_hr, coord_0[ii], 
                                                   output='both', 
                                                   constraint=None, n=n_ini+c)
                    th_n2 = np.array([th_tmp[0],th_tmp[-1]])
                    idx_n2 = np.array([idx_tmp[0],idx_tmp[-1]])
                    c+=1
                if c > len(coord_1)-3:
                    pdb.set_trace()
            p = (coord_0[ii]-th_n2[0])/(th_n2[1]-th_n2[0])                   
            coord1_eval[ii] = coord_1_hr[idx_n2[0]]+p*(coord_1_hr[idx_n2[1]]-coord_1_hr[idx_n2[0]])
                            
    # Compute the chisquare value of the considered model
    errors_tmp = errors.copy()
    if 'muto12' in fit_eq and "hc" not in fit_eq and "1param" not in fit_eq:
        # more weight on kink, less far from it
        for ii in range(len(coord_1)):
            errors_tmp[ii] = errors[ii]*max(coord_1[ii]/modelParameters[1],
                                            modelParameters[1]/coord_1[ii])
                    
    dof = spi_trace.shape[1] - modelParameters.shape[0] - 1
    chisq = np.sum(np.power(coord1_eval - coord_1,2) / np.power(errors_tmp,2))
    red_chisq = (1./dof)*chisq

    if reduced:
        return red_chisq
    else:
        return chisq



def spiral_eq_eval(coord_0, p, fit_eq='gen_archi',clockwise=-1,alpha=None,beta=None,h_c=None,r_c=None,th_c=None):
    """
    Evaluate the chosen spiral equation.
        
    Parameters
    ----------
    coord_0 : float or np.array
        theta (or r for muto12) to be evaluated in the equation
    p: numpy.array or tuple
        Set of parameters corresponding to the chosen type of spiral equation.
    fit_eq: string, {'gen_archi', 'log', 'lin_archi', 'poly', 'muto12', 
                     'muto12_4params', 'muto12_3params', 'muto12_2params',
                     'muto12_2params_hc_beta', 'muto12_1param'}
        Type of equation to which the coordinate is evaluated.
        
        
    Information on the different equations:
    --------------------------------
    - General Archimedean:
        r = a + b * theta^n
        p[0] = a
        p[1] = b
        p[2] = n
    - Log:                  https://en.wikipedia.org/wiki/Logarithmic_spiral
        p[0] = a
        p[1] = b
    - poly(nomial)
        p[0] = a_0
        p[1] = a_1
        etc.
        in r = Sum_i(a_i * theta^i)
    - Linear Archimedean:   https://en.wikipedia.org/wiki/Archimedean_spiral
        p[0] = a
        p[1] = b
    - Muto12:               http://iopscience.iop.org/article/10.1088/2041-8205/748/2/L22/pdf;jsessionid=B6BE571C45926447A1F986AD3060E2D8.c4.iopscience.cld.iop.org  (page 3)
        p[0] = theta_0
        p[1] = r_c
        p[2] = h_c
        p[3] = alpha
        p[4] = beta
    
    Returns
    -------
    out: float
    The coord_1 evaluation.
        
    """

    if fit_eq == 'gen_archi':
        return p[0]+p[1]*np.power(coord_0,p[2])
    
    elif fit_eq == 'log':
        return p[0]*np.exp(p[1]*coord_0)

    elif fit_eq == 'lin_archi':
        return p[0] + p[1]*coord_0

    elif fit_eq == 'poly':
        return polyval(coord_0,p)

    elif fit_eq == 'muto12':
        return p[0] + (np.sign(coord_0-p[1])/p[2]) * (((coord_0/p[1])**(1+p[4]) * ((1/(1+p[4]))-((1/(1-p[3]+p[4]))*(coord_0/p[1])**(-p[3])))) - ((1/(1+p[4]))-(1/(1-p[3]+p[4]))))
    
    elif fit_eq == 'muto12_4params':
        if alpha is None:
            raise ValueError('Please define parameter alpha.')
        return p[0] + (np.sign(coord_0-p[1])/p[2]) * (((coord_0/p[1])**(1+p[3]) * ((1/(1+p[3]))-((1/(1-alpha+p[3]))*(coord_0/p[1])**(-alpha)))) - ((1/(1+p[3]))-(1/(1-alpha+p[3]))))
    
    elif fit_eq == 'muto12_3params':
        if alpha is None:
            raise ValueError('Please define parameter alpha.')
        if beta is None:
            raise ValueError('Please define parameter beta.')
        return p[0] + (np.sign(coord_0-p[1])/p[2]) * (((coord_0/p[1])**(1+beta) * ((1/(1+beta))-((1/(1-alpha+beta))*(coord_0/p[1])**(-alpha)))) - ((1/(1+beta))-(1/(1-alpha+beta))))
    elif fit_eq == 'muto12_2params':
        if alpha is None:
            raise ValueError('Please define parameter alpha.')
        if beta is None:
            raise ValueError('Please define parameter beta.')
        if h_c is None:
            raise ValueError('Please define parameter h_c.')
        return p[0] + (np.sign(coord_0-p[1])/h_c) * (((coord_0/p[1])**(1+beta) * ((1/(1+beta))-((1/(1-alpha+beta))*(coord_0/p[1])**(-alpha)))) - ((1/(1+beta))-(1/(1-alpha+beta))))
    elif fit_eq == 'muto12_2params_hc_beta':
        if alpha is None:
            raise ValueError('Please define parameter alpha.')
        if r_c is None:
            raise ValueError('Please define parameter r_c.')
        if th_c is None:
            raise ValueError('Please define parameter th_c.')
        return th_c + (np.sign(coord_0-r_c)/p[0]) * (((coord_0/r_c)**(1+p[1]) * ((1/(1+p[1]))-((1/(1-alpha+p[1]))*(coord_0/r_c)**(-alpha)))) - ((1/(1+p[1]))-(1/(1-alpha+p[1]))))
    elif fit_eq == 'muto12_1param':
        if alpha is None:
            raise ValueError('Please define parameter alpha.')
        if beta is None:
            raise ValueError('Please define parameter beta.')
        if r_c is None:
            raise ValueError('Please define parameter r_c.')
        if th_c is None:
            raise ValueError('Please define parameter th_c.')
        return th_c + (np.sign(coord_0-r_c)/p[0]) * (((coord_0/r_c)**(1+beta) * ((1/(1+beta))-((1/(1-alpha+beta))*(coord_0/r_c)**(-alpha)))) - ((1/(1+beta))-(1/(1-alpha+beta))))

    else: raise ValueError('Pick a valid equation type')

    
def spiral_eq_definition(fit_eq,alpha=None,beta=None,h_c=None,r_c=None,th_c=None):
    """
    Provides expression of spiral equation used for the fit.
    
    Parameters
    ----------    
    fit_eq: string, {'gen_archi', 'log', 'lin_archi', 'muto12', 'muto12_3params','poly'}
        Type of equation to which the coordinate is evaluated.
    
    Returns
    -------
    out: str
        string with the expression of the equation used for the spiral fit.
        
    """
    
    if fit_eq == 'gen_archi':
        return "p[0]+p[1]*theta^p[2]"
    
    elif fit_eq == 'log':
        return "p[0]*exp(p[1]*theta)"

    elif fit_eq == 'lin_archi':
        return "p[0] + p[1]*theta"

    elif fit_eq == 'poly':
        return "Sum_i(p[i] * theta^i)"

    elif fit_eq == 'muto12':
        return "p[0] -clockwise* (np.sign(r-p[1])/p[2]) * (((r/p[1])**(1+p[4]) * ((1/(1+p[4]))-((1/(1-p[3]+p[4]))*(r/p[1])**(-p[3])))) - ((1/(1+p[4]))-(1/(1-p[3]+p[4]))))"
    
    elif fit_eq == 'muto12_4params':
        if alpha is None:
            raise ValueError('Please define parameter alpha.')
        return "p[0] -clockwise* (np.sign(r-p[1])/p[2]) * (((r/p[1])**(1+p[3]) * ((1/(1+p[3]))-((1/(1-alpha+p[3]))*(r/p[1])**(-alpha)))) - ((1/(1+p[3]))-(1/(1-alpha+p[3]))))"
    
    elif fit_eq == 'muto12_3params':
        if alpha is None:
            raise ValueError('Please define parameter alpha.')
        if beta is None:
            raise ValueError('Please define parameter beta.')
        return "p[0] -clockwise* (np.sign(r-p[1])/p[2]) * (((r/p[1])**(1+beta) * ((1/(1+beta))-((1/(1-alpha+beta))*(r/p[1])**(-alpha)))) - ((1/(1+beta))-(1/(1-alpha+beta))))"
        
    elif fit_eq == 'muto12_2params':
        if alpha is None:
            raise ValueError('Please define parameter alpha.')
        if beta is None:
            raise ValueError('Please define parameter beta.')
        if h_c is None:
            raise ValueError('Please define parameter h_c.')
        return "p[0] -clockwise* (np.sign(r-p[1])/h_c) * (((r/p[1])**(1+beta) * ((1/(1+beta))-((1/(1-alpha+beta))*(r/p[1])**(-alpha)))) - ((1/(1+beta))-(1/(1-alpha+beta))))"
        
    elif fit_eq == 'muto12_2params_hc_beta':
        if alpha is None:
            raise ValueError('Please define parameter alpha.')
        if r_c is None:
            raise ValueError('Please define parameter r_c.')
        if th_c is None:
            raise ValueError('Please define parameter th_c.')
        return "th_c -clockwise* (np.sign(r-r_c)/p[0]) * (((r/r_c)**(1+p[1]) * ((1/(1+p[1]))-((1/(1-alpha+p[1]))*(r/r_c)**(-alpha)))) - ((1/(1+p[1]))-(1/(1-alpha+p[1]))))"
        
    elif fit_eq == 'muto12_1param':
        if alpha is None:
            raise ValueError('Please define parameter alpha.')
        if beta is None:
            raise ValueError('Please define parameter beta.')
        if r_c is None:
            raise ValueError('Please define parameter r_c.')
        if th_c is None:
            raise ValueError('Please define parameter th_c.')
        return "th_c -clockwise* (np.sign(r-r_c)/p[0]) * (((r/r_c)**(1+beta) * ((1/(1+beta))-((1/(1-alpha+beta))*(r/r_c)**(-alpha)))) - ((1/(1+beta))-(1/(1-alpha+beta))))"

    else: raise ValueError('Pick a valid equation type')