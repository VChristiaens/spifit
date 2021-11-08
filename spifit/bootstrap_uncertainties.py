# -*- coding: utf-8 -*-
"""
Determine uncertainties on best fit spiral parameters for 'Muto12' types of fits,
using bootstrapping.
"""

#__all__ = ['nrefrac',
#           'do_boot_envt']

#from vip.conf import timeInit, timing, eval_func_tuple
#from vip.fits import open_fits, write_fits
#import vip
#plots = vip.var.pp_subplots

#from astropy.io import fits as ap_fits
#from astropy.convolution import Gaussian1DKernel, convolve_fft
#from astropy.stats import gaussian_fwhm_to_sigma
#import itertools as itt
#from matplotlib import pyplot as plt
from vip_hci.conf.utils_conf import pool_map, iterable
import numpy as np
#from operator import mul
from os.path import isfile
#import pandas as pd
#import types
#from scipy import interpolate
#from scipy.optimize import curve_fit
#from subroutines import find_nearest
from vip_hci.specfit import spec_confidence
from vip_hci.fits import write_fits, open_fits

##################################
# DEFINE CORE BOOTSTRAP FUNCTION #
##################################
            
def gauss(x, *p):
    A, mu, sigma = p
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))            
            
def eval_func_tuple(f_args):
    """ Takes a tuple of a function and args, evaluates and returns result"""
    return f_args[0](*f_args[1:])
   
   
def do_boot_spi(boot_i, min_func, p_ini, spi_coords, errors, fit_eq,
                method='Nelder-Mead', options=None, bounds=None, verbose=False, 
                **kwargs):
    """
    Function to do the model fitting for each bootstrap sample.
    
    boot_i: int
        Bootstrap index
    model_arr: numpy array
        Array containing all model values. Should be K x M x N dimensions,
        for K values probed for param 1, etc.
    range_pn: 1d numpy array or list
        Range of values for parameter n.
    data_y, data_y_err: 1d numpy arrays
        Measured data and corresponding errors.
    verbose: string
        Whether to print more details
    """
        
    data_x = spi_coords[0,:]
    data_y = spi_coords[1,:]        
    # Data
    npts = spi_coords.shape[-1]

    # Create the bootstrap
    if boot_i > 0:
        #np.random.seed(boot_i)
        boot_samp = np.random.choice(npts, npts)
        data_boot = np.array([data_x[boot_samp], data_y[boot_samp]])
        data_boot_x = data_x[boot_samp]
        data_boot_y = data_y[boot_samp]
        data_boot_err = errors[boot_samp]
        ord_ind = np.argsort(data_boot_x)
        data_boot_x = data_boot_x[ord_ind]
        data_boot_y = data_boot_y[ord_ind]
        data_boot_err = data_boot_err[ord_ind]
        data_boot = np.array([data_boot_x, data_boot_y])
    else:
        data_boot = spi_coords.copy() #data_y
        #data_boot_x = data_x
        data_boot_err = errors.copy()

    # Find minimum chisquare among your model parameters
    best_p_boots = min_func(p_ini, data_boot, data_boot_err, method=method, 
                            fit_eq=fit_eq, options=options, verbose=False, 
                            bounds=bounds, fac_hr=100, **kwargs)
    if verbose:
        print("End of bootstrap #", boot_i)
    
    # return best fit params and corresponding chi square
    return best_p_boots
    


################
## PARAMETERS ##
################

def bootstrap_unc(min_func, best_params, spi_coords, errors, fit_eq, res_file, 
                  n_boots=10000, nproc=1, plot=True, verbose=False, outpath='', 
                  method='nelder-mead', options=None, 
                  bounds=None, overwrite=False, **kwargs):     
                      
    n_params = len(best_params)
    ##############################################
    # START THE MULTIPROCESSING ON ALL BOOTSTRAPS#
    ##############################################
    #pool = Pool(processes=nproc)
    
    if not isfile(outpath+"final_bootstraps.fits") or overwrite:    
    
        best_p_boots = np.zeros([n_boots, n_params])
                    
        if nproc>1:
            #res_best_p, res_chi = pool.map(eval_func_tuple, itt.izip(itt.repeat(do_boot),
            res = pool_map(nproc, do_boot_spi, iterable(range(n_boots)), min_func, 
                           best_params, spi_coords, errors, fit_eq,
                           method, options, bounds, verbose, **kwargs)
                                                
            results =  np.array(res)
            for bb in range(n_boots):
                best_p_boots[bb] = results[bb]        
        else:
            for bb in range(n_boots):
                best_p_boots[bb] = do_boot_spi(bb, min_func, best_params, 
                                               spi_coords, errors, fit_eq, 
                                               method, options, bounds, 
                                               verbose, **kwargs)
        #pool.close()
    
        write_fits(outpath+"final_bootstraps.fits", best_p_boots)
    else:
        best_p_boots = open_fits(outpath+"final_bootstraps.fits")
    
    ###################
    # FIT to GAUSSIANS#
    ###################

#    ft_sz_ax = 14
#    lab_sz = 12
    #fig, axn = plt.subplots(n_params)
    #fig.set_size_inches(12, int(8*n_params))
    
    label_list = ['Param #{:.0f}'.format(i) for i in range(n_params)]
    
    #try:
    _, unc_dict = spec_confidence(best_p_boots, label_list, cfd=68.27, 
                                       bins=max(int(n_boots/100),50), 
                                       gaussian_fit=False, weights=None, 
                                       verbose=verbose, save=plot, 
                                       output_dir=outpath, bounds=None, 
                                       priors=None)
    uncertainties = np.zeros([2,n_params])
    for i in range(n_params):
        uncertainties[:,i] = unc_dict['Param #{:.0f}'.format(i)]
    
#    except:
#        if verbose:
#            print('WARNINGS:')
#            print("Gaussian fit of bootstrap results failed\n")
#        with open(res_file, "a+") as f:
#            f.write(' \n')
#            f.write('WARNINGS: \n')
#            f.write("Gaussian fit of bootstrap results failed\n")
                
#    for i in range(n_params):
#        
#        parami_boots = best_p_boots[:,i]
#        axi = axn[i]
#        
#        # param 1 plot
#        try:
#            d = np.diff(np.unique(parami_boots)).min()
#        except:
#            d = 1.0
#        left_of_first_bin = parami_boots.min() - float(d)/2
#        right_of_last_bin = parami_boots.max() + float(d)/2
#        axi.hist(parami_boots,np.arange(left_of_first_bin, 
#                                        right_of_last_bin + d, d))
#        axi.set_ylabel('Counts', fontsize=ft_sz_ax)
#        axi.set_xlabel('Param {:.0f}'.format(i+1), fontsize=ft_sz_ax)
#        axi.tick_params(axis='both', which='major', labelsize=lab_sz)
#            
#    if plot:
#        plt.show()
#        fig.savefig(outpath+'histogram_bootstrap_unc'.pdf)
    
        
    return uncertainties