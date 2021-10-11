#!/usr/bin/python

from pylab import *

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import math
from spiral_fit import spiral_minimize, chisquare, spiral_eq_eval, spiral_eq_definition, find_params_unc
from spiral_fit import measure_pitch_angle, spirals_muto_rc_thc_hc_beta_linsearch # should be in the same directory
import pdb
from vip_hci.fits import write_fits


def find_nearest(array, value, output='index', constraint=None):
    """
    Function to find the index, and optionally the value, of an array's closest element to a certain value.
    Possible outputs: 'index','value','both' 
    Possible constraints: 'ceil', 'floor', None ("ceil" will return the closest element with a value greater than 'value', "floor" the opposite)
    """
    if type(array) is np.ndarray:
        pass
    elif type(array) is list:
        array = np.array(array)
    else:
        raise ValueError("Input type for array should be np.ndarray or list.")
        
    idx = (np.abs(array-value)).argmin()
    if type == 'ceil' and array[idx]-value < 0:
        idx+=1
    elif type == 'floor' and value-array[idx] < 0:
        idx-=1

    if output=='index': return idx
    elif output=='value': return array[idx]
    else: return array[idx], idx
    
    
def gauss(x, *p):
    A, mu, sigma = p
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))


# TBD: make equivalent function for fitting several spirals simultaneously?
# Would call: spirals_muto_simult_linsearch or spirals_muto_simult_linsearch_gamma in spiral_fit.py
def fit_one_spiral_arm(I_polar_arr, polar_coords, fwhm, clockwise, tot_range_spi, 
                       th_spi, spiral_in_params, spiral_out_params=None, 
                       sep_ang=None, bad_angs=None, ori_img=None, 
                       rmin_trace=None, rmax_trace=None, thresh=None, 
                       bin_fact=1, r_square=None, gauss_fit=True, 
                       iterate_eq_params=False, fit_eq='gen_archi', 
                       param_estimate=np.array([[90, 30.,-0.7]]), npt_model=None, 
                       weight_type='uniform', symmetric_plot=False, log_plot=False,
                       plot_fig={1,2,3,4,5}, ang_offset=90, cbar_label='Density', 
                       y_ax_lab='d', x_ax_lab='d', pix_to_dist_factor=1., 
                       label='', search_mode = 'minimize', solver='Nelder-Mead', 
                       solver_options={'xtol':1e-1, 'maxiter':4000, 'maxfev':8000},
                       bounds=None, dist=1, outpath = '', find_uncertainty=False, 
                       step_uncertainty=0.01, label_fig='a)', font_sz=12,
                       scale_as=0.1,scale_au=10,color_trace='g+',color_trace2='c+',
                       ratio_trace=None, ratio_fit=[0,1], color_fit='b-', vmin=None, vmax=None,
                       frac_rr_muto=[1,1], delta_theta_fit=[0,0], ms=10,deproj=True,
                       n_subspi=1, txt_file='results.txt', fac_hr=100, n_boots=10000, 
                       n_proc=1, **kwargs):
    
    """
    Geometrical fit of one spiral in a given fits image.
    
    Parameters
    **********
    
    I_polar_arr: 2D numpy array
        2D DEPROJECTED AND POLAR array (r, theta) containing the mean intensity 
        in each polar bin.
    polar_coords: 2D numpy array
        Contains the (r,theta) value of each intensity bin; theta in deg.
    fwhm: float
        fwhm of the observation (used along radial direction).
    clockwise: bool
        Whether the spiral is clockwise or counter-clockwise
    tot_range_spi: float
        Total angle range subtended by the spiral (max. 2pi for this version of
        the code)
    th_spi: float
        Rough trigonometric angle of the root of the spiral
    spiral_in_params: tuple of floats
        Parameters A and B (A, B) of a model spiral: A*exp(B*sign*(spi_angs-th_spi)). 
        The closest local maxima from that model spiral will be selected as spiral trace.
    spiral_out_params: tuple of floats, opt
        Parameters A and B (A, B) of a model spiral: A*exp(B*sign*(spi_angs-th_spi)). 
        The closest local maxima from that model spiral will be selected as spiral trace.
        If None, only spiral_in_params is used.
    sep_ang: float, opt
        If spiral_out_params is provided, this trigonometric angle indicates the 
        separation between considering the closest local maximum from the inner 
        or the outer model spiral.
    bad_angs: tuple or list of 2 elements, opt
        Contains the first and last trigonometric angle of a range of angles that 
        should not be considered for the spiral tracing.
    ori_img: 2D numpy array
        Contains the original fits file of the image
    rmin_trace: float, opt
        Rough radius of the root of the spiral (ideally a bit less). The algo starts 
        to report local radial max starting at this radius. If None, starts at a radius of 1.
    rmax_trace: float, opt
        Max radius of the considered spiral. If None, it is set automatically to 
        max size of the frame.
    thresh: float, opt
        Threshold value of intensity below which it cannot be considered as spiral trace.
    bin_fact: float, opt
        Binning factor for the grid, can improve spiral tracing.
    r_square: {'multiply',None,'divide'}, opt
        Whether the image should be scaled by r^2, to improve spirals further away.
    gauss_fit: bool, opt
        Whether to find the spiral trace and the uncertainty on the radius 
        with a 1D gaussian fit to the radial intensity profile in each azimuthal 
        bin. If False, the nearest integer, corresponding to max intensity, 
        will be used.
    iterate_eq_params: bool, opt
        Whether to not do the spiral fitting after the spiral tracing, in order
        e.g. to continue iterating parameters for the spiral tracing.
    fit_eq: string, {'gen_archi', 'log', 'lin_archi', 'poly', 'muto12', 
                     'muto12_4params', 'muto12_3params', 'muto12_2params',
                     'muto12_2params_hc_beta', 'muto12_1param'}
        Which equation to use for the spiral fit. Note, several variants of
        muto12 are accepted with the format 'muto12_*params' where * is:
            '1'         (hc);
            '2'         (r_c and theta_c); 
            '2_hc_beta' (h_c and beta); 
            '3'         (h_c, r_c and theta_c);
            '4'         (h_c, r_c, theta_c and beta)).
    param_estimate: 1d array
        First estimate of the best fit parameters to the equation:
        - General Archimedean:  http://mathworld.wolfram.com/ArchimedeanSpiral.html
            p[0] = a
            p[1] = n
        - Log:                  https://en.wikipedia.org/wiki/Logarithmic_spiral
            p[0] = a
            p[1] = b
        - Linear Archimedean:   https://en.wikipedia.org/wiki/Archimedean_spiral
            p[0] = a
            p[1] = b
        - Muto12:               http://iopscience.iop.org/article/10.1088/2041-8205/748/2/L22/pdf;jsessionid=B6BE571C45926447A1F986AD3060E2D8.c4.iopscience.cld.iop.org  (page 3)
            p[0] = theta_0
            p[1] = r_c
            p[2] = h_c
            p[3] = alpha
            p[4] = beta
    npt_model: float, opt
        Number of points of the model spiral. If not provided, it will automatically 
        pick 5x the number of azimuthal sections of the input simulation.
    weight_type: str, opt, {'more_weight_in', 'uniform'}
        Whether considering uniform weights to all points of the spiral trace, or
        to put more weight to inner points (because smaller spatial bins).
    symmetric_plot: bool, opt
        Whether to plot a symmetric spiral to the best fit (e.g. to compare)
    log_plot: bool, opt
        Whether to plot in log scale the intensities
    plot_fig: dict, opt, {1,2,3,4,5}
        Which figures to be plotted:
        1 -> just the density map;
        2 -> 1 + all local radial max;
        3 -> 2 + inner/outer spiral models to isolate the good spiral traces
        4 -> isolated trace of the spiral(s)
        5 -> 4 + best fit to the selected equation       
    ang_offset: offset between trigonometric angle and angle of the input polar map.
        Default is 90deg, considering the input array has its angle axis in PA unit.
    pix_to_dist_factor: flt, opt
        Number of au per pix    
    label: str, opt
        String used to differentiate the output. E.g. 'S2_Muto'
    search_mode: str, opt, {'linear','minimize'}
        Whether to find the chi_sqaure minimum with a linear search or using scipy.minimize.
        In the first case, the test ranges (e.g. 'rc_test', 'thc_test', 'hc_test', 'beta_test')
        have to be defined and passed as kwargs.
    dist: flt, opt
        Distance of the source in pc (in order to have physical distances as x and y labels)
    find_uncertainty: bool, opt
        Whether to look for the 1sigma uncertainty on each parameters of the fit
    step_uncertainty: flt, opt
        Step (in terms of fraction of best fit parameter value) used to search for the uncertainty.
    ms: flt, opt
        markersize when the trace is plotted
    n_subspi: int, opt
        number of spiral subsections to consider to compute the evolution of the 
        pitch angle along the trace.
    ratio_trace=None
    ratio_fit=[0,1]
    frac_rr_muto: list or tuple of 2 elements, opt
        Range in radius for the best fit model to be saved. Expressed in terms
        of radius of the first and last points identified in the trace.
        Only used for Muto/Rafikov equation fits.
        Default: [1,1]
    delta_theta_fit:
        Range in angles for the best fit model to be saved. Expressed in degrees
        with respect to the PA of the first and last points identified in the 
        trace.
        Used for all non-Rafikov/Muto fits.
        Default: [0,0]
    """
        
    pi = np.pi
    
    thc_test = kwargs.get('thc_test',None)
    rc_test = kwargs.get('rc_test',None)
    hc_test = kwargs.get('hc_test',None)
    beta_test = kwargs.get('beta_test',None)     
    alpha = kwargs.get('alpha',None)
    beta =  kwargs.get('beta',None)
    r_c = kwargs.get('r_c',None)
    th_c =  kwargs.get('th_c',None)
    h_c =  kwargs.get('h_c',None)
    
    # Reading input
    if clockwise: sign = -1.
    else: sign = 1.
    A_in = spiral_in_params[0]           # Inner spiral model, parameter A of equation r = A*exp(sign*B*theta)
    B_in = spiral_in_params[1]           # Inner spiral model, parameter B of equation r = A*exp(sign*B*theta)
    if spiral_out_params is not None:
        A_out = spiral_out_params[0]     # Outer spiral model, parameter A of equation r = A*exp(sign*B*theta)
        B_out = spiral_out_params[1]     # Outer spiral model, parameter B of equation r = A*exp(sign*B*theta)
    nrad = I_polar_arr.shape[0]
    nsec = I_polar_arr.shape[1]
    Rho = I_polar_arr
    rr = polar_coords[0]
    Rmin = rr[0]
    Rmax = rr[-1]
    rad_step = abs(rr[1]-rr[0])
    #theta = polar_coords[1]
    angles = np.linspace(0., 2.*np.pi, nsec, endpoint=False) # Here we implicitly assume the polar map to go from 0 to 360 deg
    theta, rad = np.meshgrid(angles, rr)

    # 1. Spiral tracing
    if ori_img is None:
        xi = rad * np.cos(np.deg2rad(theta))
        yi = rad * np.sin(np.deg2rad(theta))
    else:
        xi = np.arange(ori_img.shape[1])-ori_img.shape[1]/2
        yi = np.arange(ori_img.shape[0])-ori_img.shape[0]/2
    full_size = xi.shape[0]
    if full_size != yi.shape[0]:
        print("x and y dimensions are not the same. Please give a square input image.")
        pdb.set_trace()
    size = (full_size/2.)-0.5 # if bug, turn back to int
    plsc = pix_to_dist_factor/dist

    ### Binning of radial bins
    if bin_fact > 1:
        nrad_b = int(nrad/bin_fact)
        rr_b = []
        Rho_bin = np.zeros([nrad_b,nsec])
        for i in range(nrad_b):
            rr_b.append(Rmin+i*(Rmax-Rmin)/nrad_b) # new list of radii
            for j in range(nsec):
                Rho_bin[i,j] = np.mean(Rho[bin_fact*i:bin_fact*(i+1),j]) # new array of density values
    elif bin_fact == 1:
        nrad_b = nrad
        rr_b = rr
        Rho_bin = Rho
    else:
        raise ValueError('bin_fact should be > 0')
    
    ### r^2 scaling
    Rho_init = Rho_bin.copy()
    if r_square == 'multiply':
        for i in range(nrad_b):
            Rho_bin[i,:] *= rr_b[i]*rr_b[i]
        #thresh *= rr[1]**2
    elif r_square == 'divide':
        for i in range(nrad_b):
            Rho_bin[i,:] /= rr_b[i]*rr_b[i]
        #thresh /= rr[1]**2
    
    ### Derivatives
    first_deriv = np.zeros([nrad_b,nsec])
    sec_deriv = np.zeros([nrad_b,nsec])
    for sec in range(nsec):
        first_deriv[:,sec] = np.gradient(Rho_bin[:,sec])
        sec_deriv[:,sec] = np.gradient(first_deriv[:,sec])
    
    ### List the polar coords of all local radial max. above rmin_trace
    spi_r = []
    spi_theta = []
    for tt, angle in enumerate(angles):
        for ra, radius in enumerate(rr_b[:-1]):
            if radius >= rmin_trace and radius <= rmax_trace:
                if first_deriv[ra,tt]*first_deriv[ra+1,tt] < 0:
                    if abs(first_deriv[ra,tt]) < abs(first_deriv[ra+1,tt]):
                        kk = ra
                    else:
                        kk = ra+1
                    if sec_deriv[kk,tt] < 0:
                        if Rho_bin[kk,tt] > thresh:
                            spi_r.append(radius)
                            spi_theta.append(angle)
    spi_trace = np.array([spi_theta,spi_r])
    npt = len(spi_r)
  
    
    #####Spiral plotting
    def plot_dens(n, title_lab='Deprojected image', format_paper=True, 
                  log_plot=False, font_sz=12, label='a)', 
                  x_ax_lab=r'$\Delta$ RA', y_ax_lab=r'$\Delta$ DEC',
                  scale_as=0.1, scale_au=40, vmin=None, vmax=None, deproj=True):
        
        if ori_img is None:
            zi = Rho_init
            full_size = Rho_init.shape[0]
        else:
            zi = ori_img
            full_size = ori_img.shape[-1]
        if log_plot:
            zi = np.log10(zi)
            
        # IMSHOW (like rest of figures from Maddalena)
        ax1.set_xlabel(x_ax_lab+' (")',fontsize=font_sz)
        ax1.set_ylabel(y_ax_lab+' (")',fontsize=font_sz)
            
        if size*plsc > 5:
            delta_tick = 2.
        elif size*plsc > 2:
            delta_tick = 1.
        elif size*plsc > 1.2:
            delta_tick = 0.5
        elif size*plsc > 1.:
            delta_tick = 0.4
        elif size*plsc > 0.6:
            delta_tick = 0.2
        else:
            delta_tick = 0.1
            
        nticks = int(2*np.floor((size*plsc)/delta_tick)+1)
        
        xticks = [size+(i*delta_tick/plsc) for i in np.linspace(-int((nticks-1)/2),int((nticks-1)/2),nticks)]    
        xticks_lab = ["{:.1f}".format((x-size)*plsc) for x in xticks]
        if "RA" in x_ax_lab:
            xticks_lab = ["{:.1f}".format(-(x-size)*plsc) for x in xticks]
        yticks = [size+(i*delta_tick/plsc) for i in np.linspace(-int((nticks-1)/2),int((nticks-1)/2),nticks)] 
        yticks_lab = ["{:.1f}".format((y-size)*plsc) for y in yticks]

        ax1.set_xticks(xticks)
        ax1.set_xticklabels(xticks_lab,fontsize=12)
        ax1.set_yticks(yticks)
        ax1.set_yticklabels(yticks_lab,fontsize=12)            
            
        ax1.minorticks_on()
        ax1.plot([size], [size], color='white',lw=3,marker='*',mew=2, ms=8)
        if scale_as is not None:        
            ax1.plot([0.85*full_size,0.85*full_size+(scale_as/plsc)], [0.9*full_size,0.9*full_size], color='white',lw=3, marker="|",markeredgewidth=3)
            ax1.plot([0.85*full_size,0.85*full_size+(scale_au/pix_to_dist_factor)], [0.79*full_size,0.79*full_size], color='white',lw=3, marker="|",markeredgewidth=3)
            ax1.text(0.85*full_size,0.93*full_size, '{:.1f}"'.format(scale_as), color='white',fontsize=font_sz)
            ax1.text(0.85*full_size,0.82*full_size, '{:.0f} AU'.format(scale_au), color='white',fontsize=font_sz)            
        else:
            ax1.plot([0.85*full_size,0.85*full_size+(scale_au/pix_to_dist_factor)], [0.9*full_size,0.9*full_size], color='white',lw=3, marker="|",markeredgewidth=3)
            ax1.text(0.85*full_size,0.93*full_size, '{:.0f} AU'.format(scale_au), color='white',fontsize=font_sz)

        ax1.text(0.06*full_size,0.9*full_size, label, color='white',fontsize=font_sz)
        if not format_paper:
            ax1.set_title(title_lab,fontsize=font_sz)
        if vmin is not None and vmax is not None:
            ax1.imshow(zi, origin='lower',cmap='gist_heat',interpolation='nearest', vmin=vmin,vmax=vmax)
        else:
            ax1.imshow(zi, origin='lower',cmap='gist_heat',interpolation='nearest')
        #ax1.set_adjustable('datalim')   
            
        if deproj:
            if (size*plsc)*dist > 500:
                delta_tick_au = 200
            elif (size*plsc)*dist > 250:
                delta_tick_au = 100
            elif (size*plsc)*dist > 150:
                delta_tick_au = 50
            elif (size*plsc)*dist > 100:
                delta_tick_au = 40
            elif (size*plsc)*dist > 50:
                delta_tick_au = 20
            else:
                delta_tick_au = 10
                
            nticks_au = int(2*np.floor(size*pix_to_dist_factor/delta_tick_au)+1)            
            
            yticks_au = [size+(i*delta_tick_au/pix_to_dist_factor) for i in np.linspace(-int((nticks_au-1)/2),int((nticks_au-1)/2),nticks_au)] 
            yticks_lab_au = ["{:.1f}".format((y-size)*pix_to_dist_factor) for y in yticks_au]
            
            ax2 = ax1.twinx()      
            ax2.set_yticks(yticks_au)
            ax2.set_yticklabels(yticks_lab_au,fontsize=12)                  

            
            ax2.set_ylabel('d (AU)',fontsize=font_sz)
            
            
    
    #figure(1)
    if deproj:
        deproj_lab = '_deproj'
    else:
        deproj_lab=''        
    if 1 in plot_fig:
        fig = plt.figure(1,figsize=(0.9*7.5,0.9*7.5))
        ax1 = fig.add_subplot(111)
        plot_dens(1, log_plot=log_plot,font_sz=font_sz,label=label_fig,
                  scale_as=scale_as, scale_au=scale_au, vmin=vmin, vmax=vmax, deproj=deproj)
        plt.savefig(outpath+"SpiralImage{}.pdf".format(deproj_lab), dpi=300, bbox_inches='tight')
        plt.show()
    
    
    ###### Spiral traces
    xspi = spi_trace[1] * np.cos(spi_trace[0]+np.deg2rad(ang_offset))
    yspi = spi_trace[1] * np.sin(spi_trace[0]+np.deg2rad(ang_offset))
    
    if 2 in plot_fig:
        print(size)
        fig = plt.figure(2,figsize=(0.9*7.5,0.9*7.5))
        ax1 = fig.add_subplot(111)
        plot_dens(2,'All local radial maxima', log_plot=log_plot, font_sz=font_sz,
                  label=label_fig,scale_as=scale_as,scale_au=scale_au,vmin=vmin,vmax=vmax, deproj=deproj)
        ax1.plot(size+xspi, size+yspi, 'co', linewidth=1)
        plt.show()
    
    ##### Isolate the 2 biggest spirals (just one needed then take the opposite coords) by taking closest points to two log spiral models
    # As it is, the following method only works to trace at most a 2PI-long arc of the spiral
    
    # First make sure th_spi is an angle of numpy array "angles", if not adjust it
    th_spi_close = min(angles, key=lambda x:abs(x-th_spi))
    diff = th_spi - th_spi_close
    th_spi -= diff
    #spi_angs = np.linspace(th_spi, sign*2*pi+th_spi, nsec)
    # IMPORTANT, WE JUST WANT TO EXPRESS ALL ANGLES AS > 0 AND CONTINUOUS (e.g. no jump from 2*pi to 0 !)
    last_ang_spi = th_spi + sign*tot_range_spi
    if clockwise and last_ang_spi < 0:
        spi_angs = np.linspace(2*pi+th_spi, th_spi, nsec)
        if sep_ang < th_spi: sep_ang += 2*pi
        if bad_angs is not None:
            for ll in range(len(bad_angs)):
                if bad_angs[ll][0] < th_spi:
                    bad_angs[ll][0] = bad_angs[ll][0]+2.*pi
                if bad_angs[ll][1] < th_spi:
                    bad_angs[ll][1] = bad_angs[ll][1]+2.*pi
    elif clockwise and last_ang_spi >= 0:
        spi_angs = np.linspace(th_spi, -2*pi+th_spi, nsec)
    else:
        spi_angs = np.linspace(th_spi, 2*pi+th_spi, nsec)
        if sep_ang < th_spi: sep_ang += 2*pi
    
    ### Here you have to play with A_in, B_in, A_out and B_out - in the input parameter section - to find two satisfactory models or the different parts of the spiral
    ### Spiral that fits well the inner part
    r_in = A_in*np.exp(B_in*sign*(spi_angs-th_spi))
    u_in = r_in * np.cos(spi_angs+np.deg2rad(ang_offset))
    v_in = r_in * np.sin(spi_angs+np.deg2rad(ang_offset))
    ### Spiral that fits well the outer part
    if spiral_out_params is not None:
        r_out = A_out*np.exp(B_out*sign*(spi_angs-th_spi))
        u_out = r_out * np.cos(spi_angs+np.deg2rad(ang_offset))
        v_out = r_out * np.sin(spi_angs+np.deg2rad(ang_offset))
    
    if 3 in plot_fig:
        fig = plt.figure(3,figsize=(0.9*7.5,0.9*7.5))
        ax1 = fig.add_subplot(111)
        plot_dens(3, 'All local radial maxima and test model(s) to isolate specific spirals', 
                  log_plot=log_plot,font_sz=font_sz,label=label_fig,scale_as=scale_as,
                  scale_au=scale_au,vmin=vmin,vmax=vmax, deproj=deproj)
        npts_plt = int(tot_range_spi/(2*pi)*nsec)
        ax1.plot(size+xspi, size+yspi, 'bo', linewidth=1)
        ax1.plot(size+u_in[:npts_plt], size+v_in[:npts_plt], 'r') #inner spiral
        ax1.plot(size+u_out[:npts_plt], size+v_out[:npts_plt], 'y') #outer spiral
        plt.show()
    
    if iterate_eq_params:
        return None, None, None
    
    else:
        ### Isolate the good points
        #bad_angs = [-2*pi+th_spi,-pi/2.] # Set this to min and max angle values that are not subtended by the spiral (in terms of spi_angs values). Note: here the spiral subtends 2pi, so both arguments are set to same value for no restriction.
        spi_1_r = []
        spi_1_theta = []
        if bad_angs is None:
            if clockwise:
                bad_angs = [[spi_angs[0],spi_angs[-1]]]
            else:
                bad_angs = [[spi_angs[-1],spi_angs[0]]]
    
        for tt, ang in enumerate(spi_angs):
            condition_bad_ang = True
            for ll in range(len(bad_angs)):
                if ang < 0 or (ang > bad_angs[ll][0] and ang < bad_angs[ll][1]):
                    condition_bad_ang = False # check next angle, because ang can be negative only for angles not subtended by the spiral
            if not condition_bad_ang:
                continue
            elif ang > 2*pi:
                ang_comp = ang - 2.*pi
            else:
                ang_comp = ang
            condition_spi = False
            if clockwise and last_ang_spi < 0 and ang >= 2*pi+th_spi-tot_range_spi:
                condition_spi = True
            elif clockwise and last_ang_spi >= 0 and ang >= th_spi-tot_range_spi:
                condition_spi = True
            elif not clockwise and ang <= th_spi+tot_range_spi:
                condition_spi = True
            if condition_spi:
                r_subset = []
                for pt in range(npt):
                    if abs(spi_trace[0,pt]-ang_comp) < pi/nsec:
                        r_subset.append(spi_trace[1,pt])
                if len(r_subset) > 0:
                    if spiral_out_params is not None:
                        if (clockwise and ang > sep_ang) or (not clockwise and ang < sep_ang):
                            spi_1_r.append(min(r_subset, key=lambda x:abs(x-r_in[tt]))) # the good r among the ones in r_subset is the one closest from the inner spiral model r_in
                        else:
                            spi_1_r.append(min(r_subset, key=lambda x:abs(x-r_out[tt]))) # the good r among the ones in r_subset is the one closest from the outer spiral model r_out
                    else:
                        spi_1_r.append(min(r_subset, key=lambda x:abs(x-r_in[tt])))
                    spi_1_theta.append(ang)
    
        first_ang = spi_1_theta[0] 
        print("First angle of the spiral: ", np.rad2deg(spi_1_theta[0]))
        with open(txt_file, "w+") as f:
            f.write('###############################################\n')
            f.write('####   RESULTS OF SPIRAL CHARACTERISATION   ###\n')
            f.write('###############################################\n')
            f.write("First angle of the spiral: {:.1f}deg\n".format(np.rad2deg(spi_1_theta[0])))
            f.write("Last angle of the spiral: {:.1f}deg\n".format(np.rad2deg(spi_1_theta[-1])))
            f.write("Length of the spiral: {:.1f}deg\n".format(np.rad2deg(spi_1_theta[-1]-spi_1_theta[0])))
        npt = len(spi_1_r)
        
        ## GET final trace + uncertainty by fitting radial gaussians
        if gauss_fit or weight_type == "gauss_fit":
            spi_trace_gauss= np.zeros([npt,3])   
            spi_trace_gauss[:,0] = (np.rad2deg(spi_1_theta)+ang_offset-90)%360 # convert to theta then back to PA
            spi_trace_gauss[:,1] = np.array(spi_1_r)*plsc
            
            for tt, trace_ang in enumerate(spi_1_theta):
                idx_tt = find_nearest(angles, trace_ang%(2*np.pi))
                idx_rr = find_nearest(rr_b, spi_1_r[tt])
                idx_ini = max(0,int(idx_rr-fwhm/rad_step)) #0.5?
                idx_fin = min(nrad,int(idx_rr+fwhm/rad_step))
                # CURVE FIT
                p0 = [Rho_bin[idx_rr,idx_tt]-np.amin(Rho_bin[idx_ini:idx_fin,idx_tt]), 
                      rr_b[idx_rr], (rr_b[idx_fin]-rr_b[idx_ini])/5]
                try:
                    coeff, var_matrix = curve_fit(gauss, rr_b[idx_ini:idx_fin], 
                                                  Rho_bin[idx_ini:idx_fin,idx_tt]-np.amin(Rho_bin[idx_ini:idx_fin,idx_tt]), 
                                                  p0=p0)
                    coeff_err = np.sqrt(np.diag(var_matrix))
                    spi_trace_gauss[tt,1] = coeff[1]*plsc
                    spi_trace_gauss[tt,2] = coeff_err[1]*plsc
                except:
                    with open(txt_file, "a") as f:
                        f.write(' \n')
                        f.write('WARNINGS: \n')
                        f.write("Gaussian fit of local max failed at PA {:.1f} deg \n".format(np.rad2deg(trace_ang%(2*np.pi))))
                    print("Gaussian fit of local max failed at PA {:.1f} deg".format(np.rad2deg(trace_ang%(2*np.pi))))
                    print("Press c if you accept to continue and: 1) set r to closest pixel max; 2) set the uncertainty on r to be fwhm/2.")
                    plt.figure()
                    plt.plot(rr_b[idx_ini:idx_fin],
                             Rho_bin[idx_ini:idx_fin,idx_tt]-np.amin(Rho_bin[idx_ini:idx_fin,idx_tt]),
                             'bo')
                    plt.show()
                    pdb.set_trace()                     
                    spi_trace_gauss[tt,2] = fwhm*plsc/2.
                    
                
            write_fits(outpath+"Trace_PAdeg_Rarcsec_Runc_gauss.fits", spi_trace_gauss)
            if gauss_fit:
                spi_1_r = spi_trace_gauss[:,1]/plsc # otherwise only the uncertainties will be used
            
        spi_1_coords = np.array([spi_1_theta,spi_1_r])
    
        ### Compute pitch angle from points of the trace:
        # tan(Pitch angle) = 1/r dr/dphi
        r_in_mean = np.mean(spi_1_coords[1,:int(npt/2)])
        r_out_mean = np.mean(spi_1_coords[1,int(npt/2):])
        r_mean = np.mean(spi_1_coords[1,:])
        tan_pitch_in = (1./r_in_mean)*(spi_1_r[int(npt/2)]-spi_1_r[0])/np.abs(spi_1_theta[int(npt/2)]-spi_1_theta[0])
        tan_pitch_out = (1./r_out_mean)*(spi_1_r[-1]-spi_1_r[int(npt/2)])/np.abs(spi_1_theta[int(npt/2)]-spi_1_theta[-1])
        tan_pitch_mean = (1./r_mean)*(spi_1_r[-1]-spi_1_r[0])/np.abs(spi_1_theta[-1]-spi_1_theta[0])
        pitch_ang_in = np.rad2deg(math.atan(tan_pitch_in))
        pitch_ang_out = np.rad2deg(math.atan(tan_pitch_out))
        pitch_ang_mean = np.rad2deg(math.atan(tan_pitch_mean))      
        print("spiral trace pitch angle (1st half): ", pitch_ang_in)
        print("spiral trace pitch angle (2nd half): ", pitch_ang_out)
        print("spiral trace pitch angle (global): ", pitch_ang_mean)
            
        # pitch angle for each pair of points
        pitch_ang = np.zeros([2,npt-1])
        for ii in range(npt-1):
            pitch_ang[0,ii] = (spi_1_coords[0,ii]+spi_1_coords[0,ii+1])/2.
            r_tmp = (spi_1_coords[1,ii]+spi_1_coords[1,ii+1])/2. 
            tmp = (1./r_tmp)*(spi_1_coords[1,ii+1]-spi_1_coords[1,ii])/np.abs(spi_1_coords[0,ii+1]-spi_1_coords[0,ii])
            pitch_ang[1,ii] = np.rad2deg(math.atan(tmp)) 
        write_fits(outpath+"Pitch_angles_{}.fits".format(label), pitch_ang)

        print("Mean pitch angle (from all consecutive pairs) {}: {}deg \n".format(label,np.mean(pitch_ang[1,:])))
        print("Median pitch angle  (from all consecutive pairs) {}: {}deg\n ".format(label,np.median(pitch_ang[1,:])))
        print("Stddev pitch angle  (from all consecutive pairs) {}: {}deg\n ".format(label, np.std(pitch_ang[1,:])))

        with open(txt_file, "a") as f:
            f.write(' \n')
            f.write('### SPIRAL TRACE ###\n')
            f.write("spiral trace pitch angle (1st half): {}deg \n".format(pitch_ang_in))
            f.write("spiral trace pitch angle (2nd half): {}deg \n".format(pitch_ang_out))
            f.write("spiral trace pitch angle (global): {}deg\n".format(pitch_ang_mean))
            f.write("Mean pitch angle  (from all consecutive pairs) {}: {}deg \n".format(label,np.mean(pitch_ang[1,:])))
            f.write("Median pitch angle  (from all consecutive pairs) {}: {}deg\n ".format(label,np.median(pitch_ang[1,:])))
            f.write("Stddev pitch angle  (from all consecutive pairs) {}: {}deg\n ".format(label, np.std(pitch_ang[1,:])))     
            
        ### Plot the good traces
        xspi_1 = size+ spi_1_coords[1] * np.cos(spi_1_coords[0]+np.deg2rad(ang_offset))
        yspi_1 = size+ spi_1_coords[1] * np.sin(spi_1_coords[0]+np.deg2rad(ang_offset))
        write_fits(outpath+"Trace_{}_xy.fits".format(label), np.array([xspi_1,yspi_1]))  

        if ratio_trace is None:
            ratio_trace = 1
        if 4 in plot_fig:
            fig = plt.figure(4,figsize=(0.9*7.5,0.9*7.5))
            ax1 = fig.add_subplot(111)
            plot_dens(4, 'Isolated spiral traces', log_plot=log_plot,font_sz=font_sz,
                      label=label_fig,scale_as=scale_as,scale_au=scale_au,vmin=vmin,vmax=vmax, deproj=deproj)

            npts = xspi_1.shape[0]
            cutoff = int(npts*ratio_trace)
            ax1.plot(xspi_1[:cutoff], yspi_1[:cutoff], color_trace, linewidth=2, markersize=ms)
            ax1.plot(xspi_1[cutoff:], yspi_1[cutoff:], color_trace2, linewidth=2, markersize=ms)
            #plt.plot(xspi_2, yspi_2, 'ro', linewidth=1)
            plt.savefig(outpath+"Spiral_trace_{}.pdf".format(label), dpi=300, bbox_inches='tight')
            plt.show()
    
        npts = spi_1_coords.shape[1]
        cutoff = int(npts*ratio_trace)
        spi_1_coords = spi_1_coords[:,:cutoff]

            
        #### convert to polar
        r_trace = spi_1_coords[1]*plsc
        PA_trace = (np.rad2deg(spi_1_coords[0])+ang_offset-90)%360
        write_fits(outpath+"Trace_{}_Rarcsec_PAdeg.fits".format(label), np.array([r_trace,PA_trace]))  
   
        ##### Fit the first spiral to spiral equations (should be the same for the symmetric one)
        ### First compute the radial width at root and tip of each spiral, in order to estimate the error bar of each point
        npts = spi_1_coords.shape[1]
        if weight_type == 'gauss_fit':
            errors = spi_trace_gauss[:,2]/plsc # to convert back to pixels
            for ee in range(errors.shape[0]):
                errors[ee] = max(errors[ee],fwhm/4.)
            print("errors on radial separation of trace (max between half kernel and gauss unc): {}".format(errors))
            with open(txt_file, "a") as f:
                f.write(' \n')
                f.write('Errors on radial separation of trace (max between half kernel and gauss unc):\n')
                if np.std(errors)==0:
                    f.write('All set to: {}\n'.format(errors[0]))
        elif weight_type != 'individual':
            rad_width = np.zeros(2)
            for ee, extr in enumerate([0,npts-1]):
                ang_extr = spi_1_coords[0,extr]
                r_extr = spi_1_coords[1,extr]
                if ang_extr < 0:
                    ang_extr += 2.*pi
                elif ang_extr > 2*pi:
                    ang_extr -= 2.*pi
                ang_idx = np.where(abs(angles-ang_extr)<pi/(4*nsec))[0][0]
                rr_idx = np.where(rr_b==r_extr)[0][0]
                for kk in range(int(nrad/2)):
                    if sec_deriv[rr_idx-kk,ang_idx] > 0:
                        rr_in = rr_idx-kk
                        break
                for kk in range(int(nrad/2)):
                    if sec_deriv[rr_idx+kk,ang_idx] > 0:
                        rr_out = rr_idx+kk
                        break
                rad_width[ee] = rr_b[rr_out]-rr_b[rr_in]
            print("radial width at spiral root and tip: ", rad_width)
            with open(txt_file, "a") as f:
                f.write(' \n')
                f.write("radial width at spiral root and tip: {}\n".format(rad_width))
            # We assume the radial width is roughly half the radial distance between two points of positive curvature in radial density profile close to the spiral
            if weight_type == 'more_weight_in':
                errors = np.array([np.linspace(rad_width[0]/2.,rad_width[1]/2.,npts)]) # uncomment if you want to give more weight to inner values
            elif weight_type == 'uniform':
                errors = np.zeros(npts)
                errors[:]= rad_width[-1]/2.
            else:
                raise ValueError('Pick a valid value for weight_type')
        else:
            errors = np.zeros(npts)
            for ee, extr in enumerate(range(npts)):
                ang_extr = spi_1_coords[0,extr]
                r_extr = spi_1_coords[1,extr]
                if ang_extr < 0:
                    ang_extr += 2.*pi
                elif ang_extr > 2*pi:
                    ang_extr -= 2.*pi
                ang_idx = np.where(abs(angles-ang_extr)<pi/(nsec/2))[0][0]
                rr_idx = np.where(rr_b==r_extr)[0][0]
                for kk in range(int(nrad/2)):
                    if sec_deriv[rr_idx-kk,ang_idx] > 0:
                        rr_in = rr_idx-kk
                        break
                for kk in range(int(nrad/2)):
                    if sec_deriv[rr_idx+kk,ang_idx] > 0:
                        rr_out = rr_idx+kk
                        break
                errors[ee] = (rr_b[rr_out]-rr_b[rr_in])/2.
            print("error on radial separation of trace: ", errors)
            with open(txt_file, "a") as f:
                f.write(' \n')
                f.write("error on radial separation of trace: {}\n".format(errors))
        #### uncertainty on r
        write_fits(outpath+"Trace_{}_Rarcsec_unc.fits".format(label), errors*plsc) 
    
        ### Do the spiral fitting
        print("Computing the best fit parameters for each spiral...")
        spi_1_coords_fit = spi_1_coords.copy()
        spi_range = abs(spi_1_coords_fit[0,0]-spi_1_coords_fit[0,-1])
        fit_coords_tmp = np.linspace(spi_range,0,spi_1_coords_fit.shape[1],endpoint=False)
        spi_1_coords_fit[0] = fit_coords_tmp[::-1]
        
        #### FIND THE BEST FITS PARAMETERS 
        with open(txt_file, "a") as f:
            f.write(' \n')
            f.write('### SPIRAL MODEL ###\n')
            f.write("The best fit parameters to the {} equation are found by {} search. \n".format(search_mode, fit_eq))
            f.write("Note: the fitted equation is: {}\n".format(spiral_eq_definition(fit_eq,alpha=alpha, beta=beta, r_c=r_c, th_c=th_c, h_c=h_c)))
        if search_mode == 'minimize':
            best_params = spiral_minimize(param_estimate, spi_1_coords_fit, errors, method=solver, options=solver_options,
                                          bounds=bounds, fit_eq=fit_eq, verbose=True, fac_hr=fac_hr, **kwargs) # if bug: add clockwise=-sign again
            print(best_params)
            if fit_eq == 'gen_archi':
                print("Best params (in arcsec): a = {:.4f}''; b = {:.4f}''/rad; n={:.3f}".format(best_params[0]*plsc,best_params[1]*plsc, best_params[2]))
                with open(txt_file, "a") as f:
                    f.write("Best params (in arcsec): a = {:.4f}''; b = {:.4f}''/rad; n={:.3f} \n".format(best_params[0]*plsc,best_params[1]*plsc, best_params[2]))
            elif fit_eq == 'log':
                r_0 = best_params[0]*plsc
                b = best_params[1]
                phi = np.rad2deg(math.atan(b))
                print("Best params (in arcsec): r_0 = {:.4f}''; b = {:.4f}".format(r_0,b))
                print("The b parameter corresponds to a pitch angle: phi={:.3f}".format(phi))
                with open(txt_file, "a") as f:
                    f.write("Best params (in arcsec): r_0 = {:.4f}''; b = {:.4f}\n".format(r_0,b))
                    f.write("The b parameter corresponds to a pitch angle: phi={:.3f}\n".format(phi))
                    

        elif search_mode == 'linear':
            if 'muto12' not in fit_eq:
                print("Use minimize for any spiral equation not involving the Muto+12 one")
                pdb.set_trace()
            best_chi, best_params = spirals_muto_rc_thc_hc_beta_linsearch(thc_test,rc_test,hc_test, beta_test,spi_1_coords_fit, errors_tmp, alpha, 
                                                                          verbose=True, nproc=1, clockwise=-sign)
        else:
            raise ValueError("search mode not recognized: pick 'minimize' or 'linear'")
                                                                  
        if fit_eq == 'gen_archi':
            spi_1_coords_crop = spi_1_coords[:,:-1].copy()
            r_fit = spiral_eq_eval(spi_1_coords_crop[1], best_params, fit_eq=fit_eq)  # if bug: add clockwise=-sign again
            numerator_fit = best_params[2]*(r_fit - best_params[0])/spi_1_coords_crop[0]
            pitch_vec = np.rad2deg(np.arctan(numerator_fit/r_fit))
            print("Pitch angle varies from: ", pitch_vec[0], " to: ", pitch_vec[-1])
            print("Global mean pitch angle: ", np.mean(pitch_vec))
            print("Global stddev pitch angle: ", np.std(pitch_vec))
            with open(txt_file, "a") as f:
                f.write("Pitch angle varies from: {} to: {}\n".format(pitch_vec[0], pitch_vec[-1]))
                f.write("Global mean pitch angle: {}\n".format(np.mean(pitch_vec)))
                f.write("Global stddev pitch angle: {}\n".format(np.std(pitch_vec)))
        if 'muto12' in fit_eq and '1param' not in fit_eq and 'hc_beta' not in fit_eq:
            print("Best fit location of the companion (theta[deg],r[au]): ", (first_ang+sign*np.rad2deg(best_params[0])%360,best_params[1]*pix_to_dist_factor))
            print("Best fit location of the companion (theta[rad],r['']): ", (first_ang+sign*best_params[0],best_params[1]*pix_to_dist_factor/dist))
            with open(txt_file, "a") as f:
                f.write("Best fit location of the companion (theta[deg],r[au]): ({},{})\n".format(np.rad2deg(best_params[0]+first_ang)%360,best_params[1]*pix_to_dist_factor))
                f.write("Best fit location of the companion (theta[rad],r['']): ({},{})\n".format(best_params[0]+first_ang,best_params[1]*pix_to_dist_factor/dist))
        red_chisq = chisquare(best_params, spi_1_coords_fit, errors, fit_eq=fit_eq, reduced=True, alpha=alpha, beta=beta, r_c=r_c, th_c=th_c, h_c=h_c)  # if bug: add clockwise=-sign again
        #chisq = chisquare(best_params, spi_1_coords_fit, errors, fit_eq=fit_eq, reduced=False, clockwise=-sign, alpha=alpha, beta=beta, r_c=r_c, th_c=th_c)
        print("Reduced chi-square of the fit "+fit_eq+": ", red_chisq)
    
        ### Find uncertainty on the params and the pitch angle
        if find_uncertainty:# and not 'muto12' in fit_eq:
            uncertainties = find_params_unc(spi_1_coords_fit, errors, best_params, 
                                            fit_eq, clockwise, fwhm, 
                                            step_unc=step_uncertainty,
                                            verbose=True, outpath=outpath, 
                                            res_file=txt_file, n_boots=n_boots, 
                                            n_proc=n_proc, bootstrap=True, 
                                            method=solver, options=solver_options,
                                            bounds=bounds, **kwargs)
            print("Uncertainties on each parameter of the fit:", uncertainties)
            with open(txt_file, "a") as f:
                f.write("Uncertainties on each parameter of the fit: {} \n".format(uncertainties))            
            write_fits(outpath+"Uncertainties_best_fit_params_{}.fits".format(label),uncertainties)
            #return best_pitch, min(low_unc,up_unc), max(low_unc,up_unc)
            
            
        ### If pitch angle to be measured in different portions of the spiral (n_subspi>1)
        if n_subspi>1 and fit_eq=='log': 
            phi = np.zeros([4, n_subspi]) # will contain mean_PA, mean_r, phi, unc_phi for each subsection
            npts_sec = int(npts/n_subspi)
            with open(txt_file, "a") as f:
                f.write("\n") 
                f.write("### PITCH ANGLE ### \n")   
            for nn in range(n_subspi):
                idx_ini = nn*npts_sec
                idx_fin = (nn+1)*npts_sec
                if nn == n_subspi-1:
                    idx_fin = -1
                spi_1_coords_sec = spi_1_coords[:,idx_ini:idx_fin]
                errors_sec = errors[idx_ini:idx_fin]
                phi[0,nn] = np.mean(np.rad2deg(spi_1_coords_sec[0])) # get it in deg
                phi[1,nn] = np.mean(spi_1_coords_sec[1])*plsc # px
                phi[2,nn], phi[3,nn] = measure_pitch_angle(spi_1_coords_sec, errors_sec, 
                                                     clockwise, fwhm,
                                                     step_unc=step_uncertainty,
                                                     p_ini=best_params, 
                                                     unc=True, verbose=True)
                with open(txt_file, "a") as f:
                    f.write("phi in section {:.0f}/{:.0f}: {:.1f}+-{:.1f}deg \n".format(nn+1, n_subspi, phi[2,nn], phi[3,nn]))   
            write_fits(outpath+"Pitch_angle_{}sec_PAdeg_Rarcsec_phi_uncphi_{}.fits".format(n_subspi,label),phi)
    
        ### Evaluate the best_fit model 
        if not npt_model: npt_model = nsec*5
        spi_1_bfit = np.zeros([2,npt_model])
        if 'muto12' in fit_eq:
            # IN THE CASE OF MUTO, the equation is theta(r) and not r(theta), hence a trick has to be used to get the right PA range
            rr_tmp_ori = frac_rr_muto[0]*spi_1_coords_fit[1,0]
            rr_tmp_fin = frac_rr_muto[-1]*spi_1_coords_fit[1,-1]
            PA_tmp_fin = spiral_eq_eval(rr_tmp_fin, best_params, fit_eq=fit_eq, alpha=alpha, beta=beta, r_c=r_c, th_c=th_c, h_c=h_c)        # if bug: add clockwise=-sign again 
            dist_PA_0 = abs(PA_tmp_fin-spi_1_coords_fit[0,-1])
            for dumb in range(nrad_b):
                rr_tmp_fin += rad_step
                PA_tmp_fin = spiral_eq_eval(rr_tmp_fin, best_params, fit_eq=fit_eq, alpha=alpha, beta=beta, r_c=r_c, th_c=th_c, h_c=h_c)    # if bug: add clockwise=-sign again
                dist_PA = abs(PA_tmp_fin-spi_1_coords_fit[0,-1])
                if dist_PA > dist_PA_0:
                    rr_tmp_fin -= rad_step
                    break                
            
            spi_1_bfit[1] = np.linspace(rr_tmp_ori, rr_tmp_fin, npt_model)
            spi_1_bfit[0] = spiral_eq_eval(spi_1_bfit[1], best_params, fit_eq=fit_eq, alpha=alpha, beta=beta, r_c=r_c, th_c=th_c, h_c=h_c)  # if bug: add clockwise=-sign again
        else:
            spi_1_bfit[0] = np.linspace(np.deg2rad(delta_theta_fit[0])+spi_1_coords_fit[0,0], 
                                        np.deg2rad(delta_theta_fit[-1])+spi_1_coords_fit[0,-1], npt_model)
            spi_1_bfit[1] = spiral_eq_eval(spi_1_bfit[0], best_params, fit_eq=fit_eq, alpha=alpha, beta=beta, r_c=r_c, th_c=th_c, h_c=h_c)  # if bug: add clockwise=-sign again
        spi_1_bf = spi_1_bfit.copy()
        spi_1_bf[0] = sign*spi_1_bfit[0]+spi_angs[0]
        spi_2_bf = np.zeros_like(spi_1_bf)
        for pt in range(npt_model):
            if spi_1_bf[0,pt] >= pi:
                spi_2_bf[0,pt] = spi_1_bf[0,pt]-pi
            else:
                spi_2_bf[0,pt] = spi_1_bf[0,pt]+pi
        spi_2_bf[1,:] = spi_1_bf[1]
        
        ### Model       
        spi_1_bf_good_units = spi_1_bf.copy()
        spi_1_bf_good_units[1] = spi_1_bf[1]*plsc
        write_fits(outpath+"Best_fit_{}_PArad_Rarcsec.fits".format(label), spi_1_bf_good_units)  

        ### Plot the best fit spiral model
        xspi_m1 = size+ spi_1_bf[1] * np.cos(spi_1_bf[0]+np.deg2rad(ang_offset))
        yspi_m1 = size+ spi_1_bf[1] * np.sin(spi_1_bf[0]+np.deg2rad(ang_offset))
        xspi_m2 = size+ spi_2_bf[1] * np.cos(spi_2_bf[0]+np.deg2rad(ang_offset))
        yspi_m2 = size+ spi_2_bf[1] * np.sin(spi_2_bf[0]+np.deg2rad(ang_offset))
        if 5 in plot_fig:
            fig = plt.figure(5,figsize=(0.9*7.5,0.9*7.5))
            ax1 = fig.add_subplot(111)
            plot_dens(5, 'Isolated spiral traces and best fit to model: '+fit_eq, format_paper=True, 
                      log_plot=log_plot,font_sz=font_sz,label=label_fig,scale_as=scale_as,scale_au=scale_au,
                      vmin=vmin,vmax=vmax, deproj=deproj)

            npts = xspi_1.shape[0]
            cutoff = int(npts*ratio_trace)
            ax1.plot(xspi_1[:cutoff], yspi_1[:cutoff], color_trace, linewidth=1, markersize = ms)
            ax1.plot(xspi_1[cutoff:], yspi_1[cutoff:], color_trace2, linewidth=1, markersize = ms)

            npts = xspi_m1.shape[0]
            cut1 = int(npts*ratio_fit[0])
            cut2 = int(npts*ratio_fit[1])
            ax1.plot(xspi_m1[cut1:cut2], yspi_m1[cut1:cut2], color_fit, linewidth=2)
            if symmetric_plot:
                ax1.plot(xspi_m2, yspi_m2, 'b--', linewidth=2)
            #pylab.show()
            plt.savefig(outpath+"Spiral_model_{}.pdf".format(label), dpi=300, bbox_inches='tight')
            plt.show()
        write_fits(outpath+"Best_fit_{}_xy.fits".format(label), np.array([xspi_m1,yspi_m1]))
        return best_params, np.array([xspi_1,yspi_1]), np.array([xspi_m1,yspi_m1]), spi_1_bf #best fit params, spiral trace, spiral model, spiral model (polar)
    