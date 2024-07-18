from astropy.stats import sigma_clipped_stats
import hciplot as hp
from matplotlib import pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import norm
import vip_hci as vip
from vip_hci.preproc import frame_rotate
from vip_hci.preproc.rescaling import cube_rescaling_wavelengths
from vip_hci.var import frame_center, mask_circle
try:
    from vip_hci.var import dist_matrix
except:
    from vip_hci.var.shapes import dist_matrix
    print('Warning: A newer version of VIP is available.')
from vip_hci.stats import cube_distance
import pdb

__all__ = ['infer_relative_rotation']


def infer_relative_rotation(frames, plsc, test_rot, shifts_xy=None, rmask=0,
                            rout=None, thr1=10, rsquare=1, thr2=0,
                            dist='pearson', rotang=0, upscal=True, debug=False,
                            figname='Comparison_frames.pdf',
                            fig_labels=None, fitsname=None):
    """
    Routine to infer relative rotation between 2nd to Nth frame with respect to
    the first one.

    Parameters
    **********
    frames: list or tuple of 2D numpy arrays
        Frames to be compared. If tuple of lenght N, a tuple of N-1 relative
        rotations will be returned
    plsc: numpy array
        Should contain the plate scale of each input frame
    test_rot: numpy array 1D
        Vector of rotation angles to be tested.
    shifts_xy: numpy array, opt
        Should contain the shifts (dimension: nframes x 2) with respect to
        star positioned on the central pixel(s). If None, assumes the frames
        are perfectly centered.
    rmask: float, opt
        If non-zero, will mask the inner part of the image before cross
        correlation. Should be provided in arcsec.
    rout: float, opt
        Radius beyond which all values are set to zero (considered noise).
        If None will be computed automatically from stddev and thr1 in the images.
    thr1: float or tuple of floats, opt
        If not zero, will threshold the image before rsquare, i.e. set values
        to 0 if below the threshold. Threshold should be provided in terms of
        standard deviation (which will be computed robustly for each image using
        astropy.stats.sigma_clipped_stats with sigma=2.5). If a tuple, the
        length should match the number of frames.
    rsquare: int, or tuple of int, opt
        Whether to rescale the image by r^2 (1). If set to 0.5, will only
        rescale by r, while 0 will not rescale radially. If a tuple, length
        should match number of frames.
    thr2: float or tuple of floats, opt
        Same as thr1 but applied after rsquare (only applied if non zero).
    dist: str, opt
        Method to measure the "distance" or cross-correlation.
        Check description of vip.stats.cube_distance() for all options;.
    rotang: float or tuple of floats, opt
        Initial rotation(s) to be applied to the frames to align North up
    upscal: bool, opt
        Whether to upscale (or leave intact) all images (i.e. by considering
        the finest image sampling, and rescaling the other images to that one).
        If false, all images are downsampled to the coarsest sampling.
    debug: bool, opt
        Whether to print and plot intermediate outputs for debugging purpose
    figname: str, opt
        If debug is set to True, this should be the full path + name of the
        file in which the comparison figure of the different input frames will
        be saved.
    fitsname: str, opt
        If debug is set to True, this should be the full path + name of the
        fits file in which the final prepared images will be saved.

    Returns
    *******
    delta_rot: numpy array
        Dimensions are (nsamples, nframes-1)
    """


    def _gauss(x, *p):
        A, mu, sigma = p
        return A*np.exp(-(x-mu)**2/(2.*sigma**2))

    nfr = len(frames)
    ncomp = nfr-1

    # convert variables to tuple if needed
    if not isinstance(thr1, (list, tuple)):
        thr1 = [thr1]*nfr
    if not isinstance(rsquare, (list, tuple)):
        rsquare = [rsquare]*nfr
    if not isinstance(thr2, (list, tuple)):
        thr2 = [thr2]*nfr
    if not isinstance(rotang, (list, tuple)):
        rotang = [rotang]*nfr

    if upscal:
        plsc_com = min(plsc)
        max_sz = 0
        for ii in range(nfr):
            if frames[ii].shape[0] > max_sz:
                max_sz = frames[ii].shape[0]
    else:
        plsc_com = max(plsc)

    ccor = np.zeros([ncomp,test_rot.shape[0]])
    stddev = np.zeros(nfr)

    # Preparation of all the frames
    for ii in range(nfr):
        frames_tmp = frames[ii].copy()
        ## replace nan with zeros
        frames_tmp[np.where(np.isnan(frames_tmp))] = 0

        ## Resample to same px scale
        if plsc_com!= plsc[ii]:
            if upscal:
                new_frames_tmp = np.zeros([max_sz,max_sz])
                new_cy, new_cx = frame_center(new_frames_tmp)
                new_frames_tmp[:frames_tmp.shape[0],:frames_tmp.shape[1]] = frames_tmp
                old_cy, old_cx = frame_center(frames_tmp)
                frames_tmp = vip.preproc.frame_shift(new_frames_tmp,
                                                     new_cy-old_cy,
                                                     new_cx-old_cx)
            frames_tmp = cube_rescaling_wavelengths(np.array([frames_tmp]),
                                                    [plsc[ii]/plsc_com],
                                                    ref_xy=None, imlib='opencv',
                                                    interpolation='lanczos4',
                                                    scaling_y=None,
                                                    scaling_x=None)[0]

        ## Mask center (just for thresholding)
        if rmask > 0:
            frame_mask = mask_circle(frames_tmp, rmask/plsc_com)
        else:
            frame_mask = frames_tmp.copy()

        ## Compute threshold if not rout
        if rout is None:
            _, _, stddev[ii] = sigma_clipped_stats(frames_tmp[np.where(frame_mask!=0)], sigma=2.5)

        ## Crop any rectangle to square
        if frames_tmp.shape[0]!=frames_tmp.shape[1]:
            frames_tmp = vip.preproc.frame_crop(frames_tmp,min(frames_tmp.shape)-2)
            frame_mask = vip.preproc.frame_crop(frame_mask,min(frame_mask.shape)-2)

        ## rescale by r^2
        r_fr1 = dist_matrix(frame_mask.shape[0])
        if rsquare[ii] == 1:
            frame_mask = frame_mask*np.power(r_fr1,2)
            #frames_tmp = frames_tmp*np.power(r_fr1,2)
        elif rsquare[ii] == 0.5:
            frame_mask = frame_mask*r_fr1

        ## Crop to same size - and possibly shift if not all even/odd
        ### find largest radius of non-zero signal
        bin_mask = np.where(frames_tmp,np.ones_like(frames_tmp),
                            np.zeros_like(frames_tmp))
        max_crop_sz = int(2*np.amax(r_fr1*bin_mask))+1
        ### define crop size
        if ii == 0:
            crop_sz = max_crop_sz
            for kk in range(nfr):
                if int(min(frames[kk].shape)*plsc[kk]/plsc_com) < crop_sz:
                    crop_sz = int(min(frames[kk].shape)*plsc[kk]/plsc_com)
            final_frames = np.zeros([nfr,crop_sz,crop_sz])
        ### crop
        if crop_sz < frames_tmp.shape[1]:
            if crop_sz%2 != frames_tmp.shape[1]%2:
                frames_tmp = vip.preproc.frame_shift(frames_tmp,0.5,0.5)
                frames_tmp = frames_tmp[1:,1:]
            if crop_sz < frames_tmp.shape[1]:
                frames_tmp = vip.preproc.frame_crop(frames_tmp,crop_sz)
        final_frames[ii] = frames_tmp.copy()

        ## shift the frames accordingly
        if shifts_xy is not None:
            final_frames[ii] = vip.preproc.frame_shift(final_frames[ii], shifts_xy[ii,1], shifts_xy[ii,0])

        ## Mask cavity + threshold + r^2 (for real) - AFTER ROTATION for FRAME 1!
        if ii > 0:
            rcav_px = rmask/plsc_com
            final_frames[ii]= mask_circle(final_frames[ii], rcav_px)

            ## Thresholdings
            if rout is None:
                final_frames[ii][np.where(final_frames[ii]<thr1[ii]*stddev[ii])] = 0.
            else:
                rout_px = rout/plsc_com
                final_frames[ii] = mask_circle(final_frames[ii], rout_px, mode='out')

            ## rescale by r^2 [MOVED AT THE END]
            r_fr1 = dist_matrix(final_frames[ii].shape[0])
            if rsquare[ii] == 1:
                final_frames[ii] = final_frames[ii]*np.power(r_fr1,2)
            elif rsquare[ii] == 0.5:
                final_frames[ii] = final_frames[ii]*r_fr1

    if debug:
        frames_tmp = [final_frames[mm] for mm in range(nfr)]
        frames_tmp = tuple(frames_tmp)
        hp.plot_frames(frames_tmp, ang_scale=True,
                       ang_ticksep=int(0.2/plsc_com),
                       pxscale=plsc_com)


    # Comparison of pairs of frames
    cube1_rot = np.zeros([len(test_rot),final_frames[0].shape[0],final_frames[0].shape[1]])
    best_rots = np.zeros(ncomp)
    rots_err = best_rots.copy()
    for jj in range(ncomp):
        for tt, rot in enumerate(test_rot):
            cube1_rot[tt] = frame_rotate(final_frames[0],-rot-rotang[0], imlib='skimage')
            rcav_px = rmask/plsc_com
            cube1_rot[tt]= mask_circle(cube1_rot[tt], rcav_px)
            if rout is None:
                cube1_rot[tt][np.where(cube1_rot[tt]<thr1[0]*stddev[0])] = 0.
            else:
                rout_px = rout/plsc_com
                cube1_rot[tt] = mask_circle(cube1_rot[tt], rout_px, mode='out')
            if rsquare[0] == 1:
                ## Scale by r^2
                r_fr1 = dist_matrix(cube1_rot[tt].shape[0])
                cube1_rot[tt] = cube1_rot[tt]*np.power(r_fr1,2)
            elif rsquare[0] == 0.5:
                r_fr1 = dist_matrix(cube1_rot[tt].shape[0])
                cube1_rot[tt] = cube1_rot[tt]*r_fr1
            if rot == 0 and debug:
                if jj == 0:
                    frame_list_tmp = [cube1_rot[tt],final_frames[jj+1]]
                else:
                    frame_list_tmp.append(final_frames[jj+1])
                if jj==ncomp-1:
                    hp.plot_frames(tuple(frame_list_tmp), ang_scale=True,
                                   ang_ticksep=int(0.2/plsc_com), colorbar=False,
                                   pxscale=plsc_com, save=figname, label=fig_labels)
                    if fitsname is not None:
                        vip.fits.write_fits(fitsname,np.array(frame_list_tmp))
        ccor[jj] = cube_distance(cube1_rot, final_frames[jj+1], 'full', dist, plot=False)
        # fit to a 1D gaussian
        p0 = [np.amax(ccor[jj]), test_rot[np.argmax(ccor[jj])], np.amax(test_rot)-np.amin(test_rot)]
        try:
            coeff, var_matrix = curve_fit(_gauss, test_rot, ccor[jj], p0=p0)
            best_rots[jj] = coeff[1]
            rots_err[jj] = np.sqrt(np.diag(var_matrix))[1]
        except:
            pdb.set_trace()
            best_rots[:]  = np.nan
            rots_err[:] = np.nan
            ccor[:,:] = np.nan
            return best_rots, rots_err, ccor

    print("Best rotation angles with respect to first (+error): ", best_rots, rots_err, " deg")


    return best_rots, rots_err, ccor



def confidence(isamples, cfd=68.27, bins=100, gaussian_fit=False, weights=None,
               best_rot=None, verbose=True, save=False, output_dir='',
               output_file='confidence.txt', output_plot = 'confi_hist', pKey=['rot'],
               label=[r'$\Delta rot$'], **kwargs):
    """
    Determine the highly probable value for each model parameter, as well as
    the 1-sigma confidence interval.

    Parameters
    ----------
    isamples: numpy.array
        The independent samples for each model parameter.
    cfd: float, optional
        The confidence level given in percentage.
    bins: int, optional
        The number of bins used to sample the posterior distributions.
    gaussian_fit: boolean, optional
        If True, a gaussian fit is performed in order to determine (\mu,\sigma)
    weights : (n, ) numpy ndarray or None, optional
        An array of weights for each sample.
    verbose: boolean, optional
        Display information in the shell.
    save: boolean, optional
        If "True", a txt file with the results is saved in the output
        repository.
    kwargs: optional
        Additional attributes are passed to the matplotlib hist() method.

    Returns
    -------
    out: tuple
        A 2 elements tuple with the highly probable solution and the confidence
        interval.

    """
    colors = ['r','b','y','c','m','g']
    plsc = kwargs.pop('plsc', 0.001)
    title = kwargs.pop('title', None)

    #output_file = kwargs.pop('filename', 'confidence.txt')

    try:
        l = isamples.shape[1]
    except:
        l = 1

    confidenceInterval = {}
    val_max = {}


    if cfd == 100:
        cfd = 99.9

    #########################################
    ##  Determine the confidence interval  ##
    #########################################
    if gaussian_fit:
        mu = np.zeros(l)
        sigma = np.zeros_like(mu)

    fig, ax = plt.subplots(1, l, figsize=(12,4))

    for j in range(l):
        label_file = pKey #['r', 'theta', 'flux']
        #label = [r'$\Delta r$', r'$\Delta \theta$', r'$\Delta f$']

        n, bin_vertices, _ = ax[j].hist(isamples[:,j], bins=bins,
                                               weights=weights, histtype='step',
                                               edgecolor='gray')
        bins_width = np.mean(np.diff(bin_vertices))
        surface_total = np.sum(np.ones_like(n)*bins_width * n)
        n_arg_sort = np.argsort(n)[::-1]

        test = 0
        pourcentage = 0
        for k, jj in enumerate(n_arg_sort):
            test = test + bins_width*n[int(jj)]
            pourcentage = test/surface_total*100
            if pourcentage > cfd:
                if verbose:
                    msg = 'percentage for {}: {}%'
                    print(msg.format(label_file[j], pourcentage))
                break
        n_arg_min = int(n_arg_sort[:k].min())
        n_arg_max = int(n_arg_sort[:k+1].max())

        if n_arg_min == 0:
            n_arg_min += 1
        if n_arg_max == bins:
            n_arg_max -= 1

        val_max[pKey[j]] = bin_vertices[int(n_arg_sort[0])]+bins_width/2.
        confidenceInterval[pKey[j]] = np.array([bin_vertices[n_arg_min-1],
                                                bin_vertices[n_arg_max+1]]
                                                - val_max[pKey[j]])

        arg = (isamples[:, j] >= bin_vertices[n_arg_min - 1]) * \
              (isamples[:, j] <= bin_vertices[n_arg_max + 1])
        if gaussian_fit:
            ax[j].hist(isamples[arg,j], bins=bin_vertices,
                          facecolor='gray', edgecolor='darkgray',
                          histtype='stepfilled', alpha=0.5)
            if best_rot is not None:
                ax[j].vlines(best_rot[j], 0, n[int(n_arg_sort[0])],
                                linestyles='dashed', color=colors[j])
            else:
                ax[j].vlines(val_max[pKey[j]], 0, n[int(n_arg_sort[0])],
                                linestyles='dashed', color=colors[j])
            ax[j].set_xlabel(label[j])
            if j == 0:
                ax[j].set_ylabel('Counts')

            mu[j], sigma[j] = norm.fit(isamples[:, j])
            n_fit, bins_fit = np.histogram(isamples[:, j], bins, normed=1,
                                           weights=weights)
            y = norm.pdf(bins_fit, mu[j], sigma[j])
            y = (n[int(n_arg_sort[0])]/np.amax(y))*y
            ax[j].plot(bins_fit, y, colors[j]+'-', linewidth=2, alpha=0.7)


            if title is not None:
                msg = r"{}   $\mu$ = {:.4f}, $\sigma$ = {:.4f}"
                ax[j].set_title(msg.format(title, mu[j], sigma[j]),
                                   fontsize=10)

        else:
            ax[j].hist(isamples[arg,j],bins=bin_vertices, facecolor='gray',
                       edgecolor='darkgray', histtype='stepfilled',
                       alpha=0.5)
            ax[j].vlines(val_max[pKey[j]], 0, n[int(n_arg_sort[0])],
                         linestyles='dashed', color='red')
            ax[j].set_xlabel(label[j])
            if j == 0:
                ax[j].set_ylabel('Counts')

            if title is not None:
                msg = r"{} - {:.3f} {:.3f} +{:.3f}"
                ax[j].set_title(msg.format(title, val_max[pKey[j]],
                                           confidenceInterval[pKey[j]][0],
                                           confidenceInterval[pKey[j]][1]),
                                fontsize=10)

        plt.tight_layout(w_pad=0.1)

    if save:
        if gaussian_fit:
            plt.savefig(output_dir+output_plot+'_gaussfit.pdf')
        else:
            plt.savefig(output_dir+output_plot+'.pdf')

    if verbose:
        print('\n\nConfidence intervals:')
        for j in range(l):
            print('{}: {} [{},{}]'.format(pKey[j],val_max['r'],
                                         confidenceInterval[pKey[j]][0],
                                         confidenceInterval[pKey[j]][1]))
            if gaussian_fit:
                print()
                print('Gaussian fit results:')
                print('{}: {} +-{}'.format(pKey[j], mu[j], sigma[j]))

    ##############################################
    ##  Write inference results in a text file  ##
    ##############################################
    if save:
        with open(output_dir+output_file, "w") as f:
            f.write('###########################\n')
            f.write('####   INFERENCE TEST   ###\n')
            f.write('###########################\n')
            f.write(' \n')
            f.write('Results of the MCMC fit\n')
            f.write('----------------------- \n')
            f.write(' \n')
            f.write('>> Position and flux of the planet (highly probable):\n')
            f.write('{} % confidence interval\n'.format(cfd))
            f.write(' \n')

            for i in range(l):
                confidenceMax = confidenceInterval[pKey[i]][1]
                confidenceMin = -confidenceInterval[pKey[i]][0]
                if i == 2:
                    text = '{}: \t\t\t{:.3f} \t-{:.3f} \t+{:.3f}\n'
                else:
                    text = '{}: \t\t\t{:.3f} \t\t-{:.3f} \t\t+{:.3f}\n'

                f.write(text.format(pKey[i], val_max[pKey[i]],
                                    confidenceMin, confidenceMax))

            f.write(' ')
            f.write('Platescale = {} mas\n'.format(plsc*1000))
            f.write('r (mas): \t\t{:.2f} \t\t-{:.2f} \t\t+{:.2f}\n'.format(
                        val_max[pKey[0]]*plsc*1000,
                        -confidenceInterval[pKey[0]][0]*plsc*1000,
                        confidenceInterval[pKey[0]][1]*plsc*1000))

    if gaussian_fit:
        return mu, sigma
    else:
        return val_max, confidenceInterval
