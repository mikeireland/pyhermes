"""This module contains the HERMES data reduction class.
"""

#Example setup analysis for all channels:
#import hermes
#hermes.go_all('/Users/mireland/data/hermes/140310/data/', '/Users/mireland/tel/hermes/140310/', '/Users/mireland/python/pyhermes/cal/')

#Example setup analysis for a full night: blue        
#hm = hermes.HERMES('/Users/mireland/data/hermes/140310/data/ccd_1/', '/Users/mireland/tel/hermes/140310/ccd_1/', '/Users/mireland/python/pyhermes/cal/ccd_1/')

#Example setup analysis for a full night: green.
#hm = hermes.HERMES('/Users/mireland/data/hermes/140310/data/ccd_2/', '/Users/mireland/tel/hermes/140310/ccd_2/', '/Users/mireland/python/pyhermes/cal/ccd_2/')

#Example setup analysis for a full night: red.
#hm = hermes.HERMES('/Users/mireland/data/hermes/140310/data/ccd_3/', '/Users/mireland/tel/hermes/140310/ccd_3/', '/Users/mireland/python/pyhermes/cal/ccd_3/')

#Example setup analysis for a full night: ir.
#hm = hermes.HERMES('/Users/mireland/data/hermes/140310/data/ccd_4/', '/Users/mireland/tel/hermes/140310/ccd_4/', '/Users/mireland/python/pyhermes/cal/ccd_4/')

#Then go!
#hm.go()

from __future__ import print_function, division
try: 
    import pyfits
except:
    import astropy.io.fits as pyfits
try:
    from PyAstronomy import pyasl
    barycorr = True
except:
    print("WARNING: PyAstronomy is required for barycentric corrections, or combining multiple epochs.")
    barycorr = False
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as nd
import matplotlib.cm as cm
import time
import glob
import os
import threading
from multiprocessing import Process
import pdb
import sys
    
class HERMES():
    """The HERMES Class. It must always be initiated with
    a data, reduction and calibration directory. 
    
    Parameters
    ----------
    ddir: string
        Raw data directory
        
    rdir: string
        Reduction directory
        
    cdir: string
        
    gdir: string (optional)
        GALAH collaboration standard output directory. If not given, the rdir is used.
    """
    def __init__(self, ddir, rdir, cdir, gdir=''):
        self.ddir = ddir
        self.rdir = rdir
        self.cdir = cdir
        if gdir=='':
            self.gdir = rdir
        else:
            self.gdir = gdir
        #Each release should increment this number.
        self.release = 0.1
        #A dictionary of central wavelengths (can be changed)
        #This comes from the SPECTID
        self.fixed_wave0 = {'BL':4700.0,'GN':5630.0,'RD':6460.0,'RR':7570.0}
        self.ccd_nums = {'BL':'1','GN':'2','RD':'3','RR':'4'}
        self.fixed_R = 200000
        self.fixed_nwave = 9000
        
    def basic_process(self, infile):
        """Read in the file, correct the over-scan region (and do anything else
        that can be done prior to bias subtraction (e.g. removal of obscure
        readout artifacts/pattern noise could go here as an option)
        
        Parameters
        ----------
        infile:    The input filename (no directory)
    
        Returns
        -------
        d: (ny,nx) array
        """
        d=pyfits.getdata(self.ddir + infile)
        header=pyfits.getheader(self.ddir + infile)
        overscan_mns = np.mean(d[:,header['WINDOXE1']:],axis=1)
        d = d[:,:header['WINDOXE1']]
        for i in range(d.shape[0]):
            d[i,:] -= overscan_mns[i]
        return d
        
    def find_badpix(self, infiles, var_threshold=10.0, med_threshold=10.0):
        """Find the bad pixels from a set of dark files. All we care about are pixels
        that vary a lot or are very hot. We will give pixels the benefit of the doubt if they are 
        only bright once (e.g. cosmic ray during readout...)
        
        Parameters
        ----------
        infiles: An array of input filenames
        
        var_threshold: float (optional)
            A pixel has to have a variance more than var_threshold times the median to be
            considered bad (NB the frame with largest flux is removed in this calculation,
            in case of cosmic rays)
        med_threshold: float (optional)
            A pixel has to have a value more than med_threshold above the median to be
            considered bad. This should be particularly relevant for finding hot pixels
            in darks.
        
        Returns
        -------
        badpix: float array
            The 2D image which is 0 for good pixels and 1 for bad pixels.
        """
        header = pyfits.getheader(self.ddir + infiles[0])
        nx = header['WINDOXE1']
        ny = header['NAXIS2']
        nf = len(infiles)
        if nf < 4:
            print("ERROR: At least 4 files needed to find bad pixels")
            raise UserWarning
        cube = np.zeros((nf,ny,nx),dtype=np.uint8)
        for i in range(nf):
            cube[i,:,:] = self.basic_process(infiles[i])
        medim = np.median(cube, axis=0)
        varcube = np.zeros((ny,nx))
        for i in range(nf):
            cube[i,:,:] -= medim
            varcube += cube[i,:,:]**2
        maxcube = np.amax(cube,axis=0)
        varcube -= maxcube**2
        varcube /= (nf-2)
        medvar = np.median(varcube)
        medval = np.median(medim)
        medsig = np.sqrt(medvar)
        print("Median pixel standard deviation: " + str(medsig))
        ww = np.where( (varcube > var_threshold*medvar) * (medim > med_threshold*medsig + medval) )
        print(str(len(ww[0])) + " bad pixels identified.")
        badpix = np.zeros((ny,nx),dtype=np.uint8)
        badpix[ww]=1
        for i in range(nf):
            header['HISTORY'] = 'Input: ' + infiles[i]
        hl = pyfits.HDUList()
        hl.append(pyfits.ImageHDU(badpix,header))
        hl.writeto(self.rdir+'badpix.fits',clobber=True)
        return badpix
        
    def median_combine(self, infiles, outfile):
        """Median combine a set of files. Most useful for creating a master bias. 
        
        Parameters
        ----------
        infiles: string array
            Input files
            
        outfile: string
            The output file (goes in the reduction directory rdir)
        
        Returns
        -------
        image: float array
            The median combined image.
        """
        header = pyfits.getheader(self.ddir + infiles[0])
        nx = header['WINDOXE1']
        ny = header['NAXIS2']
        nf = len(infiles)
        cube = np.zeros((nf,ny,nx))
        for i in range(nf):
            cube[i,:,:] = self.basic_process(infiles[i])
        medcube = np.median(cube, axis=0)
        for i in range(nf):
            header['HISTORY'] = 'Input: ' + infiles[i]
        hl = pyfits.HDUList()
        hl.append(pyfits.ImageHDU(medcube.astype('f4'),header))
        hl.writeto(self.rdir+outfile,clobber=True)
        return medcube
        
    def clobber_cosmic(self, image, threshold=3.0):
        """Remove cosmic rays and reset the pixel values to something sensible.
        As a general rule, cosmic rays should be flagged rather than clobbered,
        but this routine is here as a placeholder."""
        smoothim = nd.filters.median_filter(image,size=5)
        ww = np.where(image > smoothim*threshold)
        image[ww] = smoothim[ww]
        return image
                
    def make_cube_and_bad(self,infiles, badpix=[], threshold=6.0, mad_smooth=64):
        """Based on at least 2 input files, find outlying bright pixels and flag them
        as bad.
        
        Parameters
        ----------
        
        infiles: string array
            The array of input files to be cubed and have their bad pixels flagged.
        
        mad_smooth: int, optional
            The distance in the x-direction that the median absolute deviation (MAD)
            is smoothed over in order to determine image statistics
        
        threshold: float, optional
            The threshold in units of standard deviation to identify bad pixels.
        
        Notes
        -----
        This routine has 1/3 its time up to the reference_im and 1/3 the time
        in the loop beyond, on several lines. So tricky to optimise.
        """
        if len(infiles) < 2:
            print("Error: make_cube_and_bad needs at least 2 input files")
            raise UserWarning
        if len(badpix)==0:
            badpix = pyfits.getdata(self.cdir + 'badpix.fits')
        header = pyfits.getheader(self.ddir + infiles[0])
        szy = header['NAXIS2']
        szx = header['WINDOXE1']
        if (szx % mad_smooth != 0):
            print("ERROR: x axis length must be divisible by mad_smooth")
            raise UserWarning
        cube = np.empty((len(infiles),szy,szx))
        normalised_cube = np.empty((len(infiles),szy,szx))
        bad = np.empty((len(infiles),szy,szx), dtype=np.uint8)
        for i,infile in enumerate(infiles):
            im = self.basic_process(infile)
            cube[i,:,:]=im
            normalised_cube[i,:,:]= im/np.median(im)
            bad[i,:,:] = badpix
        # Create a mean image that ignores the maximum pixel values over all images (i.e. cosmic rays)
        reference_im = (np.sum(normalised_cube,axis=0) - np.max(normalised_cube,axis=0))/(len(infiles) - 1.0)
        szy = cube.shape[1]
        # Look for bad pixels (hot or cosmic rays) by empirically finding pixels that deviate
        # unusually from the minimum image.
        for i in range(len(infiles)):
            diff = normalised_cube[i,:,:] - reference_im
            #Entire rows can't be used for a row_deviation, as tramlines curve. But
            #a pretty large section can be used. Unfortunately, a straight median filter
            #on a large section is slow...
            row_deviation = np.abs(diff).reshape( (szy,szx//mad_smooth, mad_smooth) )
            row_deviation = np.repeat(np.median(row_deviation, axis=2),mad_smooth).reshape( (szy,szx) )
            tic = time.time()
#Slow line... even with only 21 pixels.
#            row_deviation = nd.filters.median_filter(np.abs(diff),size=[1,21])
            ww = np.where(diff > 1.4826*threshold*row_deviation)
            bad[i,ww[0],ww[1]] = 1
#Another slow alternative...
#             row_deviation = np.median(np.abs(diff),axis=1)
#             for j in range(szy):
#                 ww = np.where(diff[j,:] > 1.4826*threshold*row_deviation[j])[0]
#                 if len(ww)>0:
#                     bad[i,j,ww]=1
#                     #for w in ww:
                        
        return cube,bad
        
    def make_psf(self,npix=15, oversamp=3, fibre_radius=2.5, optics_psf_fwhm=1.0):
        """Make a 1-dimensional collapsed PSF provile based on the physics of image 
        formation. This does not include effects of variable magnification across the chip...
        (i.e. it applies to a Littrow image).  
        
        Parameters
        ----------
        npix: int, optional
            The number of pixels to create the PSF profile. Must be odd.
        
        oversamp: int, optional
            The oversampling factor for the PSF. In order for interpolation to work
            reasonably well, 3 is a default.
            
        fibre_radius: float, optional
            The radius of the fiber in pixels. 
            
        optics_psf_fwhm: float, optional
            The FWHM of a Gaussian approximation to the optical aberration function.
        
        Returns
        -------
        A point-spread function normalised to the maximum value.
        
        Notes
        -----
        This is far from complete: the PSF is variable etc, but it is a good
        approximation
        """
        x = ( np.arange(npix*oversamp) - npix*oversamp//2 )/float(oversamp)
        psf = np.sqrt( np.maximum( fibre_radius**2 - x**2 ,0) )
        g = np.exp(-x**2/2/(optics_psf_fwhm/2.35482)**2 )
        psf = np.convolve(psf,g, mode='same')
        psf = np.convolve(psf,np.ones(oversamp), mode='same')
        return psf/np.max(psf)
        
    def extract(self,infiles,cube=[],flux_on_ron_var=10.0,fix_badpix=True,badpix=[]):
        """Extract spectra from an image or a cube. Note that we do *not* apply a fibre flat
        at this point, because that can only done after (optional) cross-talk and scattered-light
        correction.
        
        Parameters
        ----------
        infiles: string array
            An array of input filenames
            
        cube: float array, optional
            The data, coming from a cleaned version of the input files.
            
        flux_on_ron_var: float, optional
            The target pixel flux divided by the readout noise variance. This determines
            the (fixed) extraction profile - extraction is only optimal for this value
            of signal-to-noise. It should be set to the minimum useable signal level.
            
        fix_badpix: bool, optional
            Does this routine attempt to fix bad pixels?
            
        badpix: float array, optional
            A *cube* of bad pixels (including a flag of where cosmic rays are in
            each frame.
        
        Returns
        -------
        (flux,sigma): (float array, float array)
            The extracted flux, and the standard deviation of the extracted flux.
        """
        if fix_badpix:
            if len(badpix)==0:
                badpix = pyfits.getdata(self.cdir + 'badpix.fits')
        return_im=False
        if len(cube)==0:
            cube = []
            for infile in infiles:
                cube.append(self.basic_process(infile))
            cube = np.array(cube)
            if len(infiles)==1:
                return_im=True
        header = pyfits.getheader(self.ddir + infiles[0])
        ftable = pyfits.getdata(self.ddir + infiles[0],1)    
        if len(cube.shape) == 2:
            return_im=True
            cube = cube[None,:]
        if len(badpix.shape) == 2:
            badpix = badpix[None,:]
        #Now create the extraction subimages.
        oversamp = 3
        nslitlets=40
        nfibres=10
        npix_extract=15
        nx = cube.shape[2]
        psf = self.make_psf(npix=npix_extract, oversamp=oversamp)
        #The following indices have a 0 for the middle index (
        y_ix_oversamp = np.arange(oversamp*npix_extract) - oversamp*npix_extract//2
        y_ix = ( np.arange(npix_extract) - npix_extract//2 )*oversamp
        #The extracted flux
        extracted_flux = np.zeros((cube.shape[0],nfibres*nslitlets,nx))
        extracted_sigma = np.zeros((cube.shape[0],nfibres*nslitlets,nx))
        #Much of this is copied from fit_tramlines - so could be streamlined !!!
        try:
            p_tramline = np.loadtxt(self.rdir + 'tramlines_p' + header['SOURCE'][6] + '.txt')
        except:
            print("No tramline file in reduction directory! Using default from calibration directory")
            p_tramline = np.loadtxt(self.cdir + 'tramlines_p' + header['SOURCE'][6] + '.txt')
        #Make a matrix that maps p_tramline numbers to dy
        tramline_matrix = np.zeros((nfibres,nx,4))
        for i in range(nfibres):
            tramline_matrix[i,:,0] = np.arange(nx)**2
            tramline_matrix[i,:,1] = np.arange(nx)
            tramline_matrix[i,:,2] = np.ones( nx )
        for k in range(nx):
            tramline_matrix[:,k,3] = np.arange(nfibres)+0.5-nfibres//2
        psfim = np.zeros((npix_extract,nx))
        psfim_yix = np.repeat(np.arange(npix_extract)*oversamp + oversamp//2,nx).reshape((npix_extract,nx))
        print("Beginning extraction...")
        for i in range(nslitlets):
            ypix = np.dot(tramline_matrix, p_tramline[i,:])
            ypix_int = np.mean(ypix,axis=1).astype(int)
            ypix_int = np.maximum(ypix_int,npix_extract//2)
            ypix_int = np.minimum(ypix_int,cube.shape[1]-npix_extract//2)
            for j in range(nfibres):
                #This image has an odd number of pixels. Lets extract in units of electrons, not DN.
                subims = cube[:,ypix_int[j] - npix_extract//2:ypix_int[j] + npix_extract//2 + 1,:]*header['RO_GAIN']
                subbad = badpix[:,ypix_int[j] - npix_extract//2:ypix_int[j] + npix_extract//2 + 1,:]
                #Start off with a slow interpolation for simplicity. Now removed...
                #for k in range(nx):
                #    psfim[:,k]  = np.interp(y_ix - (ypix[j,k] - ypix_int[j])*oversamp, y_ix_oversamp, psf)
                #A fast interpolation... A centered PSF will have ypix=ypix_int 
                ypix_diff_oversamp = -(ypix[j,:] - ypix_int[j])*oversamp
                ypix_diff_int = np.floor(ypix_diff_oversamp)
                ypix_diff_rem = ypix_diff_oversamp - ypix_diff_int
                ix0 = (psfim_yix + np.tile(ypix_diff_int,npix_extract).reshape((npix_extract,nx))).astype(int)
                ix1 = ix0+1
                ix0 = np.maximum(ix0,0)
                ix0 = np.minimum(ix0,npix_extract*oversamp-1)
                ix1 = np.maximum(ix1,0)
                ix1 = np.minimum(ix1,npix_extract*oversamp-1)
                frac = np.tile(ypix_diff_rem,npix_extract).reshape((npix_extract,nx))
                psfim = psf[ix0]*(1 - frac) + psf[ix1]*frac
                #Now we turn the PSF into weights
                weights = flux_on_ron_var*psfim/(1 + flux_on_ron_var*psfim) 
                psfim /= np.sum(psfim,axis=0)
                for cube_ix in range( cube.shape[0] ):
                    good_weights = weights*(1-subbad[cube_ix,:,:])
                    #Normalise so that a flux of 1 with the shape of the model PSF will
                    #give an extracted flux of 1. If the sum of the weights is zero (e.g. a bad
                    #column, then this is a zero/zero error. A typical weight is of order unity.
                    #There will be some divide by zeros... which we'll fix later.
                    ww = np.where(np.sum(good_weights,axis=0)==0)[0]
                    with np.errstate(invalid='ignore'):
                        good_weights /= np.sum(good_weights*psfim,axis=0)
                    #Now do the extraction!    
                    extracted_flux[cube_ix,i*nfibres + j,:] = np.sum(good_weights*subims[cube_ix,:,:],axis=0)
                    extracted_sigma[cube_ix,i*nfibres + j,:] = np.sqrt(np.sum(good_weights**2*(subims[cube_ix,:,:] + header['RO_NOISE']**2), axis=0))
                    extracted_sigma[cube_ix,i*nfibres + j,ww]=np.inf
        print("Finished extraction...")
        #!!! TODO: We can check for bad pixels that were missed at this point, e.g.
        #bad columns that weren't very bad and didn't show up pre-extraction. If we really
        #want to have fun with this, smoothing by windowing the Fourier transform of
        #the data may work best.
        if return_im:
            return extracted_flux[0,:,:], extracted_sigma[0,:,:]
        else:
            return extracted_flux, extracted_sigma
    
    def create_fibre_flat(self, infile, smooth_threshold=1e3, smoothit=0, sigma_cut=5.0):
        """Based on a single flat, compute the fiber flat field, corrected for
        individual fibre throughputs. 
        
        NB the ghost images turn up in this also. 
        
        Parameters
        ----------
        infile: string
            The input filename
            
        smooth_threshold:
            If average counts are lower than this, we smooth the fibre flat 
            (no point dividing by noise).
            
        smoothit:
            The width of the smoothing filter.
            
        sigma_cut:
            The cut in standard deviations to look for bad extracted pixels, when compared
            to median-filtered extracted pixels.
        
        Returns
        -------
        fibre_flux: float (nfibres, nx) array
            The fiber flat field.
        """
        fibre_flux, fibre_sigma = self.extract([infile])
        if np.median(fibre_flux) < smooth_threshold:
            smoothit = 25
        #First, reject outliers in individual rows rather agressively
        #(remember the spectra are smooth)
        fibre_sigma = nd.filters.median_filter(fibre_sigma, size=(1,25))
        fibre_flux_medfilt = nd.filters.median_filter(fibre_flux, size=(1,25))
        ww = np.where(np.abs(fibre_flux - fibre_flux_medfilt) > sigma_cut*fibre_sigma)
        fibre_flux[ww] = fibre_flux_medfilt[ww]
        if smoothit>0:
            fibre_flux = np.convolve(fibre_flux,np.ones(smoothit)/smoothit,mode='same')
        #Find dead fibres.
        fib_table = pyfits.getdata(self.ddir + infile,1)
        #Check for an obscure bug, where the extension orders are changed...
        if len(fib_table)==1:
            fib_table = pyfits.getdata(self.ddir + infile,2)
        off_sky = np.where( (fib_table['TYPE'] != 'S') * (fib_table['TYPE'] != 'P'))[0]
        on_sky  = np.where( 1 - (fib_table['TYPE'] != 'S') * (fib_table['TYPE'] != 'P'))[0]
        med_fluxes = np.median(fibre_flux,axis=1)
        wbad = np.where(med_fluxes[on_sky] < 0.1*np.median(med_fluxes[on_sky]))[0]
        if len(wbad)>0:
            print("Bad fibres (<10% of median flux): " + str(wbad))
        #!!! Unsure what to do with this. At the moment the data will just look bad
        #and will be cut due to S/N later.

        #We always return a *normalised* fiber flux, so that we're at least close to
        #the raw data.
        return fibre_flux/np.median(fibre_flux[on_sky,:])
        
    def sky_subtract(self, infiles, extracted_flux,extracted_sigma, wavelengths, sigma_cut=5.0, fibre_flat=[]):
        """Subtract the sky from each extracted spectrum. This should be done after 
        cross-talk and scattered light removal, but is in itself a crude way to 
        remove scattered light. Note that all files input are assumed to have
        the same wavelength scale and fiber table.
        
        The fibre flat correction is also done at this time.
        
        Notes
        -----
        There appears to be real structure in the flat at the 0.5% level... but
        this has to be confirmed with multiple epoch tests. It is a little suspicious
        as the structure is more or less white.
        
        TODO: !!!
        1) Uncertainties in sky, based on input sigma and interpolating past the chip edge.
        2) Uncertainties in data, based on sky subtraction.
        3) bad pixels! """
        #Find the list of sky and object fibres.
        fib_table = pyfits.getdata(self.ddir + infiles[0],1)
        #Check for an obscure bug, where the extension orders are changed...
        if len(fib_table)==1:
            fib_table = pyfits.getdata(self.ddir + infiles[0],2)
        sky = np.where(fib_table['TYPE']=='S')[0]
        ns = len(sky)
        #Apply fibre flat.
        if len(fibre_flat)>0:
            sky_fibre_variance = 1.0/np.median(fibre_flat[sky,:],axis=1)**2
            for cube_ix in range(len(infiles)):
                    extracted_flux[cube_ix,:,:] /= fibre_flat
                    extracted_sigma[cube_ix,:,:] /= fibre_flat
        else:
            sky_fibre_variance = np.ones(ns)
        #Go through sky fibres one at a time and reconstruct their spectra from the other
        #sky fibres.
        nx = wavelengths.shape[1]
        nf = len(infiles)
        sky_flux = extracted_flux[:,sky,:]
        sky_sigma = extracted_sigma[:,sky,:]
        bad_skies = []
        #j is an index that runs from 0 to the number of sky fibers.
        #s is the actual fits file index of the fiber.
        for j,s in enumerate(sky):
            ww = sky[np.where(sky != s)[0]]
            sky_spectra_interp = np.zeros((nf,ns-1,nx))
            sky_sigma_interp = np.zeros((nf,ns-1,nx))
            for k, sky_other in enumerate(ww):
                #Another manual interpolation... Find the index corresponding to the 
                #wavelength of our target sky fiber. e.g. if pixel 10 for s correponds
                #to pixel 11 for sky_other, we want ix=11
                ix = np.interp(wavelengths[s,:], wavelengths[sky_other,:],np.arange(nx))
                #Divide this into integer and fractional parts.
                ix_int = np.floor(ix).astype(int)
                #!!! This currently has edge effects !!!
                ix_int = np.maximum(ix_int,0)
                ix_int = np.minimum(ix_int,nx-2)
                ix_frac = ix - ix_int
                for i in range(nf):
                    sky_spectra_interp[i,k,:] = extracted_flux[i,sky_other,ix_int]*(1-ix_frac) + extracted_flux[i,sky_other,ix_int+1]*ix_frac
                    sky_sigma_interp[i,k,:] = extracted_sigma[i,sky_other,ix_int]*(1-ix_frac) + extracted_sigma[i,sky_other,ix_int+1]*ix_frac
            sky_spectra_recon = np.median(sky_spectra_interp, axis=1)
            sky_sigma_recon = np.median(sky_sigma_interp, axis=1)
            #Now find outliers and correct them. It is important that the sky_sigma is also a robust statistic.
            #... for this to work for nan values... and gradients (up to a factor of 2 for scattered light) to be fine.
            #ww = np.where(np.abs(extracted_flux[:,s,:] - sky_spectra_recon) > sigma_cut*sky_sigma_recon)
            #The debugger lines below show that there is still some work to do !!!
            for i in range(nf):
                scaling_factor = np.median(extracted_flux[i,s,:]/sky_spectra_recon[i,:])
                if np.abs(np.log(scaling_factor)) > np.log(2):
                    print("Unusual sky fiber! Scaling factor required: " + str(scaling_factor))
                    bad_skies.append(j)
#For testing                    import pdb; pdb.set_trace()
                ww = np.where(np.logical_not(np.abs(extracted_flux[i,s,:] - scaling_factor*sky_spectra_recon[i,:]) < scaling_factor*sigma_cut*sky_sigma_recon[i,:]))[0]
                #Look for invalid values... 
                if len(ww) > 400:
                    print("Crazy number of bad pixels in reconstructing sky!")
                    bad_skies.append(j)
#For testing                    import pdb; pdb.set_trace()
                sky_flux[i,j,ww] = sky_spectra_recon[i,ww]*scaling_factor
                sky_sigma[i,j,ww] = sky_sigma_recon[i,ww]*scaling_factor
        #If we've flagged a bad sky fiber, remove it now.
        good_skies = np.arange(ns)
        for abad in bad_skies:
            good_skies = good_skies[np.where(good_skies != abad)[0]]
        sky = sky[good_skies]
        sky_flux = sky_flux[:,good_skies,:]
        sky_sigma = sky_sigma[:,good_skies,:]
        sky_fibre_variance = sky_fibre_variance[good_skies]
        ns = len(sky)
        #Now do the same for the extracted object fibers... except in this case we use a weighted average.
        #Include sky fibers as "objects" as a sanity check.
        objects = np.where(np.logical_or(fib_table['TYPE']=='P',fib_table['TYPE']=='S'))[0]
        for o in objects:
            #Find the dx and dy values for the positioner.
            dx_pos = fib_table['X'][sky] - fib_table['X'][o]
            dy_pos = fib_table['Y'][sky] - fib_table['Y'][o]
            #Create the quadratic programming problem.
            #http://en.wikipedia.org/wiki/Quadratic_programming
            #Start with equality constraints...
            E = np.array([dx_pos,dy_pos,np.ones(ns)])
            c_d = np.zeros( ns+3 )
            c_d[-1] = 1.0
            the_matrix = np.zeros( (ns+3, ns+3) )
            ix = np.arange(ns)
            #Weight by inverse fiber throughput squared - appropriate for
            #readout-noise limited sky data, typical of HERMES.
            the_matrix[ix,ix] = sky_fibre_variance
            the_matrix[ns:,0:ns] = E
            the_matrix[0:ns,ns:] = np.transpose(E)
            x_lambda = np.linalg.solve(the_matrix, c_d)
            weights = x_lambda[0:ns]
            old_flux = extracted_flux.copy()
            #Lets save the weighted average sky separately... great for
            #bug-shooting.
            sky_to_subtract = np.zeros( (nf,nx) )
            for k,s in enumerate(sky):
                #Interpolate the wavelength scale for each fiber, subtracting off the interpolated
                #sky fiber flux multiplied by our pre-computed weights.
                #!!! This currently has edge effects !!! 
                #!!! And is copied from above... should be its own routine once uncertainties are sorted !!!
                ix = np.interp(wavelengths[o,:], wavelengths[s,:],np.arange(nx))
                ix_int = np.floor(ix).astype(int)
                ix_int = np.maximum(ix_int,0)
                ix_int = np.minimum(ix_int,nx-2)
                ix_frac = ix - ix_int
                for i in range(nf):
                    sky_to_subtract[i,:] += weights[k]*(sky_flux[i,k,ix_int]*(1-ix_frac) + sky_flux[i,k,ix_int+1]*ix_frac )
            #Now subtract the sky!
            extracted_flux[:,o,:] -= sky_to_subtract
        return extracted_flux, extracted_sigma
    
    def save_extracted(self, infiles, extracted_flux,extracted_sigma, wavelengths):
        """Save extracted spectra from a set of input files to individual
        files, labelled similarly to 2dFDR
        
        NOT IMPLEMENTED YET (is this a separate "manual save" routine?) """
        raise UserWarning
        
    def combine_multi_epoch_spectra(self,coord_limit=-1,search_galahic=False, fractional_snr_limit=0.3):
        """An out a flux-weighted epoch for the observation is placed in the header.
        
        Parameters
        ----------
        coord_limit: float
            Difference in coordinates (arcsec) for two objects to be considered the same.
            NOT IMPLELENTED
        search_galahic: boolearn
            Do we search non-galahic stars to see if they are galahic stars with a different
            label?
        fractional_snr_limit: float
            What fraction of the peak SNR does a new epoch need in order to be combined.
        """
        return []
    
    def make_comb_filename(self, outfile):
        """Create filename by inserting "comb" in-between the name of
        the file and ".fits
        """
        spos = outfile.find('.fits')
        if spos < 0:
            print("Weird file name... no fits extension")
            raise UserWarning
        return outfile[:spos] + 'comb' + outfile[spos:]
    
    def combine_single_epoch_spectra(self, extracted_flux, extracted_sigma, wavelengths, infiles=[], \
            csvfile='observation_table.csv', is_std=False):
        """ If spectra were taken in a single night with a single arc, we can approximate 
        the radial velocity as constant between frames, and combine the spectra in
        pixel space. 
        
        Parameters
        ----------
        extracted_flux: (nfiles,nfibres*nslitlets,nx) array
            Flux extracted for a set of files.
            
        extracted_sigma: (nfiles,nfibres*nslitlets,nx) array
            Standard deviations extracted from a set of files
            
        wavelengths: (nfibres*nslitlets,nx) array
            Common wavelength array.
            
        infiles: string list (optional)
            If given, the combined spectrum is output to a combined fits file that 
            contains the combined flux and the combined single epoch spectra.
            
        csvfile: string
            If given, key parameters are appended to the observations table. 
            NB At this point, no checking is done to see if the software runs multiple times,
            just appending to previous files.
            
        Returns
        -------
        flux_comb, flux_comb_sigma:  ((nfibres*nslitlets,nx) array, (nfibres*nslitlets,nx) array)
            Combined flux and combined standard deviation.
        """
        #The combination is simply a weighted arithmetic mean, as we already
        #have the variance as an input parameter.
        weights = 1/extracted_sigma**2
        ww = np.where(extracted_flux != extracted_flux)
        extracted_flux[ww] = 0.0
        flux_comb = np.sum(weights*extracted_flux,0)/np.sum(weights,0)
        extracted_flux[ww] = np.nan
        flux_comb_sigma = np.sqrt(1.0/np.sum(weights,0))
        #Save the data if necessary.
        if len(infiles)>0:
            headers = []
            for infile in infiles:
                headers.append(pyfits.getheader(self.ddir + infile))
            runs = [aheader['RUN'] for aheader in headers]
            start = np.argmin(runs)
            end = np.argmax(runs)
            header = headers[start]

            #To keep a record of which files went in to this.
            header['RUNLIST'] = str(runs)[1:-1]
            header['NRUNS'] = len(infiles)

            #Start and end
            header['HAEND'] = headers[end]['HAEND']
            header['ZDEND'] = headers[end]['ZDEND']
            header['UTEND'] = headers[end]['UTEND']
            header['STEND'] = headers[end]['STEND']
            header['HASTART'] = headers[start]['HASTART']
            header['ZDSTART'] = headers[start]['ZDSTART']
            header['UTSTART'] = headers[start]['UTSTART']
            header['STSTART'] = headers[start]['STSTART']

            #Means. If more accuracy than this is needed, then individual
            #(i.e. non-combined) files should be used!
            header['EPOCH'] = np.mean([aheader['EPOCH'] for aheader in headers])
            header['UTMJD'] = np.mean([aheader['UTMJD'] for aheader in headers])
            
            #For GALAH, we need to create a special directory. #Ly changed to accomodate non-standard cfg files
            cfg = header['CFG_FILE']
            if 'gf' in cfg:
                ix0 = header['CFG_FILE'].find('_')
                ix1 = header['CFG_FILE'].rfind('_')
                field_directory = header['CFG_FILE'][ix0+1:ix1]
            else:
                field_directory = cfg.replace('.sds','')
            if not os.path.exists(self.gdir + field_directory):
                os.makedirs(self.gdir + field_directory)
            
            #The header and fiber table of the first input file is retained, with
            #key parameters averaged from each header 
            #The first fits image (zeroth extension) is the combined flux.
            #The first fits extension is the fiber table
            #The second fits extension is the uncertainty in the combined flux.
            #The third fits extension is the wavelength scale.
            hl = pyfits.HDUList()
            hl.append(pyfits.ImageHDU(flux_comb.astype('f4'),header))
            fib_table = pyfits.getdata(self.ddir + infiles[start],1)
            #Check for an obscure bug, where the extension orders are changed...
            if len(fib_table)==1:
                fib_table = pyfits.getdata(self.ddir + infiles[start],2)
                hl.append(pyfits.open(self.ddir + infiles[start])[2])
            else:
                hl.append(pyfits.open(self.ddir + infiles[start])[1])
            hl.append(pyfits.ImageHDU(flux_comb_sigma.astype('f4')))

            if barycorr:
                logwave_flux_hdu, logwave_sigma_hdu, linwave_flux, linwave_sigma, wave_new, bcorr = \
                    self.create_barycentric_spectra(header, fib_table, flux_comb, flux_comb_sigma, wavelengths)
                hl.append(pyfits.ImageHDU(wave_new))          
                #Add the extra log-wavelength extensions that Mike seems to like so much. 
                hl.append(logwave_flux_hdu)
                hl.append(logwave_sigma_hdu)
                header['BCORR']='True'
            else:
                hl.append(pyfits.ImageHDU(wavelengths))
                header['BCORR']='False'
                
            #Lets always name the file by the first file in the set.
            outfile = self.make_comb_filename(infiles[start]) 
            hl.writeto(self.rdir + outfile,clobber=True)
            objects = np.where(fib_table['TYPE']=='P')[0]
            #See if we need to write a csv file header line...
            if not os.path.isfile(self.rdir + csvfile):
                f_csvfile = open(self.rdir + csvfile, 'a')
                f_csvfile.write('obsdate, run_start, run_end, fib_num, galahic_num, idname, snr, software,file, rdate\n')
                f_csvfile.close()
            f_csvfile = open(self.rdir + csvfile, 'a') 
            data_date = header['UTDATE'][2:4] + header['UTDATE'][5:7] + header['UTDATE'][8:10]
            now = time.gmtime()
            analysis_date = '{0:02d}{1:02d}{2:02d}'.format(now.tm_year-2000, now.tm_mon, now.tm_mday)
            if is_std:
                o = np.where(fib_table['PIVOT'] == (header['STD_FIB']))[0][0]
                medsnrs = np.median(flux_comb/flux_comb_sigma, axis=1)
                if np.argmax(medsnrs) != o:
                    print("Something dodgy with this standard fiber! Please check manually here...")
                    pdb.set_trace()
				#Ly - code copied from Mike's below to output individual standard star spectrum:
				#File name = standardstarname_ccd.fits
				#WG4 to rename/work around to suit their codes
                filename = "{0}_{1}.fits".format(header['STD_NAME'].replace(' ',''),self.ccd_nums[header['SPECTID']])
                flux_hdu = pyfits.ImageHDU(linwave_flux.data[o,:].astype('f4'),header)
                sig_hdu  = pyfits.ImageHDU(linwave_sigma.data[o,:].astype('f4'))
                #Add in header stuff from fiber table 
                #TODO !!!Add real RA and DEC to standard observations (not included in raw data), and reduce them separately later.              
                flux_hdu.header['RA'] = header['MEANRA']
                flux_hdu.header['DEC'] = header['MEANDEC']
                flux_hdu.header['V_BARY'] = bcorr[o]
                flux_hdu.header['FIBRE'] = o + 1
                for key in ("CRVAL1", "CDELT1", "CRPIX1", "CTYPE1", "CUNIT1"):
                    flux_hdu.header[key] = linwave_flux.header[key]
                    sig_hdu.header[key] = linwave_sigma.header[key]
                hl = pyfits.HDUList()
                hl.append(flux_hdu)
                hl.append(sig_hdu)
                hl.writeto(self.gdir + field_directory +'/' + filename, clobber=True)
                f_csvfile.write('{0:s},{1:d},{2:d},{3:d},{4:d},{5:s},{6:6.1f},{7:5.2f},{8:s},{9:s}\n'.format(
                    data_date,runs[start], runs[end], o+1, -1, header['STD_NAME'].replace(' ',''),
                    np.median(flux_comb[o,:]/flux_comb_sigma[o,:]), self.release,outfile,analysis_date))
            else:
              for o in objects:
                strpos = fib_table[o]['NAME'].find('galahic_')
                if strpos >= 0:
                    try:
                        galahic = int(fib_table[o]['NAME'][strpos+8:])
                    except:
                        galahic=-1 
                else:
                    galahic=-1
                #date, minimum file number, maximum file number, fiber number, 
                #input catalog number, input catalog name, signal-to-noise,
                #software release version, output file, analysis date
                #NB: The detector isn't here... a separate program has to take all these files
                #and add those details.     
                f_csvfile.write('{0:s},{1:d},{2:d},{3:d},{4:d},{5:s},{6:6.1f},{7:5.2f},{8:s},{9:s}\n'.format(
                    data_date,runs[start], runs[end], o+1, galahic, fib_table[o]['NAME'],
                    np.median(flux_comb[o,:]/flux_comb_sigma[o,:]), self.release,outfile,analysis_date)) 
                if (galahic==-1):
					galahic = fib_table[o]['NAME'] 
                filename = "{0}_{1}{2}.fits".format(data_date,self.ccd_nums[header['SPECTID']],galahic)
                flux_hdu = pyfits.ImageHDU(linwave_flux.data[o,:].astype('f4'),header)
                sig_hdu  = pyfits.ImageHDU(linwave_sigma.data[o,:].astype('f4'))
                #Add in header stuff from fiber table.
                flux_hdu.header['RA'] = np.degrees(fib_table[o]['RA'])
                flux_hdu.header['DEC'] = np.degrees(fib_table[o]['DEC'])
                flux_hdu.header['V_BARY'] = bcorr[o]
                flux_hdu.header['FIBRE'] = o + 1 #!!! Starting at 1 for 2dFDR convention...
                flux_hdu.header['PIVOT'] = fib_table[o]['PIVOT']
                for key in ("CRVAL1", "CDELT1", "CRPIX1", "CTYPE1", "CUNIT1"):
                    flux_hdu.header[key] = linwave_flux.header[key]
                    sig_hdu.header[key] = linwave_sigma.header[key]
                hl = pyfits.HDUList()
                hl.append(flux_hdu)
                hl.append(sig_hdu)
                hl.writeto(self.gdir + field_directory +'/' + filename, clobber=True)
            f_csvfile.close()
                        
        return flux_comb, flux_comb_sigma
            
    def create_barycentric_spectra(self, header, fib_table, flux, flux_sigma, wavelengths,is_std=False):
        """Interpolate flux onto a wavelength grid spaced regularly in log(wavelength),
        after shifting to the solar system barycenter"""
        if not barycorr:
            print("ERROR: Need PyAstronomy for create_barycentric_spectra()")
            raise UserWarning
        #The bcorr is the barycentric correction in km/s, with a sign convenction
        #with positive meaning moving towards the star. This means that we have to red-shift
        #the interpolated spectra, meaning that the new wavelength scale has to be shifted
        #to the blue.
        if is_std:
            hcorr, bcorr = pyasl.baryCorr(header['UTMJD'] + 2400000.5, header['MEANRA'], header['MEANDEC'], deq=2000.0)
        else:
            hcorr, bcorr = pyasl.baryCorr(header['UTMJD'] + 2400000.5, np.degrees(fib_table['RA']),np.degrees(fib_table['DEC']), deq=2000.0)
        nfib = wavelengths.shape[0]
        new_flux = np.zeros( (nfib,self.fixed_nwave) )
        new_flux_sigma = np.zeros( (nfib,self.fixed_nwave) )
        new_lin_flux = np.zeros( (nfib,self.fixed_nwave) )
        new_lin_flux_sigma = np.zeros( (nfib,self.fixed_nwave) )
        new_wave = self.fixed_wave0[header['SPECTID']]*np.exp(np.arange(self.fixed_nwave)/float(self.fixed_R))
        new_lin_wave = self.fixed_wave0[header['SPECTID']]*(1 + np.arange(self.fixed_nwave)/float(self.fixed_R))
        dnew_wave = new_wave[1:]-new_wave[:-1]
        dnew_wave = np.append(dnew_wave, dnew_wave[-1])
        dnew_lin_wave = new_lin_wave[1]-new_lin_wave[0]
        new_wavelengths = wavelengths.copy()
        for i in range(nfib):
            new_wavelengths[i,:] = wavelengths[i,:]*(1 + bcorr[i]/2.9979e5)
            dwave = new_wavelengths[i,1:]-new_wavelengths[i,:-1]
            dwave = np.append(dwave, dwave[-1])
            dwave_lin = np.interp(new_lin_wave, new_wavelengths[i,:], dwave)
            dwave     = np.interp(new_wave, new_wavelengths[i,:], dwave)
            new_flux[i,:] = np.interp(new_wave, new_wavelengths[i,:], flux[i,:], left=np.nan, right=np.nan)
            new_lin_flux[i,:] = np.interp(new_lin_wave, new_wavelengths[i,:], flux[i,:], left=np.nan, right=np.nan)
            #Preserve the meaning of sigma if many samples are averaged together. 
            new_flux_sigma[i,:] = np.interp(new_wave, new_wavelengths[i,:], flux_sigma[i,:], 
                left=np.nan, right=np.nan) * np.sqrt(dnew_wave/dwave)
            new_lin_flux_sigma[i,:] = np.interp(new_lin_wave, new_wavelengths[i,:], flux_sigma[i,:], 
                left=np.nan, right=np.nan) * np.sqrt(dnew_lin_wave/dwave_lin)
        # The log-wavelength header
        new_hdu = pyfits.ImageHDU(new_flux.astype('f4'))
        sig_hdu = pyfits.ImageHDU(new_flux_sigma.astype('f4'))
        new_hdu.header['CRVAL1']=np.log(self.fixed_wave0[header['SPECTID']])
        new_hdu.header['CDELT1']=1.0/self.fixed_R
        new_hdu.header['CRPIX1']=1.0
        new_hdu.header['CRVAL2']=0.0
        new_hdu.header['CDELT2']=1.0
        new_hdu.header['CRPIX2']=1.0
        new_hdu.header['CTYPE1']='log(Wavelength)'
        new_hdu.header['CUNIT1']='Angstroms'
        new_hdu.header['CTYPE2']='Fibre Number'
        new_hdu.header['CUNIT2'] = ''
        sig_hdu.header = new_hdu.header
        # The linear-wavelength header
        new_lin_hdu = pyfits.ImageHDU(new_lin_flux.astype('f4'))
        lin_sig_hdu = pyfits.ImageHDU(new_lin_flux_sigma.astype('f4'))
        new_lin_hdu.header['CRVAL1']=self.fixed_wave0[header['SPECTID']]
        new_lin_hdu.header['CDELT1']=self.fixed_wave0[header['SPECTID']]/self.fixed_R
        new_lin_hdu.header['CRPIX1']=1.0
        new_lin_hdu.header['CRVAL2']=0.0
        new_lin_hdu.header['CDELT2']=1.0
        new_lin_hdu.header['CRPIX2']=1.0
        new_lin_hdu.header['CTYPE1']='Wavelength'
        new_lin_hdu.header['CUNIT1']='Angstroms'
        new_lin_hdu.header['CTYPE2']='Fibre Number'
        new_lin_hdu.header['CUNIT2'] = ''
        lin_sig_hdu.header = new_lin_hdu.header
        return new_hdu, sig_hdu, new_lin_hdu, lin_sig_hdu, new_wavelengths, bcorr
    
    def fit_tramlines(self, infile, subtract_bias=False, fix_badpix=False):
        """Make a linear fit to tramlines, based on a simplified PSF model """
        im = self.basic_process(infile)
        header = pyfits.getheader(self.ddir + infile)
        ftable = pyfits.getdata(self.ddir + infile,1)
        if subtract_bias:
            im -= pyfits.getdata(self.rdir + 'bias.fits')
        if fix_badpix:
            medim = nd.filters.median_filter(im,size=5)
            badpix = pyfits.getdata(self.cdir + 'badpix.fits')
            ww = np.where(badpix)
            im[ww] = medim[ww]
        nsamp = 8
        nslitlets=40
        nfibres=10
        #Maximum number of pixels for extraction, including tramline tilt.
        npix_extract = 20
        #Oversampling of the PSF - should be an odd number due to symmetrical 
        #convolutions.
        oversamp = 3
        dely_deriv = 0.01
        flux_min = 10
        nx = im.shape[1]
        psf = self.make_psf(npix=npix_extract, oversamp=oversamp)
        #Manually set the index for the samples to the median filtered image.
        #!!! Hardwired numbers - to be changed for FunnelWeb
        x_ix = 256 + np.arange(nsamp,dtype=int)*512
        y_ix_oversamp = np.arange(oversamp*npix_extract) + 0.5 - (oversamp*npix_extract)/2.0
        y_ix = ( np.arange(npix_extract) + 0.5 - (npix_extract)/2.0 )*oversamp
        #Filtered image
        imf = nd.filters.median_filter(im,size=(1,11))
        imf = imf[:,x_ix]
        psfim_plus  = np.zeros((npix_extract, nsamp))
        psfim_minus = np.zeros((npix_extract, nsamp))
        dy        = np.zeros((nslitlets,nfibres,nsamp))
        weight_dy = np.zeros((nslitlets,nfibres,nsamp))
        #Read in the tramline initial parameters from the calibration directory
        p_tramline = np.loadtxt(self.cdir + 'tramlines_p' + header['SOURCE'][6] + '.txt')
        #Make a matrix that maps p_tramline numbers to dy, i.e. for parameters
        #p_tramline, we get the y positions by np.dot(tramline_matrix,p_tramline)
        tramline_matrix = np.zeros((nfibres*nsamp,4))
        tramline_matrix[:,0] = np.tile(x_ix**2,nfibres) # Parabolic term with x
        tramline_matrix[:,1] = np.tile(x_ix,nfibres)    # Linear term with x
        tramline_matrix[:,2] = np.ones( nfibres*nsamp ) # Offset term
        tramline_matrix[:,3] = np.repeat( (np.arange(nfibres)+0.5-nfibres//2),
                                  nsamp ) # Stretch term.
        #Loop through a few different offsets to get a global shift.
        ypix = np.dot(tramline_matrix,p_tramline.T)
        ypix = ypix.reshape( (nfibres,nsamp,nslitlets) )
        ypix = np.swapaxes(ypix,0,1).flatten().astype(int)
        xpix = np.repeat( range(nsamp), nfibres*nslitlets)
        nshifts = 20
        flux_peak = np.zeros(nshifts)
        for i in range(nshifts):
            flux_peak[i] = np.sum(imf[np.maximum(np.minimum(ypix+i-nshifts//2,nx),0),xpix])
        p_tramline[:,2] += np.argmax(flux_peak) - nshifts//2       
        #Make 4 Newton-Rhapson iterations to find the best fitting tramline parameters
        for count in range(0,3):
         #Go through every slitlet, fiber and sample (nsamp) in the wavelength
         #direction, finding the offsets.
         for i in range(nslitlets):
            for j in range(nfibres):
                center_int = np.int(p_tramline[i,2] + p_tramline[i,3]*(j+0.5-nfibres//2))
                center_int = np.maximum(center_int,npix_extract//2)
                center_int = np.minimum(center_int,nx-npix_extract//2)
                subim = imf[center_int - npix_extract//2:center_int + npix_extract//2,:]
                #Start off with a slow interpolation for simplicity. 
                for k in range(nsamp):
                    offset = p_tramline[i,2] + p_tramline[i,1]*x_ix[k] + p_tramline[i,0]*x_ix[k]**2 + p_tramline[i,3]*(j+0.5-nfibres//2) - center_int
                    psfim_plus[:,k]  = np.interp(y_ix - (offset + dely_deriv)*oversamp, y_ix_oversamp, psf)
                    psfim_minus[:,k] = np.interp(y_ix - (offset - dely_deriv)*oversamp, y_ix_oversamp, psf)
                psfim = 0.5*(psfim_plus + psfim_minus)
                psfim_deriv = ( psfim_plus - psfim_minus )/2.0/dely_deriv 
                psfsum = np.sum(psfim*subim,axis=0)
                dy[i,j,:] = np.sum(psfim_deriv*subim,axis=0)/np.maximum(psfsum,flux_min)*np.sum(psfim**2)/np.sum(psfim_deriv**2)
                weight_dy[i,j,:] = np.maximum(psfsum-flux_min,0)
        
         print("RMS tramline offset (iteration " +str(count)+ "): " + str(np.sqrt(np.mean(dy**2))))
         for i in range(nslitlets):
            #Now we fit to the dy values. 
            W = np.diag(weight_dy[i,:,:].flatten())
            y = dy[i,:,:].flatten()
            delta_p = np.linalg.solve(np.dot(np.transpose(tramline_matrix),np.dot(W,tramline_matrix)) ,\
                np.dot(np.transpose(tramline_matrix),np.dot(W,y)) )
            p_tramline[i,:] += delta_p
        np.savetxt(self.rdir + 'tramlines_p' + header['SOURCE'][6] + '.txt', p_tramline, fmt='%.5e')
        
    def reduce_field(self,obj_files, arc_file, flat_file, is_std=False):
        """A wrapper to completely reduce a field, assuming that a bias already exists."""
        self.fit_tramlines(flat_file)
        fibre_flat = self.create_fibre_flat(flat_file)
        arc, arc_sig = self.extract([arc_file])
        wavelengths = self.fit_arclines(arc, pyfits.getheader(self.ddir + arc_file))
        cube,badpix = self.make_cube_and_bad(obj_files)
        flux, sigma = self.extract(obj_files, cube=cube, badpix=badpix)
        if not is_std:
            flux, sigma = self.sky_subtract(obj_files, flux, sigma, wavelengths, fibre_flat=fibre_flat)
        comb_flux, comb_flux_sigma = self.combine_single_epoch_spectra(flux, sigma, wavelengths, infiles=obj_files, is_std=is_std)
        return comb_flux, comb_flux_sigma
        
    def go(self, min_obj_files=2, dobias=True, skip_done=False):
        """A simple function that finds all fully-executed fields (in this case meaning
        at least min_obj_files exposures on the field) and analyses them.
        
        Parameters
        ----------
        min_obj_files: int
            Minimum number of files per field to call it "good"
        dobias: boolean
            Do we bother subtracting the bias frame.
        """
        all_files = np.array(sorted([os.path.basename(x) for x in glob.glob(self.ddir + '[0123]*[0123456789].fit*')]))
        if len(all_files)==0:
            print("You silly operator. No files. Input directory is: " + self.ddir)
            return
        biases = np.array([],dtype=np.int)
        flats = np.array([],dtype=np.int)
        arcs = np.array([],dtype=np.int)
        objects = np.array([],dtype=np.int)
        is_stds = np.array([],dtype=np.bool)
        cfgs = np.array([],dtype=np.int)
        field_ids = np.array([],dtype=np.int)
        for i,file in enumerate(all_files):
            header= pyfits.getheader(self.ddir + file)
            try: 
                cfg = header['CFG_FILE']
            except:
                cfg = ''
            cfgs = np.append(cfgs,cfg)
            field_id = cfg
            is_std = False
            if header['NDFCLASS'] == 'BIAS':
                biases = np.append(biases,i)
            #!!! No idea what LFLAT is, but it seems to be a flat.
            #!!! Unfortunately, if it is used, the header['SOURCE'] seems to be invalid, so
            #the code doesn't know what field is in use.
            elif header['NDFCLASS'] == 'MFFFF':
                flats = np.append(flats,i)
            elif header['NDFCLASS'] == 'MFARC':
                arcs = np.append(arcs,i)
            elif (header['NDFCLASS'] == 'MFOBJECT'):
                objects = np.append(objects,i)
            elif (header['NDFCLASS'] == 'MFFLX'):
                objects = np.append(objects,i)
                is_std = True
                field_id = cfg + header['STD_NAME']
            else:
                print("Unusual (ignored) NDFCLASS " + header['NDFCLASS'] + " for file: " + file)
            field_ids = np.append(field_ids,field_id)
            is_stds = np.append(is_stds, is_std)
                
        #Forget about configs for the biases - just use all of them! (e.g. beginning and end of night)
        if len(biases) > 2 and dobias:
            if skip_done and os.path.isfile(self.rdir + '/' + 'bias.fits'):
                print("Skipping (already done) bias creation")
            else:
                print("Creating Biases")
                bias = self.median_combine(all_files[biases], 'bias.fits')
        else:
            print("No biases. Will use default...")
#Old code that treated all files with the same sds file as one.
#        for cfg in set(cfgs):
#            #For each config, check that there are enough files.
#            cfg_flats = flats[np.where(cfgs[flats] == cfg)[0]]
#            cfg_arcs = arcs[np.where(cfgs[arcs] == cfg)[0]]
#            cfg_objects = objects[np.where(cfgs[objects] == cfg)[0]]
        #Lets make a config index that changes every time there is a tumble.
        cfg_starts = np.append(0,np.where(field_ids[1:] != field_ids[:-1])[0]+1)
        cfg_ends = np.append(np.where(field_ids[1:] != field_ids[:-1])[0]+1,len(field_ids))
        for i in range(len(cfg_starts)):
            cfg_start = cfg_starts[i]
            cfg_end = cfg_ends[i]
            cfg_is_std = is_stds[cfg_starts[i]]
            #For each config, check that there are enough files.
            if is_stds[cfg_starts[i]]:
                cfg_flats = flats[np.where( (flats >= cfg_start-2) & (flats < cfg_end+2))[0]]
                same_cfg = np.where(cfgs[cfg_flats] == cfgs[cfg_starts[i]])[0]
                cfg_flats = cfg_flats[same_cfg]
                cfg_arcs = arcs[np.where( (arcs >= cfg_start-2) & (arcs < cfg_end+2))[0]]
                same_cfg = np.where(cfgs[cfg_arcs] == cfgs[cfg_starts[i]])[0]
                cfg_arcs = cfg_arcs[same_cfg]
            else:
                cfg_flats = flats[np.where( (flats >= cfg_start) & (flats < cfg_end))[0]]
                cfg_arcs = arcs[np.where( (arcs >= cfg_start) & (arcs < cfg_end))[0]]
            ww = np.where( (objects >= cfg_start) & (objects < cfg_end))[0]
            cfg_objects = objects[ww]
            if len(cfg_flats) == 0:
                print("No flat for field: " + cfgs[cfg_start] + " Continuing to next field...")
            elif len(cfg_arcs) == 0:
                print("No arc for field: " + cfgs[cfg_start] + " Continuing to next field...")
            elif len(cfg_objects) < min_obj_files:
                print("Require at least 2 object files. Not satisfied for: " + cfgs[cfg_start] + " Continuing to next field...")
            else:
                if skip_done:
                    comb_filename = self.make_comb_filename(all_files[cfg_objects[0]])
                    if os.path.isfile(self.rdir + '/' + comb_filename):
                        header = pyfits.getheader(self.rdir + '/' + comb_filename)
                        if header['NRUNS'] == len(cfg_objects):
                            print("Ignoring processed field: " + comb_filename)
                            continue
                print("Processing field: " + cfgs[cfg_start])
                #!!! NB if there is more than 1 arc or flat, we could be more sophisticated here... 
                self.reduce_field(all_files[cfg_objects], all_files[cfg_arcs[0]], all_files[cfg_flats[0]], is_std = cfg_is_std)
        
# !!! The "once-off" codes below here could maybe be their own module???
        
    def find_tramlines(self, infile, subtract_bias=False, fix_badpix=False, nsearch=20, \
        fillfrac=1.035, central_sep=9.3, global_offset=-6, c_nonlin=3.2e-4):
        """For a single flat field, find the tramlines. This is the slightly manual
        part... the 4 numbers (fillfrac,central_sep, global_offset,c_nonlin) have
        to be set so that a good fit is made.
        
        If the fit is good, the tramlines_0.txt or tramlines_1.txt should be 
        moved to the calibration directory cdir """
        
        nslitlets=40 #Number of slitlets.
        nfibres=10   #Fibres per slitlet.
        resamp = 3  #sub-pixel sampling in grid search
        
        nsearch *= resamp  
        global_offset *= resamp
        
        im = self.basic_process(infile)
        header = pyfits.getheader(self.ddir + infile)
        ftable = pyfits.getdata(self.ddir + infile,1)
        if subtract_bias:
            im -= pyfits.getdata(self.rdir + 'bias.fits')
        if fix_badpix:
            medim = nd.filters.median_filter(im,size=5)
            badpix = pyfits.getdata(self.cdir + 'badpix.fits')
            ww = np.where(badpix)
            im[ww] = medim[ww]
        szy = im.shape[0]
        szx = im.shape[1]
 
 #Next there are 2 general options... 
 #A: Deciding on the central wavelength pixel coordinates 
 #for each fiber image, and deciding on elements of a 2nd order polynomial, i.e. 
 # ypos = y(central) + a1*dx + a2*dx**2 + a3*dy*dx + a4*dx**2*dy
 #Each of these is a standard nonlinear fitting process. 
 #B: Explicitly fit to each slitlet individually. This approach was chosen.
 #
 #Fibre separations in slitlets:
 #First slitlet: 3.83 pix separations. 
 #Central slitlet: 3.67 pix separation. 
 #Last slitlet: 3.82 pix separations.
 #i.e. separation  = 3.67 + 4e-4*(slitlet - 20.5)**2
 #Good/Bad gives the brightness of each fiber. 
 #OR... just fit a parabola to each slitlet. 

        #Central solution...
        fit_px = [500,1500,2500,3500] #Pixels to try fitting the slitlet to.
        ncuts = len(fit_px)
        cuts = np.zeros((ncuts,szy*resamp))
        for i in range(ncuts):
            acut = np.median(im[:,fit_px[i]-2:fit_px[i]+3],axis=1)
            cuts[i,:] = acut[np.arange(szy*resamp)/resamp]

 # From Koala...
 # params = np.loadtxt(pfile,dtype={
 #    'names':('spos','grid_x', 'grid_y', 'good','name'),'formats':('i2','i2','i2','S4','S15')})

        #Now we run through the slitlets... some fixed numbers in here.
        soffsets = np.zeros((nslitlets,ncuts),dtype='int')

        outf = open(self.rdir + 'tramlines_p' + header['SOURCE'][6] + '.txt','w')
        plt.clf()
        plt.imshow(np.minimum(im,3*np.mean(im)),aspect='auto', cmap=cm.gray, interpolation='nearest')
        x = np.arange(szx)
        for i in np.arange(nslitlets):
            #flux = (ftable[i*nfibres:(i+1)*nfibres]['TYPE'] != 'N') *  (ftable[i*nfibres:(i+1)*nfibres]['TYPE'] != 'F')
            flux = (ftable[i*nfibres:(i+1)*nfibres]['TYPE'] != 'F')
            #8.33, 9.33, 8.44
            fsep = resamp*central_sep*(1 - c_nonlin*(i - (nslitlets-1)/2.0)**2)
            #subsample by a factor of resamp only, i.e. 4 x 3 = 12 pix per fibre.
            prof = np.zeros(szy*resamp)
            fibre_offsets = np.zeros(nfibres)
            for j in range(nfibres):
                #Central pixel for this fibre image...
                fibre_offsets[j] = (j- (nfibres-1)/2.0)*fsep
                cpix = szy*resamp/2.0 + fibre_offsets[j]
                for px in np.arange(np.floor(cpix-resamp),np.ceil(cpix+resamp+1)):
                    prof[px] = np.exp(-(px-cpix)**2/resamp**2.0)*flux[j]
            #Now find the best match. Note that slitlet 1 starts from the top.
            #!!! On this next line, the nonlinear process becomes important !!!
            offsets = global_offset -nsearch/2.0 + resamp*(1 - c_nonlin/3.0*(i - (nslitlets-1)/2.0)**2)*(i-(nslitlets-1)/2.0)/nslitlets*szy*fillfrac + np.arange(nsearch)
            offsets = offsets.astype(int)
            for k in np.arange(ncuts):
                xp = np.zeros(nsearch)
                for j in range(nsearch):
                    xp[j] = np.sum(np.roll(prof,offsets[j])*cuts[k,:])
            #   print np.argmax(xp)
                soffsets[i,k] = offsets[np.argmax(xp)]
            #Great! At this point we have everything we need for a parabolic fit to the "tramline".
            #Lets make this fit and write to file.
            p = np.polyfit(fit_px,(soffsets[i,:] + szy*resamp/2.0)/float(resamp),2)
            pp = np.poly1d(p)
            #outf.write('{0:3.4e} {1:3.4e} {2:3.4e}\n'.format(p[0],p[1],p[2]+fibre_offsets[j]/resamp))
            outf.write('{0:3.4e} {1:3.4e} {2:3.4e} {3:3.4e}\n'.format(p[0],p[1],p[2],fsep/resamp))
            for j in range(nfibres):
                if flux[j]>0:
                    if (j == 0):
                        plt.plot(x,pp(x)+fibre_offsets[j]/resamp,'r-') 
                    else:
                        plt.plot(x,pp(x)+fibre_offsets[j]/resamp,'g-')
        #import pdb; pdb.set_trace()
        outf.close()
    
    def compute_model_wavelengths(self,header):
        """Given a fits header and other fixed physical numbers, compute the model 
        wavelengths for the central fiber for HERMES.
        
        Parameters
        ----------
        header: pyfits header
            Header of a file to compute the nominal wavelengths for.
            
        Returns
        -------
        wavelengths: array
            Wavelengths of the central fiber for each pixel in Angstroms.
        """
        try:
            gratlpmm = header['GRATLPMM']
        except:
            print("ERROR: Could not read grating parameter from header")
            raise UserWarning
        #A distortion estimate of 0.04 came from the slit image - 
        #somewhat flawed because there is distortion
        #from both the collimator and camera. Distortion in the wavelength direction
        #is only due to the camera. This is the distortion due to and angle of
        #half a chip, i.e. x' = x/(1 + distortion*x^2)
        #The best value for the wavelength direction is pretty close to 0...
        distortion = 0.00
        camfl = 1.7*190.0   #In mm
        pixel_size = 0.015   #In mm
        #The nominal central wavelength angle from the HERMES design.
        #Unclear if 68.1 or 67.2 is the right number...
        beta0 = np.radians(67.2)
        gratangl = np.radians(67.2)
        szx = header['WINDOXE1'] #This removes the overscan region.
        center_pix = 2048        
        d = 1.0/gratlpmm
        #First, convert pixel to x-angle (beta) 
        #Unlike Koala, we'll ignore gamma in the first instance. 
        #gamma_max = szy/2.0*pixel_size/camfl
        dbeta = np.arange(szx,dtype=float) - center_pix
        dbeta *= (1.0 + distortion*(dbeta/center_pix)**2)
        dbeta *= pixel_size/camfl
        beta = beta0 + dbeta
        wavelengths = d*(np.sin(gratangl) + np.sin(beta))
        #Return wavelengths in Angstroms.
        return wavelengths * 1e7

    def adjust_wavelengths(self, wavelengths, p):
        """Adjust a set of model wavelengths by a quadratic function, 
        used by find_arclines to find the best fit.
        
        Parameters
        ----------
        wavelengths: (nx) array
            One-dimensional wavelength array
        p: (3) array 
            p[0] is a quadratic term, with p[0]=1 giving a 1 Angstrom shift at the edges
            of the wavelength array.
            p[1] is a linear dispersion term, with p[1]=0.01 giving a 1% shift at the 
            edges of the wavelength array with respect to the center.
            p[2] is a shift in pixels.
            
        Returns
        -------
        wavelengths: (nx) array
            One-dimensional wavelength array
        """
        #Median wavelength.
        medw = np.median(wavelengths)
        #Delta wavelength from center to edge.
        dw = max(wavelengths) - medw
        wavelengths = p[0]*((wavelengths-medw)/dw)**2 + p[1]*(wavelengths-medw) + medw
        wstep = wavelengths[1:]-wavelengths[:-1]
        wstep = np.append(wstep, wstep[-1])
        return wavelengths - p[2]*wstep

    def find_arclines(self,arc, header):
        """ Based on the model wavelength scale from degisn physical parameters only, try to find the
        positions of the arc lines. This is not necessarily a robust program  - hopefully it 
        only has to be run once, and the reasonable fit to the arc lines can then be input
        (through the calbration directory) to fit_arclines.
        
        Parameters
        ----------
        arc: (nfibres*nslitlets, nwave) array
            Extracted arc spectra
        header: pyfits header
            Header for the arc file.
            
        Returns
        -------
        wavelengths: (nslitlets*nfibres, nx) array
            Wavelengths of each pixel in Angstroms.
        """
        
        #The following code originally from quick_image_gui.py for Koala. The philosophy is to do our best
        #based on a physical model to put the arclines on to chip coordinates... to raise the
        #arc fluxes to a small power after truncating any noise, then to maximise the 
        #cross-correlation function with reasonably broad library arc line functions. 
        arc_center = np.median(arc[170:230,:],axis=0)
        arc_center = np.sqrt(np.maximum(arc_center - np.median(arc_center),0))
        szx = arc.shape[1]
        arclines = np.loadtxt(self.cdir + '../thxe.arc')
        wavelengths = self.compute_model_wavelengths(header)
        arc_ix = np.where( (arclines[:,0] > min(wavelengths) + 0.5) * ((arclines[:,0] < max(wavelengths) - 0.5)) )[0]
        if len(arc_ix)==0:
            print("Error: No arc lines within wavelength range!")
            raise UserWarning
        arclines = arclines[arc_ix,:]
        g = np.exp(-(np.arange(15)-7.0)**2/30.0)
        #Only consider offsets of up to 50 pixels.
        npix_search = 120
        nscale_search = 101
        nquad_search = 15
        scales = 1.0 + 0.0015*(np.arange(nscale_search) - nscale_search//2)
        #Peak to valley in Angstroms.
        quad = 0.2*(np.arange(nquad_search) - nquad_search//2)
        corr3d = np.zeros( (nquad_search, nscale_search, 2*npix_search) )
        print("Beginning search for optimal wavelength scaling...")
        for j in range(nquad_search):
         for i in range(nscale_search):
            xcorr = np.zeros(szx)
            pxarc = np.interp(arclines[:,0],self.adjust_wavelengths(wavelengths, [quad[j],scales[i],0]), np.arange(szx)).astype(int)
            xcorr[pxarc] = np.sqrt(arclines[:,1])
            xcorr = np.convolve(xcorr,g,mode='same')        
            corfunc=np.correlate(arc_center,xcorr,mode='same')    
            corr3d[j,i,:]=corfunc[szx//2-npix_search:szx//2+npix_search] 
            #if (i == 32):
            #    import pdb; pdb.set_trace()
        pix_offset = np.unravel_index(corr3d.argmax(), corr3d.shape)
        print("Max correlation: " + str(np.max(corr3d)))
        #corfunc[szx//2+npix_search:]  = 0
        #plt.plot(np.arange(2*npix_search) - npix_search, corfunc[szx//2-npix_search:szx//2+npix_search])
        #pix_offset = np.argmax(corfunc) - szx//2
        plt.clf()
        plt.imshow(corr3d[pix_offset[0],:,:], interpolation='nearest')
        plt.title('Click to continue...')
        plt.ginput(1)
        xcorr = np.zeros(szx)
        new_wavelengths = self.adjust_wavelengths(wavelengths, [quad[pix_offset[0]],scales[pix_offset[1]],pix_offset[2]-npix_search])
        pxarc = np.interp(arclines[:,0],new_wavelengths,np.arange(szx)).astype(int)
        pxarc = np.maximum(pxarc,0)
        pxarc = np.minimum(pxarc,szx)
        xcorr[pxarc] = np.sqrt(arclines[:,1])
        xcorr = np.convolve(xcorr,g,mode='same')
        plt.clf()
        plt.plot(new_wavelengths, xcorr)
        plt.plot(new_wavelengths, arc_center)
        plt.xlabel('Wavelength')
        
        #Now go through each slitlet (i.e. for a reliable median) 
        #Making appropriate adjustments to the scale.
        nslitlets = 40
        nfibres = 10
        slitlet_shift = np.zeros(nslitlets)
        for i in range(nslitlets):
            arc_med = np.median(arc[i*nfibres:(i+1)*nfibres,:],axis=0)
            arc_med = np.sqrt(np.maximum(arc_med - np.median(arc_med),0))
            corfunc=np.correlate(arc_med,xcorr,mode='same')
            corfunc[:szx//2-npix_search]=0
            corfunc[szx//2+npix_search:]=0
            slitlet_shift[i] = np.argmax(corfunc)-szx//2
            print("Slitlet " + str(i) + " correlation " +str(np.max(corfunc)))
        #Save a polynomial fit to the wavelengths versus pixel
        #a_5 x^5 + a_4 x^4     + a_3 x^3     + a_2 x^2 
        #        + b_4 x^4 y   + b_3 x^3 y   + b_2 x^2 y 
        #                      + c_3 x^3 y^2 + c_2 x^2 y^2
        #                                    + d_2 x^2 y^3
        x_ix = np.arange(szx) - szx//2
        poly_p = np.polyfit(x_ix,new_wavelengths,5)
        wcen = poly_p[5]
        disp = poly_p[4]
        poly2dfit = np.append(poly_p[0:4],np.zeros(6))
        np.savetxt(self.rdir + 'poly2d_p' + header['SOURCE'][6] + '.txt',poly2dfit, fmt='%.6e')
        #The individual fibres have a pixel shift, which we convert to a wavelength shift
        #at the chip center.
        fibre_fits = np.zeros((nslitlets*nfibres,2))
        fibre_fits[:,0] = disp * np.ones(nslitlets*nfibres)
        #pixel shift multiplied by dlambda//dpix = dlambda
        fibre_fits[:,1] = wcen - np.repeat(slitlet_shift,nfibres) * disp
        np.savetxt(self.rdir + 'dispwave_p' + header['SOURCE'][6] + '.txt',fibre_fits, fmt='%.6e')
        
        return new_wavelengths
        
    def find_wavelengths(self, poly2dfit, fibre_fits, nx):
        """Find the wavelengths for all fibers and all pixels, based on the model that 
        includes the polynomial 2D fit ant the linear fiber fits.
        
        Parameters
        ----------
        poly2dfit: (10) array 
            2D polynomial fit parameters.
        fiber_fits: (nfibres*nslitlets, 2) array
            Linear dispersion fit to each fiber
        nx: int
            Number of pixels in the x (dispersion, i.e. wavelength) direction
        
        Returns
        -------
        wavelengths: (nslitlets*nfibres, nx) array
            Wavelengths of each pixel in Angstroms.
        """
        nslitlets=40 #!!! This should be a property of the main class.
        nfibres=10
        wavelengths = np.zeros( (nslitlets*nfibres, nx) )
        x_ix = np.arange(nx) - nx//2
        y_ix = np.arange(nslitlets*nfibres) - nslitlets*nfibres//2
        xy_ix = np.meshgrid(x_ix,y_ix)
        #Start with the linear component of the wavelength scale.
        for i in range(nslitlets*nfibres):
            wavelengths[i,:] = fibre_fits[i,0] * x_ix + fibre_fits[i,1]
        #As we have a 2D polynomial, bite the bullet and just manually create the
        #main offset...
        poly_func = np.poly1d(np.append(poly2dfit[0:4],[0,0]))
        wavelengths += poly_func(xy_ix[0])
        poly_func = np.poly1d(np.append(poly2dfit[4:7],[0,0]))
        wavelengths += poly_func(xy_ix[0])*xy_ix[1]
        poly_func = np.poly1d(np.append(poly2dfit[7:9],[0,0]))
        wavelengths += poly_func(xy_ix[0])*xy_ix[1]**2
        wavelengths += poly2dfit[9]*xy_ix[0]**2*xy_ix[1]**3
        return wavelengths
        
    def fit_arclines(self,arc, header, plotit=False, npix_extract = 51):
        """Assuming that the initial model is good enough, fit to the arclines.
        Whenever an fibre isn't in use, this routine will take the fibre_fits from the 
        two nearest good fibres in the slitlet. 
        
        npix_extract: 
            Maximum number of pixels for extraction, including tramline tilt.
            
        The procedure is to:
        0) Based on a model, find the wavelength of every pixel.
        1) First find the x pixels corresponding to the model arc lines, as well
        as the local dispersion at each line (arc_x and arc_disp).
        2) Create a matrix such that wave = M * p, with p our parameters.
        3) Find the dx values, convert to dwave.
        4) Convert the dwave values to dp. 
        
        Parameters
        ----------
        arc: array
            Extracted arc spectrum
        header: pyfits header
            Header of the arc file
        plotit: boolean (default False)
            Do we show the arc line fits?
        npix_extract: int (default 51)
            Number of pixels to extract in the fitting of the arc line.
            
        Returns
        -------
        wavelengths: (nslitlets*nfibres, nx) array
            Wavelengths of each pixel in Angstroms.
        """
        nslitlets=40
        nfibres=10
        #Oversampling of the PSF - should be an odd number due to symmetrical 
        #convolutions.
        oversamp = 3
        delx_deriv = 0.01
        flux_min = 20
        flux_max = 200
        nx = arc.shape[1]
        poly2dfit  = np.loadtxt(self.cdir + 'poly2d_p' + header['SOURCE'][6] + '.txt')
        npoly_p = len(poly2dfit) #!!! Has to be 10 for the code below so far.
        fibre_fits = np.loadtxt(self.cdir + 'dispwave_p' + header['SOURCE'][6] + '.txt')
        wavelengths = self.find_wavelengths(poly2dfit, fibre_fits, nx)
        #Read in the arc file...
        arclines = np.loadtxt(self.cdir + '../thxe.arc')
        arc_ix = np.where( (arclines[:,0] > np.min(wavelengths) + 0.5) * ((arclines[:,0] < np.max(wavelengths) - 0.5)) )[0]
        arclines = arclines[arc_ix,:]
        narc = len(arc_ix)
        #Initialise the arc x and dispersion values...
        arc_x    = np.zeros( (nfibres*nslitlets,narc) )
        arc_disp = np.zeros( (nfibres*nslitlets,narc) )
        #Find the x pixels corresponding to each wavelength.
        #PSF stuff...
        psf = self.make_psf(npix=npix_extract, oversamp=oversamp)
        #e_ix is the extraction index.
        e_ix_oversamp = np.arange(oversamp*npix_extract) - oversamp*npix_extract//2
        e_ix = ( np.arange(npix_extract) - npix_extract//2 )*oversamp
        psfim_plus  = np.zeros((npix_extract, nslitlets*nfibres))
        psfim_minus = np.zeros((npix_extract, nslitlets*nfibres))
        #Indices for later...
        y_ix = np.arange(nslitlets*nfibres) - nslitlets*nfibres//2
        y_ix = np.repeat(y_ix,narc).reshape(nfibres*nslitlets,narc)
        x_ix = np.arange(nx) - nx//2
        arcline_matrix = np.zeros((nfibres*nslitlets, narc,npoly_p + 2*nfibres*nslitlets))
        #Whoa! That was tricky. Now lets use our PSF to fit for the arc lines.
        dx        = np.zeros((nslitlets*nfibres,narc))
        weight_dx = np.zeros((nslitlets*nfibres,narc))
        for count in range(0,3):
            wavelengths = self.find_wavelengths(poly2dfit, fibre_fits, nx)
            #Find the arc_x values...
            #Dispersion in the conventional sense, i.e. dlambda/dx
            for i in range(nfibres*nslitlets):
                xplus  = np.interp(arclines[:,0] + delx_deriv, wavelengths[i,:], x_ix)
                xminus = np.interp(arclines[:,0] - delx_deriv, wavelengths[i,:], x_ix)
                arc_x[i,:]    = 0.5*(xplus + xminus)
                arc_disp[i,:] = 2.0*delx_deriv/(xplus - xminus)
            
            #Make a matrix that maps model parameters to wavelengths, based on the arc_x values.
            #(nfibres*nslitlets,narc,npoly_p + 2*nfibres*nslitlets)
            arcline_matrix = arcline_matrix.reshape(nfibres*nslitlets, narc,npoly_p + 2*nfibres*nslitlets)
            arcline_matrix[:,:,0] = arc_x**5
            arcline_matrix[:,:,1] = arc_x**4
            arcline_matrix[:,:,2] = arc_x**3
            arcline_matrix[:,:,3] = arc_x**2
            arcline_matrix[:,:,4] = arc_x**4*y_ix
            arcline_matrix[:,:,5] = arc_x**3*y_ix
            arcline_matrix[:,:,6] = arc_x**2*y_ix
            arcline_matrix[:,:,7] = arc_x**3*y_ix**2
            arcline_matrix[:,:,8] = arc_x**2*y_ix**2
            arcline_matrix[:,:,9] = arc_x**2*y_ix**3
            for i in range(nfibres*nslitlets):
                arcline_matrix[i,:,npoly_p+i] = arc_x[i,:]
                arcline_matrix[i,:,npoly_p+nfibres*nslitlets + i] = 1.0
            arcline_matrix = arcline_matrix.reshape(nfibres*nslitlets*narc,npoly_p + 2*nfibres*nslitlets)
            
            #!!! Sanity check that this matrix actually works...
            #p_all = np.append(poly2dfit, np.transpose(fibre_fits).flatten())
            #wavelengths_test = np.dot(arcline_matrix,p_all)
            #import pdb; pdb.set_trace()

            for i in range(narc):
                #Find the range pixel values that correspond to the arc lines for all
                #fibers.
                center_int = int(np.median(arc_x[:,i]) + nx//2)
                subim = np.zeros((arc.shape[0], npix_extract))
                subim[:,np.maximum(npix_extract//2 - center_int,0):\
                        np.minimum(arc.shape[1]-center_int-npix_extract//2-1,arc.shape[1])] = \
                        arc[:,np.maximum(center_int - npix_extract//2,0):np.minimum(center_int + npix_extract//2+1,arc.shape[1])]
                subim = subim.T
                #Start off with a slow interpolation for simplicity. 
                for k in range(nslitlets*nfibres):
                    offset = arc_x[k,i] - center_int + nx//2
                    psfim_plus[:,k]  = np.interp(e_ix - (offset + delx_deriv)*oversamp, e_ix_oversamp, psf)
                    psfim_minus[:,k] = np.interp(e_ix - (offset - delx_deriv)*oversamp, e_ix_oversamp, psf)
                psfim = 0.5*(psfim_plus + psfim_minus)
                psfim_deriv = ( psfim_plus - psfim_minus )/2.0/delx_deriv 
                psfsum = np.sum(psfim*subim,axis=0)
                dx[:,i] = np.sum(psfim_deriv*subim,axis=0)/np.maximum(psfsum,flux_min)*np.sum(psfim**2)/np.sum(psfim_deriv**2)
                weight_dx[:,i] = np.maximum(psfsum-flux_min,1e-3)
                weight_dx[:,i] = np.minimum(weight_dx[:,i],1.5*np.median(weight_dx[:,i]))
                weight_dx[:,i] = np.minimum(weight_dx[:,i],flux_max)
                if count > 0 and plotit:
                    plt.clf()
                    plt.imshow(psfim, aspect='auto', interpolation='nearest')
                    plt.draw()
                    plt.imshow(np.minimum(subim,np.mean(subim)*10), aspect='auto', interpolation='nearest')
                    plt.draw()
            ww = np.where(weight_dx > 0)
            print("RMS arc offset in pix (iteration " +str(count)+ "): " + str(np.sqrt(np.mean(dx[ww]**2))))
            #For fibres with low flux, set dx to the median of the fibers around.
            med_weight = np.median(weight_dx, axis=1)
            ww = np.where(med_weight < 0.3*np.median(med_weight))[0]
            dx[ww] = (nd.filters.median_filter(dx,size=3))[ww]
            
            #Now convert the dx values to dwave.
            dwave = (dx * arc_disp).reshape(nslitlets*nfibres*narc)
            #That was easier than I thought it would be! Next, we have to do the linear fit to the
            #dwave values
            W = np.diag(weight_dx.flatten())
            y = dwave.flatten()
            
            #So the model here is:
            #delta_wavelengths = arcline_matrix . delta_p
            delta_p = np.linalg.solve(np.dot(np.transpose(arcline_matrix),np.dot(W,arcline_matrix)) ,\
                np.dot(np.transpose(arcline_matrix),np.dot(W,y)) )
            poly2dfit -= delta_p[0:npoly_p]
            fibre_fits[:,0] -= delta_p[npoly_p:npoly_p + nfibres*nslitlets]
            fibre_fits[:,1] -= delta_p[npoly_p + nfibres*nslitlets:]
        #Finally, go through the slitlets and fix the fibre fits for low SNR arcs (e.g. dead fibres)
        med_weight = med_weight.reshape((nslitlets,nfibres))
        fibre_fits = fibre_fits.reshape((nslitlets,nfibres,2))
        fib_ix = np.arange(nfibres)
        for i in range(nslitlets):
            ww = np.where(med_weight[i,:] < 0.3*np.median(med_weight[i,:]))[0]
            if len(ww)>0:
                for wbad in ww:
                    nearby_fib = np.where( (wbad - fib_ix) < 4)[0]
                    fibre_fits[i,wbad,0] = np.median(fibre_fits[i,nearby_fib,0])
                    fibre_fits[i,wbad,1] = np.median(fibre_fits[i,nearby_fib,1])
        fibre_fits = fibre_fits.reshape((nslitlets*nfibres,2))
        #Save our fits!
        np.savetxt(self.rdir + 'poly2d_p' + header['SOURCE'][6] + '.txt',poly2dfit, fmt='%.6e')
        np.savetxt(self.rdir + 'dispwave_p' + header['SOURCE'][6] + '.txt',fibre_fits, fmt='%.6e')
        return wavelengths

def worker(arm, skip_done=False):
    """Trivial function needed for multi-threading."""
    arm.go(skip_done=skip_done)
    return
     
def go_all(ddir_root, rdir_root, cdir_root, gdir_root='',skip_done=False):
    """Process all CCDs in a default way
    
    Parameters
    ----------
    ddir_root: string
        Data directory - should contain subdirectories ccd_1, ccd_2 etc
    rdir_root: string
        Reduction directory root - should contain subdirectories ccd_1, ccd_2 etc
    cdir_root: string
        Calibration directory root - this is likely CODE_DIRECTORY/cal. Surely this can 
        be made a default!
    """
    #Create directories if they don't already exist.
    if os.path.isfile(rdir_root):
        print("ERROR: reduction directory already exists as a file!")
        raise UserWarning
    if not os.path.isdir(rdir_root):
        try:
            os.mkdir(rdir_root)
        except:
            print("ERROR: Could not create directory " + rdir_root)
            raise UserWarning
    ccds = ['ccd_1', 'ccd_2', 'ccd_3', 'ccd_4']
    arms = []
    if gdir_root=='':
        gdir_root = rdir_root
    for ccd in ccds:
        if not os.path.isdir(rdir_root + '/' + ccd):
            os.mkdir(rdir_root + '/' + ccd)
        arms.append(HERMES(ddir_root + '/' + ccd+ '/', rdir_root + '/' + ccd + '/', cdir_root + '/' + ccd + '/',gdir=gdir_root))
    threads = []
    for ix,arm in enumerate(arms):
        t = Process(target=worker, args=(arm,skip_done))
        t.name = ccds[ix]
#        t = threading.Thread(target=worker, args=(arm,))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()
        print("Finished process: " + t.name)
