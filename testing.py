"""This script is a selection of test and initial setups that created the
files in the cal directory"""

from hermes import HERMES
import shutil
import pdb
import numpy as np
import matplotlib.pyplot as plt
try: 
	import pyfits
except:
	import astropy.io.fits as pyfits
#Testing each CCD - analysis from scratch to get tramlines etc.
#Change this from 1 to 2 to 3 to 4. Other settings, such as the "global_offset" may also 
#have to be changed. Type "c <Enter>" after checking each plot.
ccd = '4' 
hm = HERMES('/Users/mireland/data/hermes/140310/data/ccd_'+ccd+'/', '/Users/mireland/tel/hermes/140310/ccd_'+ccd+'/', '/Users/mireland/python/hermes/cal/ccd_'+ccd+'/')

#Bias and bad pixels. Darks should really be used here...
bfiles = ['10mar'+ccd+'{0:04d}.fits'.format(i) for i in range(1,12)]
bias = hm.median_combine(bfiles,'bias.fits')
badpix = hm.find_badpix(bfiles)
shutil.copyfile(hm.rdir + 'bias.fits', hm.cdir + 'bias.fits')
shutil.copyfile(hm.rdir + 'badpix.fits', hm.cdir + 'badpix.fits')
#Tramlines for plate 0
hm.find_tramlines('10mar'+ccd+'0021.fits', global_offset=0, fillfrac=1.042)
import pdb; pdb.set_trace() #Type c after checking plot
shutil.copyfile(hm.rdir + 'tramlines_p0.txt', hm.cdir + 'tramlines_p0.txt')
hm.fit_tramlines('10mar'+ccd+'0021.fits')
shutil.copyfile(hm.rdir + 'tramlines_p0.txt', hm.cdir + 'tramlines_p0.txt')
#Tramlines for plate 1
hm.find_tramlines('10mar'+ccd+'0027.fits', global_offset=0, fillfrac=1.042)
import pdb; pdb.set_trace() #Type c after checking plot
shutil.copyfile(hm.rdir + 'tramlines_p1.txt', hm.cdir + 'tramlines_p1.txt')
hm.fit_tramlines('10mar'+ccd+'0027.fits')
shutil.copyfile(hm.rdir + 'tramlines_p0.txt', hm.cdir + 'tramlines_p0.txt')
#Arc fit for plate 0
arc, arc_sig = hm.extract(['10mar'+ccd+'0022.fits'])
wavelengths_init = hm.find_arclines(arc, pyfits.getheader(hm.ddir+'10mar'+ccd+'0022.fits'))
import pdb; pdb.set_trace() #Type c after checking plot
shutil.copyfile(hm.rdir + 'poly2d_p0.txt', hm.cdir + 'poly2d_p0.txt')
shutil.copyfile(hm.rdir + 'dispwave_p0.txt', hm.cdir + 'dispwave_p0.txt')
wavelengths = hm.fit_arclines(arc, pyfits.getheader(hm.ddir+'10mar'+ccd+'0022.fits'))
shutil.copyfile(hm.rdir + 'poly2d_p0.txt', hm.cdir + 'poly2d_p0.txt')
shutil.copyfile(hm.rdir + 'dispwave_p0.txt', hm.cdir + 'dispwave_p0.txt')
#Arc fit for plate 1
arc, arc_sig = hm.extract(['10mar'+ccd+'0028.fits'])
wavelengths_init = hm.find_arclines(arc, pyfits.getheader(hm.ddir+'10mar'+ccd+'0028.fits'))
import pdb; pdb.set_trace() #Type c after checking plot
shutil.copyfile(hm.rdir + 'poly2d_p1.txt', hm.cdir + 'poly2d_p1.txt')
shutil.copyfile(hm.rdir + 'dispwave_p1.txt', hm.cdir + 'dispwave_p1.txt')
wavelengths = hm.fit_arclines(arc, pyfits.getheader(hm.ddir+'10mar'+ccd+'0028.fits'))
shutil.copyfile(hm.rdir + 'poly2d_p1.txt', hm.cdir + 'poly2d_p1.txt')
shutil.copyfile(hm.rdir + 'dispwave_p1.txt', hm.cdir + 'dispwave_p1.txt')

#Other random tests...

#Get our extracted spectra for a wavelength test
#hm.find_tramlines('10mar10027.fits', global_offset=0)
#hm.fit_tramlines('10mar10027.fits')
#arc = hm.extract('10mar10028.fits')
#wavelengths_init = hm.find_arclines(arc, pyfits.getheader(hm.ddir+'10mar10028.fits'))
#wavelengths = hm.fit_arclines(arc, pyfits.getheader(hm.ddir+'10mar10028.fits'))

#Extract 2DF spectra... as a wavelength test.
#tdf=pyfits.getdata('/Users/mireland/data/hermes/140304/data/ccd_1/04mar10019_outdir/04mar10019ex.alligned.fits')
#tdf[np.where(tdf != tdf)] = 0
#tdfm = np.mean(tdf,axis=0)
#tdfh=pyfits.getheader('/Users/mireland/data/hermes/140304/data/ccd_1/04mar10019_outdir/04mar10019ex.alligned.fits')
#tdfl = tdfh['CRVAL1'] + tdfh['CDELT1']*(np.arange(4096)-2048)
#a = np.median(arc[150:250,:], axis=1)