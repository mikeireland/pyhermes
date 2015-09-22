************
Introduction
************

.. highlight:: python

PyHERMES is a pure python (i.e. no IRAF) analysis package for HERMES data. It is designed to be relatively fast and
simple, and is a prototype for TAIPAN/FunnelWeb data analysis. The main directory contains the following:

 * hermes - the main module containing the analysis code
 * testing - a collection of useful test lines of codes etc.
 * cal - calibration data

Basic Algorithms
================

For each CCD, the ``go`` function does the following:

* Files in each directory are split into bias frames, arcs, flats and objects. Other files are ignored. 
* Create a bias for the night, through a median combination.
* From the list of 2dF config files used (in fits header), find the set of arcs, flats and objects for each config.
* If there are at least 1 flat, 1 arc and N object files (default N=2) then:

 * Create fiber flat.
 * Extracts arc.
 * Make a wavelength solution with ``fit_arclines``.
 * Turn multiple exposures into a data cube, and look for outlying pixels which are cosmic rays or extra bad pixels.
 * Extract each slice of the cube separately.
 * Sky subtract for each slice of the cube separately.
 * Combine the spectra, remaining in a pixel grid. Then apply a single-epoch barycentric correction.

Utility Functions
=================

The following functions should be executed from the code directory. Just typing the command gives the syntax.

``allarms`` : reduces all HERMES arms in a directory.

``listreduce`` : reduces a list of directories with HERMES arms. e.g. if you want to reduce all of 2014, type in the DATA_DIRECTORY ``ls -d 14???? > 2014.txt``, then execute ``listreduce DATA_DIRECTORY REDUCTION_DIRECTORY DATA_DIRECTORY/2014.txt data``. The last ``data`` is needed if the directories have a common subdirectory "data" containing directories like ccd_1.

``rtreduce`` : like ``allarms`` except it runs continuously, reducing data as it comes in.

Output Files
============

The *comb* output file (e.g. 10mar200020comb.fits) includes in its primary fits extension the combined spectra for a single epoch, for every fiber. The data are arranged so that in the IRAF task splot you need to type ``:dispaxis 1``.

Calibration Files
=================

The calibration directory has a list of Thorium and Xenon lines (same format as 2dFDR) and sub-directories for each CCD. Within each CCD directory, there is:

* badpix.fits: The default bad pixel map for the CCD (created with ``make_cube_and_bad``) and saved with pyfits.
* tramlines_p0.txt and tramlines_p1.txt: Tramline polynomial coefficient for the two field-plates.
* dispwave_p0.txt and dispwave_p1.txt: Dispersion and central wavelengths for each fibre.
* poly2d_p0.txt and poly2d_p1.txt: 2-dimensional polynomial coefficents for higher-order terms in the wavelength solution.

Extraction
==========

Extraction is column-based, with each fiber extracted based on a model PSF
:math:`\phi(y)`,
and the standard formula for weighted mean (e.g. Sharp and Birchall 2010) used:

.. math:: n_k = \Sigma_i ( \frac{D_i \phi_{ik}}{\sigma_i^2} ) / \Sigma_i ( \frac{\phi_{ik}^2}{\sigma_i^2} )

Rather than computing :math:`\sigma^2_i` for each pixel, a single SNR value is targeted, with:

.. math:: \sigma_i^2 \approx \phi_i + \alpha

Fitting
=======

The model is written down in a matrix form, with:

.. math:: y_i = \Sigma X_{ik} P_{i}

Here :math:`P_{ik}` is a matrix of polynomial coefficients. At each iteration, 
pixel offsets :math:`\Delta y_i` are computed via a weighted centroid, and the polynomial
coefficients are updated by solving:

.. math:: \Delta y_i = \Sigma X_{ik} \Delta P_{i}
