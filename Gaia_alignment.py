
# coding: utf-8

# Aligning HST Data to Gaia (mostly) automatically
# ==========================

# This notebook is designed to show a workflow used to align HST images (in this case ACS/WFC and WFC3/UVIS, though other detectors should work) to catalogs queried programatically from Gaia. Since Gaia's astrometry is generally very good, this provides high-quality absolute catalogs to use to align images.  This especially helpful for cases where data is taken at many pointings with little to no overlap between images (mosaicking). 
# 
# <div class="alert alert-block alert-warning">**Note:** This is **not** a guide on running TweakReg, but rather making alignments using TweakReg substantially easier</div>
# ***

# ### Table of Contents:
# > #### 0. [Setup](#setup)
# > #### 1. [Determining Coordinates to Query](#coordinates)
# > #### 2. [Querying Catalogs from Gaia](#gaia)
# > #### 3. [Aligning Data to Gaia](#alignment)

# ***
# <a id='setup'></a>
# ## Setup
# This notebook was written using a Python 3 astroconda environment, so that is the recommended setup for running it.
# 
# In addition, you will need to install Astroquery.  The installation is very straightforward if running astroconda:
# 
# `$ conda install -c astropy astroquery`
# 
# After running that in a terminal, press `Kernel -> restart` so the package can be loaded.  More information on Astroquery can be found here: http://astroquery.readthedocs.io/en/latest/index.html
# 
# Lastly, you will need a set of data to work with.  In this notebook we use example data from Visits 01 and 05 of HST proposal 14689.  You can either retrieve the data from MAST yourself and put it in your current working directory, **or** see the [Querying MAST](#mast) section.

# All packages for entire notebook here, but also imported in first relevant cell


import glob
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil

# Astropy packages we'll need
from astropy import units as u
from astropy.coordinates.sky_coordinate import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.units import Quantity
from astropy.visualization import wcsaxes
from astropy.wcs import WCS

# Astroquery packages used for queries
from astroquery.gaia import Gaia
from astroquery.simbad import Simbad
from astroquery.skyview import SkyView

# Drizzle related packages we'll need
from drizzlepac import tweakreg
from stsci.tools import teal
from stwcs import updatewcs

# Other handy parts
from ginga.util import zscale
from multiprocessing import Pool


# Python 2 compatibility
from __future__ import print_function
from __future__ import division


# In[2]:


# Some extra configuration 

SkyView.TIMEOUT = 15

mpl.rcParams['xtick.labelsize'] = 10
plt.rcParams.update({'axes.titlesize' : '18',
                     'axes.labelsize' : '14',
                     'xtick.labelsize' : '14',
                     'ytick.labelsize' : '14'})
get_ipython().magic('matplotlib inline')


# ***
# <a id='coordinates'></a>
# ## 1. Determining Coordinates

# First, we need to create a SkyCoord object to tell the Gaia query where to point.  This can be done in a number of ways:
# >1. If coordinates are known, create the SkyCoord Object directly.
# >2. Query Simbad for the target coordinates.
# >3. Obtain coordinates using WCS information in data (recommended).
# 
# We also need to supply the search area of the query.  If we know roughly how large the field is, then this is obviously straightforward.  If not, option 3 is likely the easiest way.

#    ### a. Coordinates already known

# In[3]:


# Option 3- Determine coordinates from data
from astropy.io import fits
from astropy.wcs import WCS
from astropy.visualization import wcsaxes

from matplotlib.patches import Polygon
import matplotlib.cm as cm


# ----------------------------------------------------------------------------------------------------------

# 
def get_footprints(im_name):
    """Calculates positions of the corners of the science extensions of some image 'im_name' in sky space"""
    footprints = []
    hdu = fits.open(im_name)
    
    flt_flag = 'flt.fits' in im_name or 'flc.fits' in im_name
    
    # Loop ensures that each science extension in a file is accounted for.  This is important for 
    # multichip imagers like WFC3/UVIS and ACS/WFC
    for ext in hdu:
        if 'SCI' in ext.name:
            hdr = ext.header
            wcs = WCS(hdr, hdu)
            footprint = wcs.calc_footprint(hdr, undistort=flt_flag)
            footprints.append(footprint)
    
    hdu.close()
    return footprints

# ----------------------------------------------------------------------------------------------------------
def bounds(footprint_list):
    """Calculate RA/Dec bounding box properties from multiple RA/Dec points"""
    
    # flatten list of extensions into numpy array of all corner positions
    merged = [ext for image in footprint_list for ext in image]
    merged = np.vstack(merged)
    ras, decs = merged.T
    
    # Compute width/height
    delta_ra = (max(ras)-min(ras))
    delta_dec = max(decs)-min(decs)

    # Compute midpoints
    ra_midpt = (max(ras)+min(ras))/2.
    dec_midpt = (max(decs)+min(decs))/2.
    

    return ra_midpt, dec_midpt, delta_ra, delta_dec
# ----------------------------------------------------------------------------------------------------------

imgpath = '/astro/store/gradscratch/tmp/mdurbin/m33data/*fl?.fits'
images = glob.glob(imgpath)
# footprint_list = list(map(get_footprints, images))

# If that's slow, here's a version that runs it in parallel:
# from multiprocessing import Pool
p = Pool(8)
footprint_list = list(p.map(get_footprints, images))
p.close()
p.join()


ra_midpt, dec_midpt, delta_ra, delta_dec = bounds(footprint_list)

coord = SkyCoord(ra=ra_midpt, dec=dec_midpt, unit=u.deg)
print(coord)


# Doing this programatically also makes plotting the footprints of our images very easy.

# In[6]:


def plot_footprints(footprint_list, axes_obj=None, fill=True):
    """Plots the footprints of the images on sky space on axes specified by axes_obj """
    
    if axes_obj != None: 
        ax = axes_obj
    
    else: # If no axes passed in, initialize them now
        merged = [ext for image in footprint_list for ext in image] # flatten list of RA/Dec
        merged = np.vstack(merged)
        ras, decs = merged.T
        
        # Calculate aspect ratio
        delta_ra = (max(ras)-min(ras))*np.cos(math.radians(min(np.abs(decs))))
        delta_dec = max(decs)-min(decs)
        aspect_ratio = delta_dec/delta_ra
    
        # Initialize axes
        fig = plt.figure(figsize=[8,8*aspect_ratio])
        ax = fig.add_subplot(111)
        ax.set_xlim([max(ras),min(ras)])
        ax.set_ylim([min(decs),max(decs)])
       
        # Labels
        ax.set_xlabel('RA [deg]')
        ax.set_ylabel('Dec [deg]')
        ax.set_title('Footprint sky projection ({} images)'.format(len(footprint_list)))
        
        ax.grid(ls = ':')
    
        
    colors = cm.rainbow(np.linspace(0, 1, len(footprint_list)))
    alpha = 1./float(len(footprint_list)+1.)+.2
    
    if not fill:
        alpha =.8

    for i, image in enumerate(footprint_list): # Loop over images
        for ext in image: # Loop over extensions in images
            if isinstance(ax, wcsaxes.WCSAxes): # Check axes type
                rect = Polygon(ext, alpha=alpha, closed=True, fill=fill, 
                               color=colors[i], transform=ax.get_transform('icrs'))
            else:
                rect = Polygon(ext, alpha=alpha, closed=True, fill=fill, color=colors[i])

            ax.add_patch(rect)
    
    return ax

# ----------------------------------------------------------------------------------------------------------

plot_footprints(footprint_list)


# ***
# <a id='gaia'></a>
# ## 2. Querying Gaia (and other databases)

# With the coordinates calculated, we now only need to give the search area and perform the query.  Most astroquery supported missions/databases can be passed width/height (for a rectangular search) or a radius (for a circular search).  In either case, these parameters can be passed using astropy.units Quantity 


width = Quantity(delta_ra, u.deg)
height = Quantity(delta_dec, u.deg)


# ### a. Querying Gaia
# Performing the query is very simple thanks to Astroquery's Gaia API:

# In[8]:


# Perform the query!
r = Gaia.query_object_async(coordinate=coord, width=width, height=height)


#  

# The query returns an astropy table with the number of rows equal to the number of sources returned (with many output columns):

# In[9]:


# Print the table
# r


# Accessing the data is quite straightforward, plotting the data for quick analytics is also straightforward.

# In[10]:


ras = r['ra']
decs = r['dec']
mags = r['phot_g_mean_mag']
ra_error = r['ra_error']
dec_error = r['dec_error']

fig = plt.figure(figsize=[15,15])

# Plot RA and Dec positions, color points by G magnitude
ax1 = fig.add_subplot(221)
plt.scatter(ras,decs,c=mags,alpha=.5,s=6,vmin=14,vmax=20)
ax1.set_xlim(max(ras),min(ras))
ax1.set_ylim(min(decs),max(decs))
ax1.grid(ls = ':')
ax1.set_xlabel('RA [deg]')
ax1.set_ylabel('Dec [deg]')
ax1.set_title('Source location')
cb = plt.colorbar()
cb.set_label('G Magnitude')

# Plot photometric histogram
ax2 = fig.add_subplot(222)
hist, bins, patches = ax2.hist(mags,bins='auto',rwidth=.925)
ax2.grid(ls = ':')
ax2.set_xlabel('G Magnitude')
ax2.set_ylabel('N')
ax2.set_title('Photometry Histogram')
ax2.set_yscale("log")


ax3a = fig.add_subplot(425)
hist, bins, patches = ax3a.hist(ra_error,bins='auto',rwidth=.9)
ax3a.grid(ls = ':')
ax3a.set_title('RA Error Histogram')
ax3a.set_xlabel('RA Error [mas]')
ax3a.set_ylabel('N')
ax3a.set_yscale("log")

ax3b = fig.add_subplot(427)
hist, bins, patches = ax3b.hist(dec_error,bins='auto',rwidth=.9)
ax3b.grid(ls = ':')
ax3b.set_title('Dec Error Histogram')
ax3b.set_xlabel('Dec Error [mas]')
ax3b.set_ylabel('N')
ax3b.set_yscale("log")


ax4 = fig.add_subplot(224)
plt.scatter(ra_error,dec_error,alpha=.2,c=mags,s=1)
ax4.grid(ls = ':')
ax4.set_xlabel('RA error [mas]')
ax4.set_ylabel('Dec error [mas]')
ax4.set_title('Gaia Error comparison')
ax4.set_xscale("log")
ax4.set_yscale("log")
cb = plt.colorbar()
cb.set_label('G Magnitude')
plt.tight_layout()
plt.savefig('Gaia_errors.png')

# Filtering the data is also quite easy:



# Cut the sources

def get_error_mask(catalog, max_error):
    """Returns a mask for rows in catalog where RA and Dec error are less than max_error"""
    ra_mask = catalog['ra_error']< max_error
    dec_mask = catalog['dec_error'] < max_error
    mask = ra_mask & dec_mask
#     print('Cutting sources with error higher than {}'.format(max_error))
#     print('Number of sources befor filtering: {}\nAfter filtering: {}\n'.format(len(mask),sum(mask)))
    return mask

mask = get_error_mask(r, 10.)
# Plot RA/Dec Positions after clipping 

fig = plt.figure(figsize=[10,20])
ax1 = fig.add_subplot(211)
plt.scatter(ras[mask],decs[mask],c=mags[mask],alpha=.5,s=10,vmin=14,vmax=20)
ax1.set_xlim(max(ras),min(ras))
ax1.set_ylim(min(decs),max(decs))
ax1.grid(ls = ':')
ax1.set_xlabel('RA [deg]')
ax1.set_ylabel('Dec [deg]')
ax1.set_title('Source location (error < 10. mas)')
cb = plt.colorbar()
cb.set_label('G Magnitude')


ax2 = fig.add_subplot(212)
for err_threshold in [40., 10., 5., 2.]:
    mask = get_error_mask(r, err_threshold)
    hist, bins, patches = ax2.hist(mags[mask],bins=20,rwidth=.925,
                                   range=(10,20),label='max error: {} mas'.format(err_threshold))
ax2.grid(ls = ':')
ax2.set_xlabel('G Magnitude')
ax2.set_ylabel('N')
ax2.set_title('Photometry Histogram (No Log Scale)')
# ax2.set_yscale("log")
legend = ax2.legend(loc='best')
plt.savefig('Gaia_errors_masked.png')



# ***

# <a id='alignment'></a>
# ## 3. Aligning data to Gaia
# To align the data to Gaia, we will use DrizzlePac's TweakReg module, and pass it our catalog as an input reference catalog.
# ### a. Saving catalog
# With the Gaia catalog retrieved, we now need to save the coordinates out to a file pass them to TweakReg for alignment.


tbl = Table([ras, decs]) # Make a temporary table of just the positions
tbl.write('gaia.cat', format='ascii.fast_commented_header') # Save the table to a file.  The format argument ensures
                                                            # the first line will be commented out.


# We can also use our masking code to filter out sources before saving the final catalog:


thresh = 10.
mask = get_error_mask(r, thresh)

tbl_filtered = Table([ras[mask], decs[mask]]) 
tbl.write('gaia_filtered_{}_mas.cat'.format(thresh), format='ascii.fast_commented_header')


# We can see that this simply wrote out text files:



# ls *cat


# ### b. Running TweakReg
# Generally, this step is similar to running TweakReg in other cases.  However, we are going to pass TweakReg our Gaia catalog as a reference, rather than having it create it's own catalog from the images itself.  Note that getting good alignments still requires good source detection in the input images, so some smart parameter selection may still be necessary, depending on the input data.
# 
# Before running TweakReg, we must update the WCS's of our data (only needs to be done for flt/flc data):


input_images = sorted(glob.glob(imgpath)) 
# derp = list(map(updatewcs.updatewcs, input_images))

# Parallelized option
p = Pool(8)
derp = p.map(updatewcs.updatewcs, input_images)
p.close()
p.join()

cat = 'gaia.cat'
wcsname ='GAIA'
teal.unlearn('tweakreg')
teal.unlearn('imagefindpars')

cw = 3.5 # psf width measurement (2*FWHM).  Use 3.5 for WFC3/UVIS and ACS/WFC and 2.5 for WFC3/IR

tweakreg.TweakReg(input_images, # Pass input images
                  updatehdr=True, # update header with new WCS solution
                  imagefindcfg={'threshold':250.,'conv_width':cw},# Detection parameters, threshold varies for different data
                  separation=0.0, # Allow for very small shifts
                  refcat=cat, # Use user supplied catalog (Gaia)
                  clean=True, # Get rid of intermediate files
                  interactive=False,
                  see2dplot=False,
                  shiftfile=True, # Save out shift file (so we can look at shifts later)
                  wcsname=wcsname, # Give our WCS a new name
                  reusename=True,
                  fitgeometry='general') # Use the 6 parameter fit


# To look at how well the fit didm we can check the output shifts.txt file.
# 
# 
# The columns in the file are: xshift, yshift, rotation, scale, xrms yrms

# In[29]:


print(open('shifts.txt').read())


# The X and Y RMS are generally .1 pixels or below (if you used the example dataset), indicating a rather good fit!

# ***
# ### Conclusion
# At this point, the data is ready to be drizzled.  Better fits with even smaller residuals may be obtained by changing various parameters such as thresholds for source detection, and error budgets for Gaia sources, though the optimal set of parameters will vary between datasets.  Furthermore, the parameters shown here may not always work for other datasets, but tests using several very large datasets indicate that the performance is quite good with rough parameters.
# 
# However, trying to align single exposures is not always as easy as it was for this dataset, due to artifacts in the flts/flcs, such as cosmic rays, bad pixels, etc.  In the next notebook we present a workflow where we initially drizzle images taken in the same filter/visit (and would subsequently align those drizzled images similarly to how we did here). This method of processing is generally easier and faster, but may come at the expense of minimal levels of accuracy.
# ***

# ### More information
# **Astroquery**: http://astroquery.readthedocs.io/en/latest/index.html
# 
# **Astropy Coordinates**: http://docs.astropy.org/en/stable/coordinates/
# 
# **Astropy Tables**: http://docs.astropy.org/en/stable/table/
# 
# **Astropy Units and Quantities**: http://docs.astropy.org/en/stable/units/
# 
# **Astropy WCS**: http://docs.astropy.org/en/stable/wcs/, 
# http://docs.astropy.org/en/stable/visualization/wcsaxes/
# 
# **TweakReg**: http://stsdas.stsci.edu/stsci_python_sphinxdocs_2.13/drizzlepac/tweakreg.html,
# http://drizzlepac.stsci.edu/
