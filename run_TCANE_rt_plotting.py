#!/usr/bin/env python
# coding: utf-8

# In[1]:
import numpy as np
import pandas as pd
import os,glob,re,sys
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
import palettable
from scipy.stats import skewnorm, rv_histogram
import scipy.stats as sps
from utils.tcane_data_funcs import climo_to_df, read_in_TCANE 
from utils.tcane_plotting import make_TCANE_dists_pdf_cdf, make_all_plotting_data 
from utils.bdeck_edeck_funcs import get_bdecks, get_edeck_probs
from utils.tcane_track_plotting import get_plot_vars_TRACK
from utils import plots
from utils import mahalanobis
from utils.plots import plot_probability_ellipses
from utils.tcane_rt_plotting import TCANE_plots_intensity_forecasts, TCANE_plots_cat_and_RI_plots, TCANE_track_plots_with_climo, TCANE_track_plot_all, tcane_plotting_make_ALL
import cartopy as ct
import warnings
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
import scipy.stats as sps
import os,glob,sys
import cartopy.crs as ccrs

# In[2]:

import warnings
warnings.simplefilter(action='error', category=FutureWarning)
warnings.simplefilter(action='error', category=UserWarning)

def run_plot_code(in_dir,out_dir,clim_dir,bdeck_dir,storm_ID,fore_date):
    tcane_plotting_make_ALL(in_dir,out_dir,clim_dir,bdeck_dir,storm_ID,fore_date)
    return "done plotting" 

# In[3]: These are the directories that point to TCANE output, TCANE input, TCANE climo, and realtime bdeck files. These might be different 
# on different machines, but should not change from day to day. 

out_dir = '/home/mcgraw/tcane_2024/output_files/'
clim_dir = '/mnt/ssd-data1/galina/tcane/data/climo/'
in_dir = '/home/mcgraw/tcane_2024/input_files/'
bdeck_dir = '/mnt/ssd-data1/ATCF/TC-INGEST2/ATCF_RT/ATCF/dat/'

# User input: You will define storm ID and forecast date in the command line--first storm ID, then forecast date. 

if __name__ == "__main__":
    storm_ID = sys.argv[1]
    fore_date = sys.argv[2]
    print(run_plot_code(in_dir,out_dir,clim_dir,bdeck_dir,storm_ID,fore_date))




